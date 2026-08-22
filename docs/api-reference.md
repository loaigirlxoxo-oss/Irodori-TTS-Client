# Irodori-TTS Desktop API リファレンス

Local FastAPI server runs at `http://127.0.0.1:8080` and is spawned by the
Electron app. All endpoints are unauthenticated (local-only), and the server
binds `0.0.0.0` with CORS `allow_origins=["*"]`, so it is also reachable from
other devices on the same LAN. See [external-api.md](external-api.md) for a
practical guide to calling it from outside the Electron UI.

> [!IMPORTANT]
> **ポートは固定ではありません。** Electron は起動時に 8080 から順に空きポートを探し、
> 見つかった番号を環境変数 `IRODORI_PORT` で `server.py` に渡します。
> 8080 が別のプロセスに使われていれば 8081、そこも埋まっていれば 8082 …と繰り上がります
> （8080〜8129 を確認）。
> 実際に使われているポートは次のいずれかで確認できます。
> - アプリ画面右上の **API Base URL** 表示
> - `GET /api/v1/network_info` の `port` フィールド
>
> 以下の例では既定値の 8080 を使っています。環境に合わせて読み替えてください。
> `python server.py` を直接起動した場合は `IRODORI_PORT` 未設定なので 8080 固定です。

## 共通仕様
- **Base URL**: `http://127.0.0.1:<port>`（既定 8080。上記のとおり変動する）
- **Encoding**: UTF-8 JSON (`Content-Type: application/json`)。`/api/v1/synthesize` のみ multipart にも対応。
- **エラー形式**: FastAPI 標準。`HTTP 4xx/5xx` + `{"detail": "<message>"}`（一部 `{"error": "..."}` も混在）
- **バリデーションエラー**: Pydantic モデルを使うエンドポイント（`/api/v1/loras/import`、`/api/v1/loras/merge`、`/api/v1/lora/jobs` 等）は範囲外・型不正の値で FastAPI 自動生成の `422` を返す
- **同時実行**: シングルユーザー・シングルジョブ前提。LoRA 学習中の二重実行は 409。
- **データ保存先**: `loras/` `voices/` `datasets/` `lora_jobs/` は既定で `IRODORI_DATA_DIR`（パッケージ版はインストール先の `data/`）配下。未設定時（開発時に `python server.py` を直接叩く場合など）は `APP/` 配下にフォールバックする。

---

## 1. システム

### `GET /`
ルート。エンドポイント一覧を返す診断用。

### `GET /api/v1/status`
**Response**:
```json
{
  "status": "online",
  "device": "cuda",
  "available_models": ["v1", "v2", "voice_design", "v3"],
  "registered_voices": [{"id": "voice_xxx", "name": "Marin"}, ...]
}
```

### `GET /api/v1/network_info`
LAN IP 用。OpenAI 互換クライアントから繋ぐ時の URL 生成に使う。
```json
{ "ip": "192.168.1.10", "port": 8080 }
```

---

## 2. 音声合成

### `POST /api/v1/synthesize`
（エイリアス: `/api/v1/synthesize/`, `/api/v1/synthesize/{voice_id}`）

**Content-Type**:
- `application/json` または `multipart/form-data` (`ref_wav` ファイルアップロード時)

**Body / Form fields**:
| Key | Type | Default | 説明 |
|-|-|-|-|
| `text` | string | **required** | 合成テキスト（絵文字含む可） |
| `model_type` | string | `"v2"` | `v1` / `v2` / `voice_design` / `v3` |
| `voice_id` | string | `null` | 登録済みボイスID（metadata.json から path を解決） |
| `ref_wav` | file | `null` | (multipart のみ) 参照音声を直接アップ |
| `caption` | string | `null` | `model_type=voice_design` 時のスタイル指定 |
| `lora_name` | string | `null` | 登録済み LoRA 名。指定時は `lora_adapter` パスに変換 |
| `num_steps` | int | `40` | Euler サンプリングステップ |
| `num_candidates` | int | `1` | 同時生成数 |
| `seed` | int | `null` | 再現用シード |
| `cfg_guidance_mode` | string | `"independent"` | `independent` / `joint` / `alternating` |
| `cfg_scale_text` | float | `2.0` | CFG (text) |
| `cfg_scale_speaker` | float | `3.0` | CFG (speaker) |
| `cfg_scale_caption` | float | `4.0` | CFG (caption) |
| `cfg_scale` | float | `null` | 旧API互換オーバーライド |
| `cfg_min_t` / `cfg_max_t` | float | `0.5` / `1.0` | CFG 適用 t 範囲 |
| `context_kv_cache` | bool | `true` | KVキャッシュON/OFF |
| `truncation_factor` / `rescale_k` / `rescale_sigma` | float | `null` | 上級 |
| `speaker_kv_scale` / `speaker_kv_min_t` / `speaker_kv_max_layers` | mixed | `null` / `0.9` / `null` | 話者埋め込み強度制御 |
| `max_text_len` / `max_caption_len` | int | `null` | 切詰めトークン数 |
| `t_schedule_mode` | string | `"linear"` | `linear` / `sway` |
| `sway_coeff` | float | `-1.0` | Sway係数 |

> **注意（`0`指定の落とし穴）**: `num_steps` / `num_candidates` / `cfg_scale_text` / `cfg_scale_speaker` / `cfg_scale_caption` / `cfg_min_t` / `cfg_max_t` / `speaker_kv_min_t` はサーバー側で `parse_x(...) or <default>` というパターンで処理されるため、**明示的に `0` や `0.0` を渡してもデフォルト値に置き換わる**（Pythonの falsy 判定のため）。`0` を本当に使いたい場合はこれらのフィールドでは不可。（`cfg_scale` だけはこのフォールバックが無く `0` を素通しする）

**v1 専用挙動**: `model_type=v1` の場合、サーバーは `../Irodori-TTS-v1/` worktree の `infer.py` をサブプロセス実行する。LoRA は v1 では非対応で、解決できた `lora_name` は無視される。ただし **`lora_name` の存在チェック自体は v1 でも行われる**ため、未登録の名前を指定すると v1 でも `400` になる。

**Response (success)**:
```json
{
  "status": "success",
  "seed_used": 42,
  "results": ["/api/v1/outputs/sample_20260520_xxx_001.wav"],
  "timings": ["tokenize_text: 1.4ms", "sample_rf: 800ms", ...]
}
```

**Errors**:
- `400` text 空 / LoRA 名が未登録
- `500` 不正な `model_type` を含むランタイム例外全般（subprocess失敗、モデル読込失敗など）

### `GET /api/v1/outputs/{filename}`
合成済 wav の取得。`results[]` の URL を直接叩く。
**Response**: `audio/wav` バイナリ。 **Errors**: `404 {"error": "File not found"}`

### `DELETE /api/v1/outputs/{filename}`
合成済 wav を削除。
**Response**: `{"status": "ok"}` / `500 {"error": "..."}`（OSError時）

---

## 3. Voice Library

### `GET /api/v1/voices`
登録済 voice 一覧。data は `voices/metadata.json` から。
```json
{ "voices": [{ "id": "voice_xxx", "name": "Marin", "path": "...wav", "created_at": "..." }, ...] }
```

Voice の **追加・削除** は Electron IPC で行う（API公開していない）：
- `window.api.addVoice({name, filePath})` → `voices/voice_<id>.wav` に copy
- `window.api.deleteVoice(id)` → ファイル削除

---

## 4. LoRA Registry

### `GET /api/v1/loras`
登録済 LoRA 一覧。
```json
{
  "loras": [
    { "name": "Marin", "base": "v3", "imported_at": "...", "source": "...", "notes": "..." },
    ...
  ]
}
```

### `POST /api/v1/loras/import`
外部 PEFT アダプタディレクトリを取り込む（Manage LoRAs モーダルから使用）。

**Body**:
```json
{
  "name": "Marin",
  "base": "v3",
  "source_path": "D:/path/to/checkpoint_xxxx",
  "notes": ""
}
```
- `name`: 1-64文字。`[A-Za-z0-9_\-. 一-龥ぁ-んァ-ヶー]`（半角英数・`_-.`・半角スペース・日本語〈ひらカナ漢〉）
- `base`: `v3` / `v2` / `voice_design`
- `source_path`: `adapter_config.json` と `adapter_model.safetensors`（または `adapter_model.bin`）を含む dir

**Response**: `{ "status": "ok", "name": "...", "base": "..." }`
**Errors**: `400` 不正 name / 不正 base / source_path 不存在 / adapter 形式不正 / `409` 同名重複 / `500` アダプタコピー失敗

### `DELETE /api/v1/loras/{name}`
**Response**: `{ "status": "ok", "name": "..." }` / `404` not found

### `POST /api/v1/loras/merge`
複数 LoRA を重み付き合成して新しい LoRA を作る（Manage LoRAs のマージ機能から使用）。

**Body**:
```json
{
  "name": "Marin_x_Akari",
  "sources": [{"name": "Marin", "weight": 0.6}, {"name": "Akari", "weight": 0.4}],
  "method": "linear",
  "notes": ""
}
```
- `sources`: 2〜3個。**全て同じ `base`** であること。`weight` は自動的に合計1へ正規化される
- `method`: `linear`（層ごとの `ΔW=B@A` を加重平均しSVDで再分解）/ `slerp`（球面線形補間・`sources` がちょうど2個の時のみ）
- 出力 `name` は既存 LoRA 名・source 名のいずれとも重複不可
- マージ対象は `adapter_model.safetensors` のみ（`.bin` はインポート時と異なり非対応）

**Response**: `{ "status": "ok", "name": "...", "base": "...", "keys_merged": N }`
**Errors**:
- `400` sources 数が2〜3の範囲外 / `slerp` で sources≠2 / 出力名が source と重複 / 合計weightが0以下 / source に safetensors 無し / base 不一致
- `404` source LoRA 不存在
- `409` 出力名が既存 LoRA と重複

---

## 5. Audio Processing

### `POST /api/v1/audio/split`
Silero-VAD で wav を speech chunk に分割。staging dir に書き出す。

**Body**:
```json
{
  "path": "D:/abs/path.wav",
  "min_sec": 3.0,
  "max_sec": 20.0,
  "speech_pad_ms": 400,
  "min_silence_ms": 500,
  "output_sample_rate": 48000
}
```

**Response**:
```json
{
  "chunks": [
    { "index": 0, "path": "...chunk_0000.wav", "start": 0.0, "end": 5.7, "duration": 5.7 },
    ...
  ],
  "source": "...",
  "staging_dir": "..."
}
```
発話が検出できなかった場合は `{"chunks": [], "source": "..."}` のみが返り、`staging_dir` キー自体が存在しない。

**Errors**: `400` 指定 path が存在しない/ファイルでない

### `POST /api/v1/audio/transcribe`
litagin/anime-whisper (transformers 直叩き) で1ファイル転記。

**Body**:
```json
{
  "path": "D:/abs/path.wav",
  "language": "Japanese",
  "no_repeat_ngram_size": 0,
  "repetition_penalty": 1.0
}
```

**Response**:
```json
{
  "text": "今日はいい天気ですね。",
  "segments": [],
  "duration": 4.2,
  "language": "ja",
  "language_probability": 1.0
}
```

**注**: 公式パラメータ準拠（`initial_prompt` は意図的に未公開、ハルシネーション誘発のため）。

**Errors**: `400` 指定 path が存在しない/ファイルでない

---

## 6. Datasets

### `GET /api/v1/datasets`
取込済データセット一覧。
```json
{
  "datasets": [
    { "name": "Marin", "num_clips": 753, "location": "...", "created_at": "...", "notes": "..." },
    ...
  ]
}
```

外部 location 保存も含む（`_locations.json` レジストリ参照）。

### `GET /api/v1/datasets/{name}`
詳細 + clips 全部。
```json
{
  "meta": { "name": "...", "num_clips": 753, ... },
  "clips": [
    { "index": 1, "wav_path": "...", "text": "...", "duration": 3.4 },
    ...
  ]
}
```
**Errors**: `404` dataset 不存在

### `POST /api/v1/datasets`
新規作成。クリップ wav と転記テキストを bundle する。

**Body**:
```json
{
  "name": "Marin",
  "clips": [
    { "path": "<staging chunk path>", "text": "セリフ1" },
    { "path": "...", "text": "セリフ2" },
    ...
  ],
  "source_files": ["D:/raw/long.wav", ...],
  "staging_dirs": [],
  "notes": "",
  "overwrite": false,
  "target_dir": null
}
```

- `target_dir` 指定: そのフォルダ配下に `<target_dir>/<name>/` で作成 + レジストリ登録
- `target_dir=null`: デフォルト場所（`IRODORI_DATA_DIR/datasets/<name>/`。開発時フォールバックでは `APP/datasets/<name>/`）に作成
- `source_files`: 元音声への**パスを `meta.json` に記録するのみ**。`<dataset>/source_audio/` へのコピーは行わない
- `staging_dirs`: 保存完了後に削除する VAD ステージングディレクトリのリスト
- `overwrite=true`: 同名既存を削除して上書き

**Response**: `{ "status": "ok", "name": "...", "num_clips": N, "location": "<abs>" }`
**Errors**:
- `400` `target_dir` が存在しない/ディレクトリでない / clip の元ファイルが見つからない
- `409` 同名データセットが既存（`overwrite` なし）
- `500` クリップ保存失敗

### `GET /api/v1/datasets/{name}/auto_config`
学習推奨パラメータを返す（Train タブの Auto Setting ボタンが使用）。

**Response**:
```json
{
  "name": "Marin",
  "num_clips": 753,
  "avg_duration": 3.34,
  "total_duration": 2547.95,
  "recommended": {
    "max_steps": 700,
    "save_every": 100,
    "preset": "standard",
    "target_epoch": 30
  }
}
```

計算式：
- `max_steps = round(num_clips × 30 / 32)` を丸め幅で整形。丸め幅は `num_clips` の規模で変わる（< 200 → 10刻み / < 1000 → 50刻み / < 5000 → 100刻み / それ以上 → 500刻み）
- `save_every = round(num_clips × 5 / 32)`（同様に丸め）。最小10、かつ `max_steps` を超えないようキャップされる
- `preset = light if <500 / standard if 500-3000 / broad if >3000`

**Errors**: `404` dataset 不存在 / `400` `clips/` ディレクトリが無い / `400` 読める clip が0件

### `PUT /api/v1/datasets/{name}/clips/{index}`
クリップのテキストを更新。
**Body**: `{ "text": "新しい転記" }`
**Errors**: `404` dataset 不存在 / clip 不存在

### `DELETE /api/v1/datasets/{name}/clips/{index}`
1クリップ削除。
**Errors**: `404` dataset 不存在 / clip 不存在

### `DELETE /api/v1/datasets/{name}`
データセット丸ごと削除（外部 location も含む）。

---

## 7. Training Jobs

### `POST /api/v1/lora/jobs`
LoRA 学習ジョブ起動。

**Body**（例。値はデータセットに応じて調整。既定値は下記参照）:
```json
{
  "lora_name": "Marin",
  "dataset": "Marin",
  "base": "v3",
  "preset": "standard",
  "max_steps": 700,
  "save_every": 100,
  "batch_size": 4,
  "gradient_accumulation_steps": 8,
  "log_every": 50,
  "learning_rate": null,
  "overwrite": false
}
```

**フィールド既定値**: `base="v3"` / `preset="standard"` / `max_steps=2000` / `save_every=1000` / `batch_size=4` / `gradient_accumulation_steps=8` / `log_every=50` / `learning_rate=null`（config既定の`1e-4`）/ `overwrite=false`

- `base`: `v3` / `v2` / `voice_design`
- `preset`: `light` (`diffusion_attn`) / `standard` (`diffusion_attn_mlp`) / `broad` (`all_attn_mlp`)
- `overwrite=true`: 同名既存 LoRA を**学習成功時に**上書き（事前削除はしない、安全）
- `learning_rate=null`: YAML config のデフォルト使用
- バリデーション範囲: `max_steps` 10〜200000 / `batch_size` 1〜128 / `gradient_accumulation_steps` 1〜64 / `save_every` ≥10 / `log_every` ≥1 / `learning_rate` >0（範囲外は `422`）

**Response**: `{ "status": "ok", "job_id": "<12hex>" }`
**Errors**:
- `400` 不正な base/preset / dataset に `clips/` が無い
- `404` dataset 不存在
- `409` LoRA 同名既存 (overwrite なし) / 他ジョブが active
- `422` フィールドが上記バリデーション範囲外

### `GET /api/v1/lora/jobs`
全ジョブ一覧（active + 完了）。
```json
{
  "jobs": [
    { "id": "...", "name": "...", "state": "...", "current_step": 1234, "max_steps": 2050,
      "current_loss": 0.234, "created_at": "...", "finished_at": "..." },
    ...
  ]
}
```

### `GET /api/v1/lora/jobs/{job_id}`
詳細 + ログ末尾。フロントが 2 秒ごとに polling。
```json
{
  "id": "...", "name": "...", "dataset": "...", "base": "...", "preset": "standard",
  "max_steps": 700, "state": "training",
  "current_step": 234, "current_loss": 0.617,
  "created_at": "...", "started_at": "...", "finished_at": null,
  "log_tail": ["step=200 loss=0.6...", ...],
  "error": null, "pid": 12345,
  "updated_at": "..."
}
```
`registered_as` キーは `state="done"`（学習成功・LoRA登録済み）になって初めて追加される。それ以外の状態（`pending`/`preparing`/`training`/`stopping`/`failed`/`stopped`）では**キー自体が存在しない**（`null` ではなく欠落）。

**state 値**: `pending` / `preparing` (マニフェスト構築中) / `training` / `stopping` / `done` / `failed` / `stopped`

### `POST /api/v1/lora/jobs/{job_id}/stop`
進行中ジョブを停止。Windows なら `taskkill /F /T /PID`、Unix は `os.killpg(SIGTERM)`。

**Response**:
- 停止処理を行った場合: `{ "status": "ok", "job_id": "..." }`
- 停止可能な状態でなかった場合: `{ "status": "noop", "state": "<現在の state>" }`（`job_id` は含まれない）

**Errors**: `404` job 不存在 / `500` プロセス強制終了失敗

**制限**: マニフェスト構築フェーズ（in-process Python、DACVAEエンコード中）は subprocess kill による即時停止はできないが、10クリップごとに停止要求をチェックしており、次のチェックポイントで `stopped` になる（数クリップ分のタイムラグあり）。training フェーズ以降は subprocess kill で即時停止。

---

## 8. OpenAI 互換

### `GET /v1/models`
（エイリアス: `/v1/models/`）
OpenAI Speech クライアントが叩く一覧。`tts-1`, `tts-1-hd` 等の標準ID + 登録 voice 名を返す。

### `POST /v1/audio/speech`
（エイリアス: `/v1/audio/speech/`, `/audio/speech`, `/`）

**Body** (OpenAI互換):
```json
{ "input": "...", "voice": "Marin", "model": "tts-1" }
```

`voice` または `model` が登録済 voice の **`id` または `name`** のいずれかと一致すれば、そのリファレンス音声で合成。
**Response**: `audio/wav` バイナリを直接返却。合成結果は通常の synthesize と同様に `outputs/openai_<timestamp>.wav` として**保存もされる**（`GET /api/v1/outputs/{filename}` から後で再取得可能）。

**Errors**: `400` JSON不正 / `input` 欠落 / `500` 生成失敗・ランタイム例外

**制限**:
- 常に `model_type="v2"` 固定
- LoRA 非対応（必要なら拡張可）
- caption / sway / その他高度パラ非対応

---

## 9. レート制限・並行性
- レート制限なし
- 同時 synthesize リクエストは InferenceRuntime キャッシュにより**逐次実行**（並列処理時の VRAM 衝突回避）
- LoRA 学習中は他ジョブ POST が 409
- LoRA 学習中の synthesize は VRAM 圧迫で OOM の可能性あり（要警告）

---

## 10. 既知の制限
| 機能 | 制限 |
|-|-|
| 同時 LoRA 学習 | 1 ジョブのみ |
| マニフェスト構築の途中停止 | 即時ではない（10クリップごとのチェックポイントで反映、DACVAEエンコード中は数クリップ分ラグあり） |
| 学習用配信フォーマット | wav のみ（mp3/flac は事前変換が必要） |
| Whisper モデル | `litagin/anime-whisper` 固定（ハードコード） |
| ベース TTS モデル | v1/v2/voice_design/v3 のみ |

---

## 11. 関連ドキュメント
- [external-api.md](external-api.md) — 外部（別スクリプト・別端末）から呼び出す実用ガイド
- [desktop-app.md](desktop-app.md) — エンドユーザー向けセットアップ・操作
- [lora-usage.md](lora-usage.md) — LoRA インポート・運用
- [parameters.md](parameters.md) — 推論/学習パラメータの詳細（上流）
- [plans/2026-05-15-lora-import-design.md](plans/2026-05-15-lora-import-design.md) — LoRA 実装設計
