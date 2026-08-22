# Irodori-TTS デスクトップアプリ — LoRA 使い方ガイド

LoRA（Low-Rank Adaptation）を使うと、**特定キャラの声・話し方・スタイル**を学習させて、推論時に切替えながら使えます。

LoRAの学習は **アプリ内蔵の `Train` タブで完結します（推奨）**。データセット作成から学習・ジョブ監視・学習済みLoRAの登録まで、外部ツールなしでこのアプリだけで行えます。
外部で学習したPEFTアダプタ（[iron-mukakin/Emoji-TTS](https://github.com/iron-mukakin/Emoji-TTS) 等）を持ち込んでインポートする使い方も引き続き可能です。

---

## 1. LoRA を使う（登録済みの場合）

1. アプリ起動（`起動.bat` ダブルクリック）
2. メイン画面右上の **`Manage LoRAs`** クリック → LoRA Library モーダルが開く
3. 登録済みなら一覧表に並んでいる（自分で学習したものも、インポートしたものも同じ一覧に出る）
4. モーダルを閉じる
5. モデルdropdownでベース選択（例：`v3 (Latest)`）
6. その下のラジオを **`LoRA`** に切替
7. LoRA dropdown が表示される → 使いたいLoRAを選択
8. テキスト入力 → Generate

ベースモデルと**互換性のないLoRA**は dropdown に出ません（v3 ベースを選んでいる時は v3 用LoRAだけ表示）。

---

## 2. アプリ内で LoRA を学習する（`Train` タブ・推奨）

### 2.1 データセットを作る（`Dataset` タブ）

1. 音声ファイルを追加（Browse files / Browse folder〈フォルダ内を5階層まで再帰スキャン〉/ ドラッグ&ドロップ）
2. **Process** を実行 → Silero-VAD が発話チャンクに自動分割し、続けて Anime-whisper（`litagin/anime-whisper`）が各チャンクを自動文字起こし
   （初回はWhisperモデル約1.5GBのダウンロードが走る）
3. 誤字があればクリップごとにテキスト編集、不要なクリップは除外チェック
4. データセット名を付けて **Save**（`APP/datasets/<name>/` に `meta.json` + `clips/`(wav+txt) + `source_audio/` として保存。既存の外部保存先を指定することも可能）
5. 同名で保存しようとすると `409`（`overwrite` を明示すれば上書き可）

### 2.2 学習を開始する（`Train` タブ）

| 項目 | 内容 |
|-|-|
| **Dataset** | 上で作ったデータセットを選択 |
| **LoRA Name** | 空欄ならデータセット名がそのまま使われる |
| **Base Model** | `v3` / `v2` / `Voice Design` |
| **Preset** | `Light`＝`diffusion_attn`（軽量）／`Standard`＝`diffusion_attn_mlp`（**推奨・既定**）／`Broad`＝`all_attn_mlp`（拡散バックボーンに加えテキスト/キャプション/話者エンコーダも学習対象。効果は強いがデータが少ないと過学習しやすい） |
| **Max Steps** | 既定 2000。目安は「典型 2000〜5000」だが、データ量に応じて `Auto Setting` の値を使うのが確実 |
| **Save Every** | 既定 500 |
| **Batch Size** | 既定 4 |
| **Gradient Accumulation Steps** | 既定 8 |
| **Learning Rate** | 空欄ならconfigの既定値（`1e-4`） |

**Auto Setting** ボタンを押すと、選択中データセットのクリップ数・総尺から以下を自動提案する：
- `max_steps ≈ クリップ数 × 30 / 32`（キリの良い値に丸め）
- `save_every ≈ クリップ数 × 5 / 32`
- `preset`：クリップ数 500未満 → `light` ／ 3000超 → `broad` ／ それ以外 → `standard`

**Start Training** を押すとジョブが始まる（同時に走らせられるのは1ジョブのみ。実行中に別ジョブを開始しようとすると `409`）。

### 2.3 進捗を見る

- 2秒ごとに自動ポーリングし、状態（`pending → preparing → training → done/failed/stopped`）・現在ステップ・現在loss・ログ末尾を表示
- **Stop** で中断可能。ただしマニフェスト構築（DACVAE潜在の事前計算）フェーズ中は即時停止できない場合がある
- 完了すると、出力先の最新 `checkpoint_*` が**自動的にLoRAとして登録される**（`notes` に学習ステップ数・base・presetが記録される）。手動インポートは不要——そのまま Synthesize タブや `Manage LoRAs` の一覧に現れる

---

## 3. LoRA を手動でインポートする（外部で学習したアダプタを取り込む場合）

1. メイン画面右上の **`Manage LoRAs`** クリック
2. **`Pick adapter folder…`** ボタン
3. PEFTアダプタディレクトリを選択。中に以下のファイルがあること:
   - `adapter_config.json`
   - `adapter_model.safetensors`（または `adapter_model.bin`）
4. フォームに入力:
   - **Name**: 表示名（例：`Marin`、`Akari`）。アプリ内で一意
   - **Base**: 学習に使ったベースモデル（v3 / v2 / Voice Design）
   - **Notes**: 任意のメモ（学習データ量・ステップ数・特徴等）
5. **`Import`** クリック
6. 一覧表に追加され、すぐに Synthesize タブで使える

ファイルは `%APPDATA%\Irodori-TTS\loras\<name>\adapter\` にコピーされます。

---

## 4. 外部ツール（Emoji-TTS）で学習する場合（上級者向け）

このアプリの `Train` タブで足りるなら本節は不要です。別環境で学習したい場合や、[iron-mukakin/Emoji-TTS](https://github.com/iron-mukakin/Emoji-TTS)（Irodori-TTSフォーク）のWebUIを使いたい場合の手順です。

### 前提
- Python 3.10〜3.12
- `uv` パッケージマネージャ（`pip install uv` で導入可）
- GPU（最低 8GB VRAM、推奨 12GB+）
- 学習用音声データ：1キャラあたり 30 分〜 数時間（多いほど良い）

### セットアップ
```powershell
git clone https://github.com/iron-mukakin/Emoji-TTS
cd Emoji-TTS
uv sync
```

### Web UI 起動
```powershell
uv run python gradio_app.py
```
ブラウザで `http://localhost:7860` を開く。

### 学習フロー（Emoji-TTS の WebUI 上で）

1. **データ作成タブ**
   - `Slice`: 長尺音声を Silero VAD で発話チャンクに自動分割
   - `Caption`: Whisper で各チャンクを文字起こし → CSV/JSONL 出力
   - 必要なら手動で書き起こし誤りを修正
2. **マニフェスト準備タブ**
   - データセット CSV/JSONL を読み込んで DACVAE 潜在を事前計算
3. **LoRA学習タブ**
   - ベースモデル選択（v3 推奨）
   - LoRAパラメータ設定:
     - `rank`: 16（デフォルト）— 表現力 vs サイズのバランス
     - `target_modules`: `diffusion_attn`（軽量）/ `diffusion_attn_mlp`（**推奨・標準**）/ `all_attn_mlp`（テキスト/キャプション/話者エンコーダも含む・広範囲）
     - `learning_rate`: 1e-4（デフォルト）
     - `max_steps`: データ量に比例させるのが目安（本アプリの `Auto Setting` と同じ考え方＝クリップ数 × 30/32 ステップ程度から様子見）
   - **Start Training** → ログとloss曲線で進捗確認
4. 学習完了 → `outputs/<run_name>/checkpoint_NNNNN/` に PEFT アダプタが出力される

### このアプリにインポート
上記「3. LoRA を手動でインポートする」の手順で `checkpoint_NNNNN/` を取り込む。

### 学習中に VRAM 不足エラー
本アプリの `Train` タブ・Emoji-TTS どちらも同じ `train.py` ベースの学習器を使っているため、対処は共通:
- `batch_size` を下げる（既定4 → 2 等）
- `gradient_accumulation_steps` を上げる（既定8 → 16 等）で実効バッチを維持
- `max_latent_steps`（`--max-latent-steps`）を下げる（既定750 → 400等）

---

## 5. LoRA をマージする

複数のLoRAを重み付き合成して新しいLoRAを作れます（`Manage LoRAs` のマージ機能 / `POST /api/v1/loras/merge`）。

- **sources**: 2〜3個の `{name, weight}`。**同じbaseのLoRA同士のみ**マージ可能
- **method**: `linear`（加重平均）または `slerp`（球面線形補間・sourcesがちょうど2個の時のみ）
- weightは自動的に合計1に正規化される
- 出力名は既存LoRA名・source名と重複不可

```bash
curl -X POST http://localhost:8080/api/v1/loras/merge \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Marin_x_Akari",
    "sources": [{"name": "Marin", "weight": 0.6}, {"name": "Akari", "weight": 0.4}],
    "method": "linear",
    "notes": "half-and-half blend"
  }'
```

---

## 6. パラメータ別の効きどころ

LoRA を有効にしたとき、各 CFG スライダーの**意味合いが少し変わります**:

| パラメータ | LoRA 無効時 | LoRA 有効時 |
|-|-|-|
| **Text CFG** (3.0) | テキストへの忠実度 | 同じ（LoRA に影響されない） |
| **Speaker CFG** (5.0) | リファレンス音声への忠実度 | **LoRA が学習した話者特徴への忠実度** |
| **Caption CFG** | VoiceDesign 用 | VoiceDesign + LoRA でも同じ |

**LoRA モード時はリファレンス音声欄が消えます**。LoRA 自体が話者情報を内包しているため不要です。

### LoRA の効きが弱いとき
- **Speaker CFG を上げる**（5.0 → 6〜7）
- 学習側で `lora_alpha` を上げる（再学習が必要）
- Preset を `Standard`（`diffusion_attn_mlp`）や `Broad`（`all_attn_mlp`）に広げて再学習
  - `Broad` はテキスト/キャプション/話者エンコーダも学習対象に含むため、声質だけでなく間の取り方・抑揚まで変化しうる点に注意

### LoRA の効きが強すぎる（不自然・歪む）
- **Speaker CFG を下げる**（5.0 → 3〜4）
- 学習側で過学習している可能性 → より早いチェックポイント（例：`checkpoint_5000/` 等）を試す

---

## 7. トラブルシューティング

### LoRA dropdown が空
- 選択中のベースモデルと一致する LoRA が登録されていない
- Manage LoRAs を開いて Base 列を確認

### Import で「adapter_config.json not found」系エラー
- 実際のメッセージ: `source_path must contain adapter_config.json and adapter weights (adapter_model.safetensors or adapter_model.bin)`
- 選択したフォルダが正しい PEFT 出力ディレクトリではない
- 1階層深い `checkpoint_NNNNN/` を選び直す

### 音質が突然劣化する / 文字化けする
- ベースモデルと LoRA の組合せが不一致（v3 LoRA を v2 で使うなど）
- 解決：正しいベースに切り替える、または LoRA を再インポートして Base を正しく設定

### Manage LoRAs ボタンが効かない / モーダルが開かない
- DevTools コンソールを開いて（Ctrl+Shift+I）エラー確認
- 画面右上の **API Base URL** に出ているポートでサーバー生存確認
  （ポートは 8080 固定ではなく、起動時に空きを探して決まる）

### 学習中に VRAM 不足エラー
「4. 外部ツール（Emoji-TTS）で学習する場合」末尾の VRAM 対処を参照（`Train` タブ・Emoji-TTS共通）。

---

## 8. ファイル配置（参考）

```
%APPDATA%\Irodori-TTS\
├── loras\
│   └── Marin\
│       ├── adapter\
│       │   ├── adapter_config.json
│       │   └── adapter_model.safetensors
│       └── meta.json      # name, base, imported_at, source, notes
└── lora_jobs\              # Trainタブで開始した学習ジョブのログ・状態
    └── <job_id>\
        ├── status.json
        ├── training.log
        └── train_output\   # 学習中/完了後のcheckpoint_*（完了時にlorasへ登録コピー）
```

LoRA を完全削除したい場合は Manage LoRAs の **×** ボタンを使う（ディレクトリごと消える）。

---

## 9. API 経由で使う

このアプリは外部からも HTTP API でアクセス可能（既定 `http://localhost:8080`。
ポートは起動時に空きを探して決まるため、実際の値は画面右上の **API Base URL** を参照）。
LAN越し・別スクリプトからの呼び出し方の詳細ガイドは [external-api.md](external-api.md) を参照:

```bash
# LoRA 一覧
curl http://localhost:8080/api/v1/loras

# LoRA 付き合成
curl -X POST http://localhost:8080/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"こんにちは","model_type":"v3","lora_name":"Marin","num_steps":32}'

# LoRA インポート（CLI からも可）
curl -X POST http://localhost:8080/api/v1/loras/import \
  -H "Content-Type: application/json" \
  -d '{"name":"Marin","base":"v3","source_path":"D:/path/to/checkpoint_10000","notes":"30min 10kstep"}'

# LoRA 削除
curl -X DELETE http://localhost:8080/api/v1/loras/Marin

# LoRA 学習ジョブ起動（Trainタブと同じ機能をAPI経由で）
curl -X POST http://localhost:8080/api/v1/lora/jobs \
  -H "Content-Type: application/json" \
  -d '{"lora_name":"Marin","dataset":"Marin","base":"v3","preset":"standard","max_steps":2000}'
```

OpenAI 互換エンドポイント `/v1/audio/speech` は現状 LoRA 非対応です（必要なら拡張可能）。
