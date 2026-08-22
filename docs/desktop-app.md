# Irodori-TTS デスクトップアプリ

Electron 製の GUI。Irodori-TTS の合成・LoRA 学習・データセット作成までを1アプリで完結。

> **配布バイナリは提供していません**。下記セットアップを各自で実施してください（ML依存が15GB級のため、配布パッケージ化は現状無し）。

## セットアップ（初回のみ）

### 必要なもの
- **Python 3.10**（必須。`pyproject.toml` は `>=3.10` としか書いていないが、3.11 以降では
  `sentencepiece` の wheel が無くソースビルドに落ち、CMake の非互換で失敗する。
  `setup.bat` は `py -3.10` を明示的に選ぶ）
- **Node.js 18+**（Electron 用）
- **uv** （Python パッケージマネージャ） — `pip install uv`
- **CUDA 12.x 対応 GPU**（推奨 VRAM 12GB+。RTX 3060 12GB / 3080 / 4070 級〜）
- **Git**

### 手順

```powershell
# 1. リポジトリ取得
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS

# 2. Python 環境構築（CUDA 12.8 ビルドが自動で入ります）
uv sync

# 3. デスクトップアプリ用の追加依存（FastAPI / Silero VAD 等）
.venv\Scripts\python.exe -m pip install -r APP\requirements.txt

# 4. Electron 側の依存
cd APP
npm install
cd ..

# 5. モデル重みを事前ダウンロード（任意・1回6GB）
.venv\Scripts\python.exe scripts\download_models.py
```

### 動作確認
```powershell
# Python サーバー単体テスト
.venv\Scripts\python.exe APP\server.py
# 別ターミナルで:  curl http://127.0.0.1:8080/api/v1/status
# 終わったら Ctrl+C で停止
```

## 起動

```text
起動.bat   ← ダブルクリック
```

中で `npm start` → Electron 起動 → 内部で FastAPI サーバー（`server.py`）を立ち上げます。

ポートは固定ではありません。Electron が 8080 から順に空きを探し、`IRODORI_PORT` で
`server.py` に渡します。実際のポートは画面右上の **API Base URL** に表示されます。

ウィンドウ上部にタブが7つ：

| タブ | 役割 |
|-|-|
| **Synthesize** | テキスト→音声合成、LoRA切替 |
| **Dataset** | 生音声→自動分割→転記→データセット保存 |
| **Train** | データセットから LoRA を学習、ジョブ監視 |
| **朗読** | テキストファイルを読み込んで連続合成、しおり機能 |
| **青空文庫** | 著作権切れ作品を検索・取得して朗読タブへ渡す |
| **LoRAマージ** | 複数の LoRA を比率指定で合成 |
| **辞書** | 読み間違いの矯正（朗読・Synthesize 共通） |

---

## Synthesize タブ

### モデル選択
| 選択肢 | 用途 |
|-|-|
| `v3 (Latest)` | 推奨。Duration Predictor で長さ自動 |
| `Normal (v2)` | 旧base |
| `Voice Design (v2)` | キャプションで声質指定（参照音声不要） |
| `Legacy (v1, subprocess)` | 旧モデル（参考まで） |

### 1ショット vs LoRA
モデル選択の下にラジオボタン：
- **1ショット**：左サイドの Voice Library で選んだ参照音声で zero-shot クローン
- **LoRA**：学習済み LoRA を適用（参照音声欄は消える）。base モデルと互換な LoRA だけ dropdown に出る

### Sampling パラメータ
- `Num Steps`：40 (推奨)。Sway 有効時は 16〜24 でOK
- `Text/Speaker/Caption CFG`：各条件への忠実度
- `Time Schedule = sway` + `Sway Coeff = -1.0` で速度2.6倍に
- 詳細は [parameters.md](parameters.md) を参照

### 絵文字パレット
v3 公式 45 個（囁き 👂 / 笑い 🤭 / 喘ぎ 🥵 / 朗読 📖 ...）。
クリックでテキスト挿入。各ボタンの tooltip に細かいニュアンス説明あり。

### Manage LoRAs
右上のボタンでモーダル。外部学習した PEFT アダプタの **インポート / 削除** ができる。
詳細：[lora-usage.md](lora-usage.md)

---

## Dataset タブ

### 1. 音声の追加
3通り：
- **Browse files**：複数 wav を選択
- **Browse folder**：フォルダ内 `*.wav` を5階層まで再帰
- **ドラッグ&ドロップ**：wav ファイル / フォルダどちらでも

### 2. 分割 + 自動転記
`Min length` / `Max length` を調整して **Process** ボタン：
- Silero-VAD で `Min`〜`Max` 秒のチャンクに分割
- 各チャンクを **Anime-whisper** で日本語転記（初回起動時にモデル ~1.5GB DL）
- 結果がテーブルに表示される

### 3. クリップの編集
- テキスト入力欄で直接修正
- チェックボックスで除外切替（除外したクリップは保存対象外）
- 各クリップに音声プレーヤー付き

### 4. 保存
- `Dataset name` を入力
- `Save location...` で**任意のフォルダ**を指定可（未指定なら APP データ領域）
- **Save dataset** クリック
- 完了表示に**絶対パス**が出る → エクスプローラーで確認可

### データセットの中身
```
<保存先>/<dataset_name>/
├── meta.json
├── clips/         # 分割クリップ wav + txt ペア
├── source_audio/  # 元音声ファイルのコピー
├── manifest.jsonl # Train開始時に自動生成
└── latents/       # 同上（DACVAE事前計算）
```

---

## Train タブ

### 設定
- **Dataset**：Dataset タブで作ったもの
- **LoRA Name**：完成後の登録名
- **Base Model**：v3 / v2 / Voice Design
- **Preset**：
  - `Light` (`diffusion_attn`)：軽量、データ少なめ
  - `Standard` (`diffusion_attn_mlp`) ← 推奨
  - `Broad` (`all_attn_mlp`)：データ多い時
- **Max Steps**：典型 2000〜5000
- **Save Every**：500〜1000

### 詳細（VRAM チューニング）
- `Batch Size = 4` + `Gradient Accum = 8` で実効バッチ32（VRAM 12〜16GB目安）
- VRAM 不足なら Batch を 2 や 1 に下げる

### 実行
**Start Training** クリックで：
1. 推論用モデル（Whisper / TTS）を一旦アンロード（VRAM解放）
2. データセットを DACVAE で事前エンコード（マニフェスト構築）
3. `train.py` をサブプロセス起動
4. **2秒ごとにポーリング**：state / current step / loss / プログレスバー / ログ tail を表示
5. 完了 → LoRA registry に **自動登録** → Synthesize タブの LoRA dropdown に即出現

### Stop
学習中ジョブを taskkill。途中保存された checkpoint は残る（手動でインポートして使える）。

### 制限
- **同時1ジョブのみ**（VRAM競合回避）
- 学習中は推論できない（VRAM占有）

---

## データ保存先

| モード | パス |
|-|-|
| Electron 開発（npm start） | `APP/` 配下 |
| Electron パッケージ版（.exe） | `<install_dir>/data/` |
| `python server.py` 直接 | `APP/` 配下 |

レイアウト：
```
<data_root>/
├── voices/
│   ├── metadata.json
│   └── voice_*.wav         # 参照音声ライブラリ
├── outputs/
│   └── sample_*.wav        # 生成結果
├── datasets/
│   ├── _locations.json     # 外部保存先レジストリ
│   └── <name>/...
├── loras/
│   └── <name>/adapter/...  # 学習済み LoRA
└── lora_jobs/
    └── <id>/status.json    # 学習ジョブ状態
```

旧 `references/` + `metadata.json` がある場合は初回起動時に自動で `voices/` へ移行されます。

---

## トラブルシューティング

### 「Connecting...」のままで接続できない
- バックエンドの Python サーバーが起動失敗してる可能性
- バッチを実行したコマンドプロンプトを残してログ確認（`[Electron] API port:` に実際のポートが出る）
- ポート衝突ならアプリが自動で次の番号にずらすため、ここが原因になることはまずない。
  8080〜8129 が全て埋まっている場合のみエラーダイアログを出して終了する

### Process ボタンを押しても何も起きない
- Anime-whisper モデル初回 DL 中（~1.5GB、数十秒〜数分）
- ステータス行に "Transcribing..." が出てれば動いてる

### Save dataset で 409 エラー
- 同じ名前のデータセットが既に存在
- 別名にするか、先に Manage LoRAs で削除

### Train Start で 409 エラー
- 別ジョブが進行中（同時1ジョブ制限）
- Job History で stopping 中のものがあれば完了を待つ、または Stop

### Train で OOM
- Batch Size を下げる（4 → 2 → 1）
- Gradient Accum を増やす（8 → 16）で実効バッチ維持
- max_latent_steps は config 側で 400 程度に縮められる（要 yaml 編集）

### LoRA で生成すると変な音
- ベースと LoRA が不一致（v3 LoRA を v2 で使う等）
- データ少なすぎ／step数足りない（5min 1000step は厳しい）
- 過学習：もっと早い checkpoint を試す

---

## 関連ドキュメント
- [external-api.md](external-api.md) — 外部（別スクリプト・別端末）から呼び出す実用ガイド
- [lora-usage.md](lora-usage.md) — LoRA インポート詳細・トラブルシュート
- [parameters.md](parameters.md) — 全推論・学習パラメータの一覧（上流ドキュメント）
- [plans/2026-05-15-lora-import-design.md](plans/2026-05-15-lora-import-design.md) — LoRA インポート設計（実装者向け）
