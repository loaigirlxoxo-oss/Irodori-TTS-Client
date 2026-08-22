# Irodori-TTS（上流エンジンの日本語要約）

このファイルは [README.upstream.md](README.upstream.md)（上流の英語版）の日本語要約です。
最新の正確な内容は英語版を参照してください。

Flow Matching に基づく日本語 TTS のモデルと、その学習・推論コードです。
アーキテクチャと学習設計はおおむね [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/)
に倣い、生成対象として [DACVAE](https://github.com/facebookresearch/dacvae) の
連続潜在表現を使います。

> [!IMPORTANT]
> `main` は **v4** 系のコードを追っており、`Irodori-TTS-v4.1-Small` と組み合わせて
> 使うことを想定しています。現行のコードは v2/v3 の base および VoiceDesign の
> チェックポイントとも後方互換です。過去の状態は `v3` / `v2` / `v1` タグにあります。
> v1 のチェックポイントと前処理は v2 以降と互換がありません。

---

## できること

- **Flow Matching による音声合成** — DACVAE の連続潜在の上で Rectified Flow
  Diffusion Transformer（RF-DiT）を動かします
- **ゼロショットの声の複製** — 参照音声からその場で声を写します
- **多モーダルな声の設計** — v4-Small はテキスト・参照音声・キャプション文の
  3 つを組み合わせ、声の同一性に加えて話し方や感情も指定できます
- **長い参照音声** — 複数のクリップを繋いで、チェックポイントの上限
  120 秒まで使えます
- **絵文字による表現の指定** — 入力テキストに絵文字を挟むと、対応する
  チェックポイントでは話し方や非言語の発声に影響します
- **尺の自動推定** — v4-Small は `--seconds` を指定しなくても出力長を見積もります
- **電子透かし** — 生成音声には
  [SilentCipher](https://github.com/sony/silentcipher) が使える環境で自動的に
  透かしが入ります
- **複数 GPU での学習** — `torchrun` による分散学習、勾配累積、bf16 の混合精度、
  W&B へのログ
- **PEFT / LoRA での微調整** — 公開チェックポイントに対する軽量な適応
- **Speaker Inversion** — ベースモデルを凍結したまま、目的の声に対する
  再利用可能な話者埋め込みトークンを学習します
- **柔軟な推論** — CLI、Gradio の Web UI、HuggingFace Hub のチェックポイントに対応

---

## 構造

現行の **`Aratako/Irodori-TTS-v4.1-Small`** は、以前は分かれていた base 系と
VoiceDesign 系をひとつのチェックポイントに統合したものです。テキスト・参照音声・
キャプション文の 3 経路からの条件づけに対応します。v2/v3 のチェックポイントも
推論では引き続き使えます。

共通の構成要素:

1. **テキスト／キャプション共用エンコーダ** — 微調整した ModernBERT が、
   読み上げるテキストとキャプション文の両方を処理します
2. **参照潜在エンコーダ** — 話者の同一性を条件づけるために、パッチ化した参照音声の
   潜在を符号化します。v4-Small では合計 120 秒まで
3. **条件プロジェクタ** — 共用エンコーダの出力を、テキスト用とキャプション用の
   それぞれの条件空間へ写します
4. **Diffusion Transformer** — Joint-attention の DiT ブロック。Low-Rank AdaLN
   （タイムステップで条件づけた適応的レイヤ正規化）、half-RoPE、SwiGLU の MLP
5. **Duration Predictor** — 出力長を自動で見積もる予測器を内蔵

音声はチェックポイントが指定するコーデックを通して連続潜在の列として扱います。
公開されている v2/v3/v4 のチェックポイントは 32 次元の
[Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
を使い、48kHz の波形に戻します。

---

## 導入

`uv` を使う場合、環境に合わせて extra を選びます。

```bash
# NVIDIA CUDA 12.8（Linux / Windows）
uv sync --extra cu128

# AMD ROCm（Linux / WSL）
uv sync --extra rocm

# Intel XPU（Linux / Windows）
uv sync --extra xpu

# CPU のみ、または macOS の CPU/MPS
uv sync --extra cpu
```

ROCm の extra は Linux 向けです。Windows の Radeon については、
AMD が別途 Windows 用の wheel を配布しています（[UPSTREAM.md](UPSTREAM.md) と `setup.bat` を参照）。

---

## 推論

### コマンドラインから

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

ローカルのチェックポイント（`.pt` / `.safetensors`）も使えます。

```bash
uv run --no-sync python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "こんにちは、私はAIです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

v4-Small はキャプションによる条件づけに対応します。`--no-ref` を渡せば
キャプションだけで、参照音声と併用することもできます。

パラメータの詳細は [docs/parameters.md](docs/parameters.md) にあります。

### Gradio の Web UI

```bash
uv run --no-sync python gradio_app.py
```

---

## 学習

大まかな流れは次のとおりです。

1. `prepare_manifest.py` で音声データを DACVAE の潜在へ前処理し、manifest を作る
2. `configs/` の設定ファイルを選ぶ（モデルの規模と用途で分かれています）
3. `train.py` を実行する

LoRA で微調整する場合は `--lora` を付け、`configs/train_v4_small_lora.yaml`
などの LoRA 用設定を使います。

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small_lora.yaml \
  --manifest data/manifest.jsonl \
  --output-dir outputs/my_lora \
  --init-checkpoint path/to/model.safetensors \
  --lora
```

複数 GPU で回す場合は `torchrun` を使います。詳細と全パラメータは英語版の
README と [docs/parameters.md](docs/parameters.md) を参照してください。

---

## 構成

```text
Irodori-TTS/
├── train.py                    学習の入口（DDP 対応）
├── infer.py                    CLI からの推論
├── gradio_app.py               Gradio の Web UI
├── gradio_app_voicedesign.py   VoiceDesign 用の Web UI
├── prepare_manifest.py         データセット → DACVAE 潜在への前処理
├── quantize_checkpoint.py      torchao によるチェックポイントの量子化
│
├── docs/parameters.md          パラメータの詳細
│
├── irodori_tts/                本体
│   ├── model.py                TextToLatentRFDiT の実装
│   ├── rf.py                   Rectified Flow と Euler CFG のサンプリング
│   ├── codec.py                DACVAE コーデックのラッパ
│   ├── dataset.py              データセットと collator
│   ├── tokenizer.py            事前学習トークナイザのラッパ
│   ├── config.py               モデルと学習の設定
│   ├── inference_runtime.py    推論ランタイム（キャッシュ・スレッド安全）
│   ├── lora.py                 PEFT / LoRA の統合
│   ├── quantization.py         量子化チェックポイントの読み書き
│   └── speaker_inversion.py    話者埋め込みの学習
│
└── configs/                    学習の設定ファイル
```

---

## ライセンスと出典

コードは MIT です。詳細は [LICENSE](LICENSE) を参照してください。

上流のエンジン部分（学習・推論コード）は Aratako 氏によるものです。
このリポジトリは
[github.com/Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
を土台にしており、独自の改変は [UPSTREAM.md](UPSTREAM.md) に記録しています。

モデルの利用条件は配布元の規約に従ってください。
