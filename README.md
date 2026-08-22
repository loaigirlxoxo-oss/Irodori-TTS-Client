# Irodori-TTS Desktop

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（Aratako 氏）に、音声合成・データセット作成・LoRA 学習・朗読を
ひとつのウィンドウで扱える Electron アプリを載せたものです。

コードの大部分は上流のものです。上流の README は [README.upstream.md](README.upstream.md)（英語）と
[README.upstream.ja.md](README.upstream.ja.md)（日本語の要約）に置いてあります。

---

## 導入

**必要なもの**

| 項目 | 条件 |
|-|-|
| OS | Windows 10 / 11 |
| Python | **3.10**（[3.10.11](https://www.python.org/downloads/release/python-31011/)）／AMD 使用時のみ 3.12 |
| Node.js | LTS 版（[nodejs.org](https://nodejs.org/en/download)） |
| Git | [Git for Windows](https://git-scm.com/install/windows) |
| GPU | NVIDIA 製・VRAM 8GB 以上を推奨（AMD は試験対応、GPU 無しでも CPU で動作） |
| 空き容量 | **30GB 以上**（環境 5.7GB ＋ モデル 17GB ＋ 作業領域） |

Python は通常 **3.10** を使います。3.11 以降では `sentencepiece` の導入に失敗するためです。

ただし **AMD (Radeon) を使う場合だけ 3.12** が必要です。AMD が配る Windows 用 PyTorch が
3.12 でしか配布されていないためで、`setup.bat` が GPU を見て自動で使い分けます。
Radeon は試験対応です。生成は動く見込みですが、AMD は Windows での学習を公式に
対応していません（学習には NVIDIA か WSL2 が要ります）。
インストーラの「Add python.exe to PATH」に必ずチェックを入れてください。

**手順**

```
1. setup.bat をダブルクリック   … 30〜60分（環境構築＋モデル約17GB取得）
2. 起動.bat をダブルクリック
```

詳細は [はじめにお読みください.md](はじめにお読みください.md) を参照してください。

**更新するとき**

以前の版は v4 を別の Python 環境で動かしていました。いまはひとつに統合されて
いるので、`Irodori-TTS-v4\` フォルダ（約 5GB）は使われません。残っていれば
削除して構いません。`setup.bat` は触らないので、消さない限り残り続けます。

---

## タブ

| タブ | 役割 |
|-|-|
| Synthesize | テキスト→音声合成（v4.1 / v4 / v3 / v2 と VoiceDesign 2種の計6モデル） |
| Dataset | 音声の自動分割→書き起こし→データセット化 |
| Train | データセットから LoRA を学習 |
| 朗読 | テキストファイルを連続合成、しおり機能 |
| 青空文庫 | 著作権切れ作品を検索・取得して朗読へ渡す |
| LoRAマージ | 複数の LoRA を比率指定で合成 |
| 辞書 | 読み間違いの矯正 |

---

## 検証状況

確認できている範囲を明記します。

**確認済み**

- `setup.bat` の完走（Node / Python 環境 / 音声モデル6種＋コーデック・トークナイザ・書き起こし・電子透かし、約17GB）
- CUDA 有効（torch 2.10.0+cu128 / torchcodec 0.10.0）
- アプリ起動と7タブの表示
- 6モデルすべての生成（v4.1 / v4 / v3 / v2 / v3 VoiceDesign / v2 VoiceDesign）
- 参照音声あり・なしの両方
- 計算装置（GPU / CPU）と計算精度（fp32 / bf16）の切り替え
- LoRA 学習と登録（v4.1 / v4 / v3 / v2 と VoiceDesign 2種の計6ベース）
- Dataset タブの音声分割・書き起こし（silero-vad + anime-whisper）
- データセットの latents 生成
- 朗読 / 青空文庫 / LoRAマージ / 辞書
- ポート衝突時の自動回避（8080 使用中なら 8081 へ）
- LoRA・音声・データセットが空の状態でも落ちないこと
- モデルのオフライン解決（`HF_HUB_OFFLINE=1`）

**未確認**

- AMD (Radeon) での動作。導入経路は用意しましたが実機で確かめていません
- Linux / macOS。`setup.bat` と `起動.bat` は Windows 専用です

確認した環境は Windows / RTX 5080 (16GB) と RTX 3080 の2構成です。それ以外では確認していません。

---

## 構成

```
APP/                  Electron アプリ本体（main.js / renderer.js / server*.py）
irodori_tts/          推論・学習ランタイム（上流 main。v2/v3/v4/VoiceDesign 共通）
configs/ docs/        設定とドキュメント
setup.bat 起動.bat    セットアップと起動
```

以前は v4 だけ別ツリー・別 Python 環境に置き、サブプロセスで呼んでいました。
上流の `main` が v2/v3/VoiceDesign のチェックポイントと後方互換になったので、
ひとつのツリーに統合し、v4 も同じランタイムで直接読むようにしています。

データは `APP/` 配下（`voices` `loras` `datasets` `outputs` など）に置かれ、フォルダごと移動できます。

---

## ライセンス

コードは MIT（[LICENSE](LICENSE)）。

| 範囲 | 著作権 |
|-|-|
| エンジン（学習・推論コード） | Copyright (c) 2026 Aratako — [上流リポジトリ](https://github.com/Aratako/Irodori-TTS) |
| デスクトップアプリ層（`APP/`） | Copyright (c) 2026 Lo-Ai girl |

モデルの利用条件は配布元の規約に従ってください。
生成した音声の扱いは利用者の責任です。実在人物の声を本人の許可なく再現する用途には使わないでください。
