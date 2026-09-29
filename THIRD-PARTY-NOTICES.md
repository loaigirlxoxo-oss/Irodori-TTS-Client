# 第三者のソフトウェアとモデルについて

Irodori-TTS Client は、他の方が作ったソフトウェアとモデルの上に成り立っています。
ここに、同梱しているもの・利用者の環境が取得するものを、それぞれの許諾と一緒に並べます。

このファイルはインストール先の直下にも置かれます。

---

## 1. インストーラに同梱しているもの

インストーラの中に実体が入っていて、そのまま配られるものです。

| 中身 | 許諾 | 権利者 | 同梱している許諾文 |
|-|-|-|-|
| Python 3.12.10（embeddable） | PSF License Agreement | Python Software Foundation | `resources\python-embed\LICENSE.txt` |
| pip 26.2.1 | MIT | The pip developers | wheel 内の `LICENSE.txt`（初回起動で展開され、`data\runtime\Lib\site-packages\pip-*.dist-info\` に残る） |
| Electron | MIT | GitHub Inc. および貢献者 | `LICENSE.electron.txt` |
| Chromium ほか Electron の依存 | BSD-3-Clause ほか | 各権利者 | `LICENSES.chromium.html` |
| Irodori-TTS 本体（学習・推論コード） | MIT | Copyright (c) 2026 Aratako | `resources\backend\LICENSE` |
| 夜永オールド明朝 Bold（`YonagaOldMincho-Bold.woff2`） | SIL Open Font License 1.1 | Copyright (c) 2023, ichi, with Reserved Font Name 'Yonaga' | `assets\OFL.txt`（アプリ内）／このファイルの末尾に全文 |

**Git と uv は同梱していません。** 本アプリの導入・実行に Git も uv も使いません。
上流リポジトリの手順では両方が必要ですが、本アプリは Python を同梱し、
依存の取得は同梱の pip が行います。

---

## 2. 初回起動時に利用者の環境が取得するもの

これらはインストーラに入っていません。初回の準備で、利用者のパソコンが
配布元から直接取得します。**本アプリが再配布しているものではありません。**
利用条件は各配布元の規約に従ってください。

### 音声モデル（Hugging Face）

| リポジトリ | 用途 | 許諾 |
|-|-|-|
| `Aratako/Irodori-TTS-500M-v2` | 音声モデル v2 | MIT |
| `Aratako/Irodori-TTS-500M-v3` | 音声モデル v3 | MIT |
| `Aratako/Irodori-TTS-v4-Small` | 音声モデル v4 | MIT |
| `Aratako/Irodori-TTS-v4.1-Small` | 音声モデル v4.1（既定） | MIT |
| `Aratako/Irodori-TTS-v4-Large` | 音声モデル v4-Large | [Gemma Terms of Use](https://ai.google.dev/gemma/terms)（[Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) を含む） |
| `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | VoiceDesign v2 | MIT |
| `Aratako/Irodori-TTS-600M-v3-VoiceDesign` | VoiceDesign v3 | MIT |
| `Aratako/Semantic-DACVAE-Japanese-32dim` | 音声コーデック | MIT |

v4-Large は文章の読み取り部分が `google/t5gemma-2-1b-1b` 由来のため、Gemma の規約に従います
（配布元のモデルカードより）。トークナイザもこのモデルに同梱のものを使い、`google/t5gemma-2-1b-1b` からは取得しません。

音声モデルには、許諾とは別に**利用上の約束**が書かれています（配布元のモデルカードより）。

- 本人の明確な同意なく、特定の個人（声優・著名人・公人など）の声を複製したり
  なりすましたりしないこと
- 人を欺いたり誤情報を広めたりする目的の合成音声・ディープフェイクを作らないこと
- 文章だけから作った声が実在の人物にたまたま似ることがあるが、それは確率的な
  副産物であって意図した再現ではないこと
- 生成物の扱いは利用者の責任であり、適用される法令の遵守は利用者が負うこと

### 付随するモデル

| リポジトリ | 用途 | 許諾 |
|-|-|-|
| `sbintuitions/modernbert-ja-310m` | 日本語トークナイザ（v4 / v4.1） | MIT |
| `llm-jp/llm-jp-3-150m` | 日本語トークナイザ（v2 / v3 / VoiceDesign） | Apache-2.0 |
| `litagin/anime-whisper` | 書き起こし（データセットタブ） | MIT |
| `sony/silentcipher` | 生成音声への電子透かし | MIT |
| `speechbrain/spkrec-ecapa-voxceleb` | 話者の似ている度（かんたん学習の選別） | Apache-2.0 |

トークナイザの2つは、トークナイザの定義ファイルだけを取得します（重みは取得しません）。

### Python パッケージ

`requirements.txt` に書かれた依存（PyTorch ほか）を、同梱の pip が PyPI から取得します。
AMD (Radeon) を選んだ場合は、AMD が配布する Windows 向け PyTorch を
`repo.radeon.com` から取得します。いずれも本アプリは再配布していません。

### 任意で追加できるモデル

| リポジトリ | 用途 | 許諾 |
|-|-|-|
| `phasefield-audio/Irodori-TTS-v4.1-Anime` | 任意追加のアニメ調モデル | MIT（倫理的な制限はベースモデルを引き継ぎます） |

これは初回準備の取得対象に入れていません。使いたい人が手で置くものです。

---

## 3. 画像とロゴ

アプリの銘（`assets/logo.png`）は **叛逆明朝**（作者: daredemotypo、Freeware）を
画像化したものです。フォントファイル自体は同梱していません。
配布元の記載では、宗教的・政治的・公序良俗に反する用途を除き商用利用が認められています。

---

## 4. 夜永オールド明朝 の許諾全文（SIL Open Font License 1.1）

Copyright (c) 2023, ichi, with Reserved Font Name 'Yonaga'.

This Font Software is licensed under the SIL Open Font License, Version 1.1.
全文は `APP/assets/OFL.txt`、および https://openfontlicense.org を参照してください。

---

## 5. 本アプリ自体

| 範囲 | 許諾 | 権利者 |
|-|-|-|
| エンジン（学習・推論コード） | MIT | Copyright (c) 2026 Aratako |
| デスクトップアプリ層（`APP/`） | MIT | Copyright (c) 2026 Lo-Ai girl |

全文は [LICENSE](LICENSE) を参照してください。
