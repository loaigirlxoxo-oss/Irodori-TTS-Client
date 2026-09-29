# Irodori-TTS Client App

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（Aratako 氏）に、音声合成・データセット作成・LoRA 学習・朗読を
ひとつのウィンドウで扱える Electron アプリを載せたものです。

コードの大部分は上流のものです。上流の README は [README.upstream.md](README.upstream.md)（英語）と
[README.upstream.ja.md](README.upstream.ja.md)（日本語の要約）に置いてあります。

---

## 導入

### インストーラで入れる（おすすめ）

簡単なので、こちらをおすすめします。

[Releases](https://github.com/loaigirlxoxo-oss/Irodori-TTS-Client/releases) から
`Irodori-TTS-Client-App-Setup-<版>.exe` をダウンロードして実行します。
Python は同梱しているので、Python・Node.js・Git を別に入れる必要はありません。

1. インストーラを実行し、入れる場所を選んで「インストール」
2. デスクトップのアイコンから起動
3. 初回だけ準備の画面が出て、動かすための部品と音声モデル（約30GB）を取得します

インストーラには電子署名がないため、実行すると「Windows によって PC が保護されました」と表示されることがあります。その場合は「詳細情報」を押し、「実行」を押してください。

| 項目 | 条件 |
|-|-|
| OS | Windows 10 / 11 |
| GPU | NVIDIA 製を推奨（AMD は試験対応、GPU 無しでも CPU で動作） |
| VRAM | 6〜12GB（生成・朗読は 6GB 以上、学習は 12GB 以上を推奨） |
| 空き容量 | **45GB 以上を推奨**（環境 5.7GB ＋ モデル 30GB ＋ 作業領域） |

### ソースから入れる

緑の **Code** ボタンから **Download ZIP** を選び、好きな場所に展開します。

Git を使う場合は次のようにします。

```
git clone https://github.com/loaigirlxoxo-oss/Irodori-TTS-Client.git
```

展開して置くだけです。この時点ではまだ動きません。下記の必要なものを揃えてから、
`setup.bat` を実行してください。環境とモデルはそこで取得されます。

**必要なもの**

| 項目 | 条件 |
|-|-|
| OS | Windows 10 / 11 |
| Python | **3.10**（[3.10.11](https://www.python.org/downloads/release/python-31011/)）／AMD 使用時のみ 3.12 |
| Node.js | LTS 版（[nodejs.org](https://nodejs.org/en/download)） |
| Git | [Git for Windows](https://git-scm.com/install/windows) |
| GPU | NVIDIA 製を推奨（AMD は試験対応、GPU 無しでも CPU で動作） |
| VRAM | 6〜12GB（生成・朗読は 6GB 以上、学習は 12GB 以上を推奨） |
| 空き容量 | **45GB 以上を推奨**（環境 5.7GB ＋ モデル 30GB ＋ 作業領域） |

かんたん学習タブだけは **VRAM 12GB 以上**を見てください。長い文（200文字程度）を
鳴らす時点で合成に 8.5GB ほど要り、学習と採用の評価まで通すとピークが約 11GB でした
（RTX 5080 での実測）。8GB だと途中で足りなくなる可能性があります。生成と朗読は
GTX 1060（6GB）で確認しています。ただし v4-Large の生成だけは VRAM を約7.6GB 使います
（RTX 5080 での実測）。6GB の GPU では v4.1-Small など他のモデルを使ってください。

Python は通常 **3.10** を使います。3.11 以降では `sentencepiece` の導入に失敗するためです。

ただし **AMD (Radeon) を使う場合だけ 3.12** が必要です。AMD が配る Windows 用 PyTorch が
3.12 でしか配布されていないためで、`setup.bat` が GPU を見て自動で使い分けます。
Radeon は試験対応です。生成は動く見込みですが、AMD は Windows での学習を公式に
対応していません（学習には NVIDIA か WSL2 が要ります）。
インストーラの「Add python.exe to PATH」に必ずチェックを入れてください。

**手順**

```
1. setup.bat をダブルクリック   … 30〜60分（環境構築＋モデル約30GB取得）
2. 起動.bat をダブルクリック
```

詳細は [はじめにお読みください.md](はじめにお読みください.md) を参照してください。

---

## タブ

| タブ | 役割 |
|-|-|
| Synthesize | テキスト→音声合成（v4-Large / v4.1 / v4 / v3 / v2 と VoiceDesign 2種の計7モデル、任意で Anime を追加可） |
| かんたん学習 | 声を決めるだけで、素材の生成→選別→学習→登録までを通す（VRAM 12GB 以上を推奨） |
| Dataset | 音声の自動分割→書き起こし→データセット化 |
| Train | データセットから LoRA を学習 |
| 朗読 | テキストファイルを連続合成、しおり機能 |
| 青空文庫 | 著作権切れ作品を検索・取得して朗読へ渡す |
| LoRAマージ | 複数の LoRA を比率指定で合成 |
| 辞書 | 読み間違いの矯正 |

---

## 任意で追加できるモデル

`setup.bat` が取得するのは上流（Aratako 氏）のモデルだけです。それ以外を使いたい
場合は、手動で置けば選択肢に出ます。

### v4.1-Anime

v4.1-Small をアニメ調の音声で追加学習した、第三者製のモデルです。

| | |
|-|-|
| 配布元 | [phasefield-audio/Irodori-TTS-v4.1-Anime](https://huggingface.co/phasefield-audio/Irodori-TTS-v4.1-Anime) |
| ライセンス | MIT（倫理的な制限もベースモデルと同じものを引き継ぎます） |
| サイズ | 約 2.9GB |

**置き方**

1. 上のページから `model.safetensors` をダウンロードします
2. `models\manual\Irodori-TTS-v4.1-Anime\` を作ります
3. そこに `model.safetensors` を置きます

```
models\manual\Irodori-TTS-v4.1-Anime\model.safetensors
```

アプリを起動すると、モデル選択に「v4.1-Anime」が出ます。置いていなければ選んだ
時点でエラーになるだけで、他の動作には影響しません。

**LoRA について**

v4.1-Small と構造が同じ（safetensors のキー 714 件と全ての形状が一致）なので、
**v4.1 用に作った LoRA はそのまま当てられます。** 作り直す必要はありません。
声の傾向は、当てた側のベースに寄ります。

このモデルは上流ではないため `setup.bat` の取得対象に入れていません。配布元が
なくなってもセットアップは通ります。

---

## 検証状況

確認できている範囲を明記します。

**確認済み**

- `setup.bat` の完走（Node / Python 環境 / 音声モデル7種＋コーデック・トークナイザ・書き起こし・電子透かし・話者照合、約30GB）
- CUDA 有効（torch 2.10.0+cu128 / torchcodec 0.10.0）
- アプリ起動と8タブの表示
- 7モデルすべての生成（v4-Large / v4.1 / v4 / v3 / v2 / v3 VoiceDesign / v2 VoiceDesign）
- 参照音声あり・なしの両方
- 計算装置（GPU / CPU）と計算精度（fp32 / bf16）の切り替え
- LoRA 学習と登録（v4.1 / v4 / v3 / v2 と VoiceDesign 2種の計6ベース）
- v4-Large の LoRA 学習（**ベータ**）。RTX 5080 (16GB) で10ステップの学習・登録・生成まで。
  学習中の VRAM は約15.9GB で、16GB の GPU ではほぼ上限です
- かんたん学習タブの通し（生成→選別→学習→登録）
- Dataset タブの音声分割・書き起こし（silero-vad + anime-whisper）
- データセットの latents 生成
- 朗読 / 青空文庫 / LoRAマージ / 辞書
- ポート衝突時の自動回避（8080 使用中なら 8081 へ）
- LoRA・音声・データセットが空の状態でも落ちないこと
- モデルのオフライン解決（`HF_HUB_OFFLINE=1`）

**未確認**

- v4-Large の LoRA 学習を最後まで回したときの品質。上流が v4-Large の LoRA 用設定を
  出していないため、設定は v4-Small のものを元にこちらで作っています
- AMD (Radeon) での動作。導入経路は用意しましたが実機で確かめていません
- Linux / macOS。`setup.bat` と `起動.bat` は Windows 専用です

確認した環境は Windows / RTX 5080 (16GB)・RTX 3080・GTX 1060（生成と朗読）の3構成です。それ以外では確認していません。

---

## ライセンス

コードは MIT（[LICENSE](LICENSE)）。

| 範囲 | 著作権 |
|-|-|
| エンジン（学習・推論コード） | Copyright (c) 2026 Aratako — [上流リポジトリ](https://github.com/Aratako/Irodori-TTS) |
| デスクトップアプリ層（`APP/`） | Copyright (c) 2026 Lo-Ai girl |

### 同梱しているもの

インストーラには、このアプリのコード以外に次のものが入っています。許諾文も一緒に入れてあります。

| 中身 | 許諾 | 許諾文の場所 |
|-|-|-|
| Python 3.12.10（embeddable） | PSF License Agreement | `resources\python-embed\LICENSE.txt` |
| pip | MIT | 初回起動で `data\runtime` に展開されます |
| Electron / Chromium ほか | MIT ／ BSD-3-Clause ほか | `LICENSE.electron.txt` `LICENSES.chromium.html` |
| 夜永オールド明朝 Bold | SIL Open Font License 1.1 | `OFL.txt` |

**Git と uv は同梱していません。** アプリの導入にも実行にも使いません。
上流の手順では両方が必要ですが、こちらは Python を同梱し、依存の取得は同梱の pip が行います。

### 取得するもの

音声モデル約30GB、書き起こしモデル、電子透かし、話者照合モデルは、初回の準備で
**利用者のパソコンが配布元から直接取得します**。このアプリが再配布しているものでは
ありません。取得先と許諾の一覧は [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)
にまとめてあります（本文にも同じものが入ります）。

v4-Large だけは許諾が MIT ではありません。文章の読み取り部分が google/t5gemma-2-1b-1b 由来のため、
[Gemma 利用規約](https://ai.google.dev/gemma/terms) と
[Gemma 使用禁止ポリシー](https://ai.google.dev/gemma/prohibited_use_policy) に従う必要があります。

### 生成した音声について

音声モデルの配布元は、許諾とは別に次のことを求めています。

- 本人の明確な同意なく、特定の個人（声優・著名人・公人など）の声を複製したり
  なりすましたりしないこと
- 人を欺いたり誤情報を広めたりする目的の合成音声・ディープフェイクを作らないこと
- 生成物の扱いは利用者の責任であり、適用される法令の遵守は利用者が負うこと

文章だけから作った声が実在の人物にたまたま似ることがありますが、それは確率的な
副産物であって意図した再現ではありません。
