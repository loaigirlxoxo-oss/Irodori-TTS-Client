# Irodori-TTS Client App

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（Aratako 氏）に、音声合成・データセット作成・LoRA 学習・朗読を
ひとつのウィンドウで扱える Electron アプリを載せたものです。

使い方の動画（YouTube）：https://youtu.be/eO6362tjJRE

更新の内容は [更新履歴.md](更新履歴.md) にあります。

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
3. 初回だけ準備の画面が出ます。取得する音声モデルを選ぶと、動かすための部品と音声モデルを取得します
   （最初から選ばれている3つで約15GB。分からなければそのまま進んで大丈夫です）

インストーラには電子署名がないため、実行すると「Windows によって PC が保護されました」と表示されることがあります。その場合は「詳細情報」を押し、「実行」を押してください。

| 項目 | 条件 |
|-|-|
| OS | Windows 10 / 11 |
| GPU | NVIDIA 製を推奨（AMD は試験対応、GPU 無しでも CPU で動作） |
| VRAM | 6〜16GB（生成・朗読は 6GB 以上、学習は 12GB 以上を推奨。v4-Large の学習は 16GB） |
| 空き容量 | **30GB 以上を推奨**（環境 5.7GB ＋ 最初から選ばれているモデル 約15GB ＋ 作業領域）。全モデルを取得する場合は 60GB 以上 |

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
GTX 1060（6GB）で確認しています。v4-Large の生成は VRAM を約8.5GB 使います（RTX 5080 での実測）。
VRAM が少ない GPU では、v4.1-Small か、v4-Large の軽量版（RTX 30 系以降）を使ってください。

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
| Synthesize | テキスト→音声合成（v4-Large とその軽量版3種 / v4.1 / v4.1-Anime / v4 / v3 / v2 と VoiceDesign 2種）。生成した WAV から設定を読み戻す「メタデータ読込」つき |
| かんたん学習 | 声を決めるだけで、素材の生成→選別→学習→登録までを通す（VRAM 12GB 以上を推奨） |
| Dataset | 音声の自動分割→前後の空白カット→書き起こし（Anime Whisper / Qwen3-ASR / 両方で突き合わせ）→一括置き換え→データセット化。音源と GPU を測っておすすめの設定を入れる機能つき |
| Train | データセットから LoRA を学習。設定は全部画面に出ていて、学習後に各ステップを測っておすすめに印を付ける |
| 朗読 | テキストファイルを連続合成、しおり機能 |
| 青空文庫 | 著作権切れ作品を検索・取得して朗読へ渡す |
| LoRAマージ | 複数の LoRA を比率指定で合成 |
| 辞書 | 読み間違いの矯正 |

右上の歯車（設定）には、音声モデルの取得と削除、アプリのアップデート、電子透かしの入れ方をまとめています。

---

## 音声モデル

取得するモデルは、初回の準備の画面と、右上の歯車（設定）で選べます。使うものだけを取得し、
あとから足したり消したりできます。取得していないモデルは、選択肢に「（未取得）」と出ます。

| モデル | 容量 | 生成時の VRAM（RTX 30 系以降） | 生成時の VRAM（それより前の世代） |
|-|-|-|-|
| v4.1-Small（推奨・最初から選択） | 3.1GB | 4.0GB | 7.5GB |
| v4-Large 軽量 int8（最初から選択） | 3.7GB | 6.0GB | 動きません |
| v3 VoiceDesign（最初から選択） | 2.5GB | 3.5GB | 7.0GB |
| v4.1-Anime（第三者製） | 3.1GB | 4.0GB | 7.5GB |
| v4-Large | 13.2GB | 8.5GB | 動きません |
| v4-Large 軽量 int4 | 2.8GB | 5.5GB | 動きません |
| v4-Large 軽量 float8（RTX 40 系以降） | 3.7GB | 6.0GB | 動きません |
| v4-Small | 3.1GB | 4.0GB | 7.5GB |
| v3 / v2 / v2 VoiceDesign | 2.0〜2.1GB | 3.5GB | 6.5GB |

VRAM の値は RTX 5080 での実測です。一番重い使い方（30秒の文、参照音声30秒）で測り、
0.5GB 刻みで切り上げています。RTX 30 系より前の世代は計算の精度が変わるため、使う VRAM が増えます。
どのモデルを選んでも要る部品（コーデック・書き起こし・電子透かしなど）は、別に約6GB 取得します。

### v4-Large の軽量版

[Aratako/Irodori-TTS-v4-Large-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized)
の int8・int4・float8 です。v4-Large の重みを小さくしたもので、少ない VRAM で動きます。

- **v4-Large で学習した LoRA が、そのまま当てられます。** 学習は通常版の v4-Large で行います
  （軽量版は学習に使えません）
- NVIDIA の RTX 30 系以降が必要です。float8 は RTX 40 系以降でしか動きません
- 許諾は v4-Large と同じ Gemma 利用規約です

### v4.1-Anime

v4.1-Small をアニメ調の音声で追加学習した、第三者製のモデルです
（[phasefield-audio/Irodori-TTS-v4.1-Anime](https://huggingface.co/phasefield-audio/Irodori-TTS-v4.1-Anime)、MIT）。
v4.1-Small と構造が同じなので、**v4.1 用に作った LoRA はそのまま当てられます。**
声の傾向は、当てた側のベースに寄ります。

以前の版のように `models\manual\Irodori-TTS-v4.1-Anime\model.safetensors` へ手で置いたものも、
そのまま使えます。

### VRAM が足りないとき

VRAM が足りなくなると、Windows は足りない分を本体のメモリへ逃がし、PC 全体が重くなります。
このアプリは、生成・書き起こし・学習の前に、その時点で空いている VRAM に合わせて上限をかけ、
あふれる前に止めます。「VRAM が足りません」と出たら、ほかのアプリを閉じるか、軽いモデル・短い文で
試してください。上限は環境変数 `IRODORI_VRAM_FRACTION`（既定 0.95、0 で無効）で変えられます。

---

## 検証状況

確認できている範囲を明記します。

**確認済み**

- `setup.bat` の完走（Node / Python 環境 / 音声モデル7種＋コーデック・トークナイザ・書き起こし・電子透かし・話者照合、約30GB）
- CUDA 有効（torch 2.10.0+cu128 / torchcodec 0.10.0）
- アプリ起動と8タブの表示
- 7モデルすべての生成（v4-Large / v4.1 / v4 / v3 / v2 / v3 VoiceDesign / v2 VoiceDesign）
- v4-Large の軽量版3種（int8 / int4 / float8）の生成と、v4-Large で学習した LoRA を当てた生成（RTX 5080）
- モデルの選択・取得・削除（初回の準備の画面と設定画面。インストール版で確認）
- VRAM の上限（上限を約4GB に下げ、収まらない生成があふれずに止まること）
- 参照音声あり・なしの両方
- 計算装置（GPU / CPU）と計算精度（fp32 / bf16）の切り替え
- LoRA 学習と登録（v4.1 / v4 / v3 / v2 と VoiceDesign 2種の計6ベース）
- v4-Large の LoRA 学習（**ベータ**）。RTX 5080 (16GB) で1500ステップの学習・各ステップの評価・登録・生成まで。
  「学習を速くする」がオンのとき、1ステップ（32本）約1.8秒です
- かんたん学習タブの通し（生成→選別→学習→登録）
- Dataset タブの音声分割（silero-vad / 音量）・前後の空白カット・書き起こし（anime-whisper）・一括置き換え・おすすめ設定
- Dataset タブの Qwen3-ASR（画面からの取得と書き起こし。開発環境で確認）
- インストーラでの新規インストールと、前の版からのアップデート（データが残ること）
- データセットの latents 生成
- 朗読 / 青空文庫 / LoRAマージ / 辞書
- ポート衝突時の自動回避（8080 使用中なら 8081 へ）
- LoRA・音声・データセットが空の状態でも落ちないこと
- モデルのオフライン解決（`HF_HUB_OFFLINE=1`）

**未確認**

- v4-Large の LoRA 学習を、VRAM 16GB 未満の GPU で動かすこと。上流が v4-Large の LoRA 用設定を
  出していないため、設定は v4-Small のものを元にこちらで作っています
- v4-Large の軽量版を、VRAM 6〜8GB の実機で動かすこと。使える VRAM を約5.2GB に絞った状態で、
  int4 が30秒の文を生成できるところまでは確認しました
- 設定画面の「ダウンロードして更新」。1.2.2 より新しい版が出るまで試せません
- AMD (Radeon) での動作。導入経路は用意しましたが実機で確かめていません
- Linux / macOS。`setup.bat` と `起動.bat` は Windows 専用です

確認した環境は Windows / RTX 5080 (16GB)・RTX 3080・GTX 1060（生成と朗読）の3構成です。それ以外では確認していません。

---

## 検討中の機能

今後の対応を検討・予定している機能です。

- YukkuriMovieMaker（YMM4）のプラグインとして、YMM4 の中から呼び出せるようにする
- VOICEVOX 互換の API（VOICEVOX に対応したアプリから呼び出せるようにする）
- CSV・Markdown の台本読み込み（行ごとに話者を指定して一括生成）
- 出力後に、セリフとファイル名の対応表を CSV に書き出す
- v4-Large の正式対応（いまはベータ）
- v4.1-Small-MF への対応（少ないステップで速く生成）
- Claude・Codex の出力の読み上げ

進み具合や新しい機能は X（[@Lo_Ai_girl](https://x.com/Lo_Ai_girl)）でお知らせします。
要望は Issues へどうぞ。

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

音声モデル、書き起こしモデル、電子透かし、話者照合モデルは、初回の準備や設定画面で
**利用者のパソコンが配布元から直接取得します**。このアプリが再配布しているものでは
ありません。取得先と許諾の一覧は [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)
にまとめてあります（本文にも同じものが入ります）。

v4-Large とその軽量版だけは許諾が MIT ではありません。文章の読み取り部分が google/t5gemma-2-1b-1b 由来のため、
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

生成した音声には、[SilentCipher](https://github.com/sony/silentcipher) の電子透かしが必ず入ります。
AI で生成した音声であることを後から確かめられるようにするためで、外す設定はありません。
WAV には、生成に使った設定（モデル、seed など）も埋め込みます。
