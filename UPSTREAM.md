# 上流との関係

このリポジトリは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
を土台に、Electron のデスクトップアプリ（`APP/`）を足したもの。

## 上流を追う

remote はローカル設定なので clone には付いてこない。まず一度だけ登録する。

```
git remote add upstream https://github.com/Aratako/Irodori-TTS.git
git fetch upstream --tags
```

以後は差分をいつでも見られる。

```
git fetch upstream
git log upstream/main --oneline -10          # 上流に何が入ったか
git diff upstream/main -- irodori_tts/       # 学習・推論の本体との差
git diff --name-status upstream/main -- irodori_tts/ configs/ train.py infer.py
```

タグ（`v1` / `v2` / `v3`）も取得済み。上流は main を v4 系にし、過去の版は
タグで残す運用。

## 取り込み時点

- 2026-08-18 に `upstream/main` の `8224daf`（Update default model to
  v4.1-Small）まで取り込んだ。
- ファイルの欠落は無い。差分は下記の独自改変のみ。

最初に突き合わせたときは `8224daf` が触った `README.md` / `docs/parameters.md` /
`gradio_app.py` / `gradio_app_voicedesign.py` が旧内容のまま残っていた。
APP 側は v4.1 を既定にしているのに同梱の gradio と parameters.md は v4-Small を
指す、という食い違いが出ていたので、後から取り込み直した。
突き合わせるときはファイルの有無だけでなく中身も見ること。

## 独自改変（上流には無いもの）

上流を取り込み直すときは、これらを消さないこと。理由は各ファイルの
コメントにも書いてある。

### `irodori_tts/codec.py`（+131 行）

torchcodec が使えない環境のフォールバック。Windows では torchaudio の
バックエンド（torchcodec）が FFmpeg の DLL を見つけられず必ず失敗する。
上流は失敗するたびに例外を捕まえて soundfile に落ちる書き方で、生成の
たびに失敗する呼び出しを繰り返す。可否を 1 度だけ調べて使い回すようにした。
読み込みと書き込みは別経路なので判定を分けて持つ。

### `irodori_tts/inference_runtime.py`（+57 行）

1. **LoRA アダプタの LRU キャッシュ（上限 4 本）**
   PEFT はロードしたアダプタを自力で解放しない。ワーカーが常駐するので
   キャラを切り替えるたびに積み上がる。1 本あたり約 124 MiB、20 本で
   +2.3 GiB を実測。80 キャラ回すと約 +10 GB 成長し、8 GB GPU では
   10〜15 回の切り替えで OOM する。
   壊してはいけない不変条件: `delete_adapter` が例外を投げたとき、台帳から
   先に消してはならない。台帳だけ減ると PEFT 側に実体が残り、上限の勘定が
   狂って実体数が際限なく増える。

2. **torchcodec バックエンドの共有**
   `import torchaudio` を外し、codec 側の `load_audio` / `save_audio` を使う。
   読み書きで同じ判定を共有する。

### `configs/train_v4_small_lora.yaml`（1 行）

`dataloader_persistent_workers` を false → true。上流は v4 の 2 ファイルだけ
false で、他 14 ファイルは true。LoRA ジョブは 1 エポックが数バッチしかなく、
false だとエポックの切れ目ごとに全ワーカーが破棄される。Windows は spawn
なので起動のたびに torch を import し直し、GPU が待ちに入る
（実測 36.5 秒/step → 3.3 秒/step）。

### `pyproject.toml` / `requirements.txt`（1 行）

`sentencepiece` の上限 `<0.2` を外した。この上限だけが Python 3.12 を
塞いでいる（0.1.99 の Windows wheel は cp311 まで、0.2.0 には cp312 がある）。
Radeon 版 PyTorch は Windows では cp312 のみの配布なので、3.12 を選べる
必要がある。実測では sentencepiece は生成にも学習にも使われていない
（transformers 5 では extra 扱い。v3/v4 のトークナイザは tokenizer.json を
読む TokenizersBackend で、import を遮断しても通る）。

## 上流に無いもの（このリポジトリの追加）

- `APP/` — Electron アプリ一式
- `setup.bat` / `起動.bat` — Windows 向けの導入と起動
- `scripts/` — 検証用のスクリプト

## 経緯

`Irodori-TTS-v4/` に上流 main のコピーを同梱し、v4 だけ別 venv の
subprocess で動かす構成だった。上流 README が「main は v2/v3/VoiceDesign と
後方互換」と明記しており、実測でも上流 main 単体で全モデルが動いたので、
2026-08-17 に単一ツリーへ統合した（詳細はコミット 49186a7 以降）。
