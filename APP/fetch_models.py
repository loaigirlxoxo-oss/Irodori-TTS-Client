"""Pre-download every model the desktop app needs.

The app fetches these lazily on first use, which means the first synthesis
sits silently for several minutes on a cold cache. Running this during setup
moves that wait into the install step, where a progress bar is expected.

Everything lands in the shared HuggingFace cache (``HF_HOME``), so re-running
is cheap: already-present files are skipped.
"""
from __future__ import annotations

import sys

# (repo_id, filename or None for whole-repo, label)
#
# 全て必須。「使わない人には要らない」という切り分けはしない。
#   - VoiceDesign は Synthesize / Train / LoRAマージ の3画面で選択肢に出る。
#     選んだ時点で必要になり、無ければその場でエラーになる。
#   - anime-whisper は Dataset タブの書き起こしが from_pretrained で直接読む。
#     無いとデータセットを作れず、LoRA 学習の入口ごと塞がる。
# UI に出ている機能が使えない状態で「セットアップ完了」と言わないため、
# 1つでも欠けたら失敗として扱う。
TARGETS = [
    ("Aratako/Irodori-TTS-500M-v2", "model.safetensors", "音声モデル v2"),
    ("Aratako/Irodori-TTS-500M-v3", "model.safetensors", "音声モデル v3"),
    ("Aratako/Irodori-TTS-v4-Small", "model.safetensors", "音声モデル v4"),
    # v4.1 は v4 の duration predictor だけを差し替えたもので、上流の推奨。
    # v4 も残すのは、既存の LoRA を作った条件で鳴らし直せるようにするため。
    ("Aratako/Irodori-TTS-v4.1-Small", "model.safetensors", "音声モデル v4.1"),
    (
        "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        "model.safetensors",
        "音声モデル VoiceDesign",
    ),
    (
        "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        "model.safetensors",
        "音声モデル VoiceDesign v3",
    ),
    ("Aratako/Semantic-DACVAE-Japanese-32dim", None, "音声コーデック DACVAE"),
    # トークナイザは世代で違う。configs/train_*_lora.yaml の
    # text_tokenizer_repo / caption_tokenizer_repo が実体で、
    #   v4 / v4.1        → sbintuitions/modernbert-ja-310m
    #   v2 / v3 と両 VoiceDesign → llm-jp/llm-jp-3-150m
    # モデル本体(model.safetensors)には同梱されないので別途要る。
    # 欠けると HF_HUB_OFFLINE=1 のもとで生成が 500 になる（実際に起きた）。
    # 使うのはトークナイザだけ。重み(約1GB)は要らない。
    ("llm-jp/llm-jp-3-150m", "tokenizer.json", "日本語トークナイザ v2/v3"),
    ("llm-jp/llm-jp-3-150m", "tokenizer_config.json", "日本語トークナイザ v2/v3(設定)"),
    ("llm-jp/llm-jp-3-150m", "config.json", "日本語トークナイザ v2/v3(定義)"),
    ("llm-jp/llm-jp-3-150m", "special_tokens_map.json", "日本語トークナイザ v2/v3(特殊記号)"),
    ("sbintuitions/modernbert-ja-310m", "tokenizer.json", "日本語トークナイザ v4"),
    ("sbintuitions/modernbert-ja-310m", "tokenizer_config.json", "日本語トークナイザ v4(設定)"),
    ("sbintuitions/modernbert-ja-310m", "config.json", "日本語トークナイザ v4(定義)"),
    ("sbintuitions/modernbert-ja-310m", "special_tokens_map.json", "日本語トークナイザ v4(特殊記号)"),
    ("litagin/anime-whisper", None, "書き起こしモデル（Dataset タブ用）"),
    # 生成音声に電子透かしを入れる。無くても生成は続くが警告が出て透かしが入らない。
    # 取らずにおくと初回生成時に裏で 68MB 落ちてきて、その分だけ待たされる。
    #
    # ファイル名を明示するのは、下の filename=None 経路が .ckpt を除外するため。
    # silentcipher には safetensors が無く実体は .ckpt だけなので、全体取得に
    # 任せると重みが 1 つも来ない。
    # アプリが使うのは 44.1k だけだが、silentcipher は snapshot_download で
    # リポジトリ全体を見るので 16k も揃えておく（欠けると差分取得が走る）。
    (
        "sony/silentcipher",
        "44_1_khz/73999_iteration/hparams.yaml",
        "電子透かし 44.1k(設定)",
    ),
    ("sony/silentcipher", "44_1_khz/73999_iteration/enc_c.ckpt", "電子透かし 44.1k(符号器)"),
    ("sony/silentcipher", "44_1_khz/73999_iteration/dec_c.ckpt", "電子透かし 44.1k(復号器)"),
    ("sony/silentcipher", "44_1_khz/73999_iteration/dec_m_0.ckpt", "電子透かし 44.1k(検出器)"),
    ("sony/silentcipher", "44_1_khz/73999_iteration/opt.ckpt", "電子透かし 44.1k(最適化状態)"),
    ("sony/silentcipher", "16_khz/97561_iteration/hparams.yaml", "電子透かし 16k(設定)"),
    ("sony/silentcipher", "16_khz/97561_iteration/enc_c.ckpt", "電子透かし 16k(符号器)"),
    ("sony/silentcipher", "16_khz/97561_iteration/dec_c.ckpt", "電子透かし 16k(復号器)"),
    ("sony/silentcipher", "16_khz/97561_iteration/dec_m_0.ckpt", "電子透かし 16k(検出器)"),
    ("sony/silentcipher", "16_khz/97561_iteration/opt.ckpt", "電子透かし 16k(最適化状態)"),
    ("sony/silentcipher", "config.json", "電子透かし(定義)"),
]


def fetch(repo_id: str, filename: str | None) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    if filename:
        hf_hub_download(repo_id=repo_id, filename=filename)
        return

    # リポジトリ全体が要るものは、snapshot_download ではなくファイルを1つずつ取る。
    #
    #   snapshot_download は取得後にスナップショット配下へシンボリックリンクを張る。
    #   Windows で開発者モードが無効・非管理者だとこれが WinError 1314
    #   （クライアントは要求された特権を保有していません）で失敗する。
    #   配布先の多くはこの条件に当てはまるため、素の hf_hub_download で回す。
    #   （こちらは同じキャッシュに入るがリンクを張らない）
    api = HfApi()
    skip_ext = (".bin", ".h5", ".ckpt", ".msgpack", ".onnx")
    for sibling in api.model_info(repo_id).siblings:
        name = sibling.rfilename
        if name.endswith(skip_ext):
            continue  # 重みの別形式は使わない
        if name.startswith("."):
            continue  # .gitattributes 等のメタファイルは不要
        hf_hub_download(repo_id=repo_id, filename=name)


RETRIES = 3
RETRY_WAIT_SECONDS = 5


def fetch_with_retry(repo_id: str, filename: str | None) -> None:
    """一時的な回線断で数GBの取得を捨てないよう、少し粘ってから諦める。"""
    import time

    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            fetch(repo_id, filename)
            return
        except Exception as exc:  # noqa: BLE001 - 理由を問わず再試行する
            last = exc
            if attempt < RETRIES:
                print(f"再試行 {attempt}/{RETRIES - 1} ... ", end="", flush=True)
                time.sleep(RETRY_WAIT_SECONDS * attempt)  # 5s, 10s
    assert last is not None
    raise last


def main() -> int:
    total = len(TARGETS)
    failed: list[str] = []

    for i, (repo_id, filename, label) in enumerate(TARGETS, 1):
        print(f"[{i}/{total}] {label} ... ", end="", flush=True)
        try:
            fetch_with_retry(repo_id, filename)
            print("OK")
        except Exception as exc:  # noqa: BLE001 - ここは理由を問わず記録して続行
            print("失敗")
            print(f"        {type(exc).__name__}: {exc}")
            failed.append(label)

    print()
    if failed:
        print("次のモデルの取得に失敗しました:")
        for name in failed:
            print(f"  - {name}")
        print()
        print("回線を確認してから setup.bat を再実行してください。")
        print("取得済みのぶんは飛ばすので、続きから再開します。")
        return 1

    print("モデルの取得が完了しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
