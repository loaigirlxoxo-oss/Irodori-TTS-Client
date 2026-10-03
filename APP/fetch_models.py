"""Pre-download every model the desktop app needs.

The app fetches these lazily on first use, which means the first synthesis
sits silently for several minutes on a cold cache. Running this during setup
moves that wait into the install step, where a progress bar is expected.

Everything lands in the shared HuggingFace cache (``HF_HOME``), so re-running
is cheap: already-present files are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

# (repo_id, filename or None for whole-repo, label)
#
# どのモデルを選んでも要る部品。選ばせずに必ず取る。音声モデル本体は
# model_catalog.py にあり、取得するものを利用者が選ぶ（初回セットアップと設定画面）。
#   - anime-whisper は Dataset タブの書き起こしが from_pretrained で直接読む。
#     無いとデータセットを作れず、LoRA 学習の入口ごと塞がる。
# UI に出ている機能が使えない状態で「セットアップ完了」と言わないため、
# 1つでも欠けたら失敗として扱う。
COMMON_TARGETS = [
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


# ECAPA-TDNN（話者の似ている度）。かんたん学習が、生成した素材の選別と
# チェックポイントの採用に使う。
#
# これだけ HF キャッシュではなく models/ecapa に実体で置く。speechbrain の
# from_hparams が「1つのフォルダに全部入っている」前提で読むため。
# 上の TARGETS に載せられないのは、汎用の取得経路が .ckpt を除外していて、
# ECAPA は重みが .ckpt しかないから。
from data_paths import models_root

ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_DIR = models_root() / "ecapa"
ECAPA_FILES = (
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
)


def fetch_ecapa() -> None:
    from huggingface_hub import hf_hub_download

    ECAPA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ECAPA_FILES:
        hf_hub_download(
            repo_id=ECAPA_REPO,
            filename=name,
            local_dir=str(ECAPA_DIR),
        )


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


def with_retry(call) -> None:
    """一時的な回線断で数GBの取得を捨てないよう、少し粘ってから諦める。"""
    import time

    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            call()
            return
        except Exception as exc:  # noqa: BLE001 - 理由を問わず再試行する
            last = exc
            if attempt < RETRIES:
                print(f"再試行 {attempt}/{RETRIES - 1} ... ", end="", flush=True)
                time.sleep(RETRY_WAIT_SECONDS * attempt)  # 5s, 10s
    assert last is not None
    raise last


def fetch_with_retry(repo_id: str, filename: str | None) -> None:
    with_retry(lambda: fetch(repo_id, filename))


def model_targets(model_ids: list[str]) -> list[tuple[str, str, str]]:
    import model_catalog

    out = []
    for mid in model_ids:
        m = model_catalog.BY_ID[mid]
        for repo, filename in m["files"]:
            out.append((repo, filename, f"音声モデル {m['label']}" + ("" if filename.endswith("model.safetensors") else "（付属ファイル）")))
    return out


def parse_model_ids(argv: list[str]) -> list[str]:
    """--models id,id,...。無ければ以前の版と同じ一式（setup.bat の開発版はこれ）。"""
    import model_catalog

    if "--models" in argv:
        raw = argv[argv.index("--models") + 1] if argv.index("--models") + 1 < len(argv) else ""
        ids = [x for x in raw.split(",") if x]
    else:
        ids = list(model_catalog.LEGACY_IDS)
    unknown = [x for x in ids if x not in model_catalog.BY_ID]
    if unknown:
        raise SystemExit(f"知らないモデル: {', '.join(unknown)}")
    return ids


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # 設定画面から1つだけ取るときは、共通の部品はもう揃っているので取らない。
    with_common = "--no-common" not in args
    targets = model_targets(parse_model_ids(args)) + (COMMON_TARGETS if with_common else [])
    total = len(targets) + (1 if with_common else 0)
    failed: list[str] = []

    for i, (repo_id, filename, label) in enumerate(targets, 1):
        print(f"[{i}/{total}] {label} ... ", end="", flush=True)
        try:
            fetch_with_retry(repo_id, filename)
            print("OK")
        except Exception as exc:  # noqa: BLE001 - ここは理由を問わず記録して続行
            print("失敗")
            print(f"        {type(exc).__name__}: {exc}")
            failed.append(label)

    if with_common:
        print(f"[{total}/{total}] 話者照合モデル ECAPA（かんたん学習用） ... ",
              end="", flush=True)
        try:
            # 他と同じだけ粘る。1ファイルの瞬断でセットアップ全体を終わらせない。
            with_retry(fetch_ecapa)
            print("OK")
        except Exception as exc:  # noqa: BLE001
            print("失敗")
            print(f"        {type(exc).__name__}: {exc}")
            failed.append("話者照合モデル ECAPA")

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
