"""学習が止まったときの文（server_train.failure_message）のテスト。サーバーも GPU も使わない。
  python scripts/test_train_failure_message.py

server_train を import するとサーバーの初期化が走るので、関数と正規表現だけを取り出して動かす。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SRC = (Path(__file__).resolve().parent.parent / "APP" / "server_train.py").read_text(encoding="utf-8").replace("\r\n", "\n")
start = SRC.index("_OOM_RE = ")
end = SRC.index("\n\n\n", SRC.index("def failure_message("))
ns: dict = {"re": re}
exec(SRC[start:end], ns)  # noqa: S102 - 自分のリポジトリの関数を取り出すだけ
failure_message = ns["failure_message"]

OOM_LOG = [
    "[job] spawning: python train.py ...",
    "VRAM cap: 11.2 GB of 12.0 GB, free now 11.7 GB (overflow fails instead of spilling to system RAM).",
    "step=25 loss=0.91",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.20 GiB.",
]


def test_oom_says_cause_and_what_to_do() -> None:
    msg = failure_message(1, OOM_LOG, {"batch_size": 4, "gradient_accumulation_steps": 8,
                                       "ref_max_seconds": 120.0, "gradient_checkpointing": False})
    assert msg.startswith("VRAM が足りず、学習を止めました（使えた量: 11.2GB）。次のどれかを試してください。"), msg
    assert "バッチサイズを 4 から 2 に減らし、勾配の蓄積を 8 から 16 に増やす" in msg, msg
    assert "参照音声の上限を短くする（いま 120秒）" in msg and "勾配チェックポイントを入れる" in msg, msg
    assert "ほかのアプリを閉じて" in msg, msg


def test_oom_skips_tips_that_do_not_apply() -> None:
    # バッチが 1 なら減らせない。勾配チェックポイントが入っていれば勧めない
    msg = failure_message(1, OOM_LOG, {"batch_size": 1, "gradient_accumulation_steps": 32,
                                       "ref_max_seconds": None, "gradient_checkpointing": True})
    assert "バッチサイズ" not in msg and "勾配チェックポイント" not in msg, msg
    assert "参照音声の上限を短くする" in msg and "（いま" not in msg, msg


def test_other_failure_shows_last_error_line() -> None:
    log = ["step=10 loss=0.9", "Traceback (most recent call last):", "  File ...", "FileNotFoundError: manifest.jsonl が無い"]
    msg = failure_message(1, log, {"batch_size": 4, "gradient_accumulation_steps": 8})
    assert msg == "学習が途中で止まりました（終了コード 1）。\n原因: FileNotFoundError: manifest.jsonl が無い", msg


def test_unknown_failure_points_to_log() -> None:
    msg = failure_message(-9, ["step=10 loss=0.9"], {"batch_size": 4, "gradient_accumulation_steps": 8})
    assert "終了コード -9" in msg and "記録" in msg, msg


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
