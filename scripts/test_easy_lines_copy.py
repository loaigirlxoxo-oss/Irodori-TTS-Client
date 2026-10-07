"""かんたん学習の台本の写しを、手を入れていないときだけ新しい同梱版に置き換える判定のテスト。
  python scripts/test_easy_lines_copy.py

server_easy を import すると起動時の後始末が走って記録を書き換えるので、判定の関数だけを取り出して動かす。
古い版の台本は git から取り出す（リポジトリの中で動かすこと）。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
PRESET_DIR = ROOT / "APP" / "presets"
SRC = (ROOT / "APP" / "server_easy.py").read_text(encoding="utf-8").replace("\r\n", "\n")
start = SRC.index("def _is_untouched_old_copy(")
ns: dict = {"json": json, "Path": Path, "PAST_LINES_FILE": PRESET_DIR / "easy_train_lines.past.json"}
exec(SRC[start:SRC.index("\n\n\n", start)], ns)  # noqa: S102
is_untouched_old_copy = ns["_is_untouched_old_copy"]


def old_version(commit: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:APP/presets/easy_train_lines.txt"], cwd=ROOT, capture_output=True, check=True).stdout


def write(data: bytes) -> Path:
    path = Path(tempfile.mkdtemp(prefix="easy_lines_test_")) / "easy_train_lines.txt"
    path.write_bytes(data)
    return path


def test_old_bundled_copy_is_replaced() -> None:
    # 1.2.2 に同梱していた台本のまま（手を入れていない）→ 置き換える。改行が CRLF でも LF でも同じ判定
    old = old_version("73344c2").replace(b"\r\n", b"\n")
    assert is_untouched_old_copy(write(old))
    assert is_untouched_old_copy(write(old.replace(b"\n", b"\r\n")))


def test_edited_copy_is_kept() -> None:
    # 1 文字でも直してあれば、利用者が手を入れた写し → 置き換えない
    old = old_version("73344c2").replace(b"\r\n", b"\n")
    assert not is_untouched_old_copy(write(old + "😊 自分で足したセリフ。\n".encode("utf-8")))


def test_current_bundled_copy_is_not_touched() -> None:
    # いまの同梱版と同じ写しは、置き換える必要が無い（過去の版の一覧に、いまの版を入れていない）
    current = (PRESET_DIR / "easy_train_lines.txt").read_bytes()
    assert not is_untouched_old_copy(write(current))
    past = json.loads((PRESET_DIR / "easy_train_lines.past.json").read_text(encoding="utf-8"))["sha256"]
    assert hashlib.sha256(current.replace(b"\r\n", b"\n")).hexdigest() not in past


def test_missing_file_is_not_touched() -> None:
    assert not is_untouched_old_copy(Path(tempfile.mkdtemp(prefix="easy_lines_test_")) / "none.txt")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
