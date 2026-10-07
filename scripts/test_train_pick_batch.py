"""おまかせ設定のバッチの見積もり（server_train.pick_batch）のテスト。サーバーも GPU も使わない。
  python scripts/test_train_pick_batch.py

server_train を import するとサーバーの初期化が走るので、関数と定数だけを取り出して動かす。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SRC = (Path(__file__).resolve().parent.parent / "APP" / "server_train.py").read_text(encoding="utf-8").replace("\r\n", "\n")
ns: dict = {}
for name in ("_PROBE_CANDIDATES", "_PROBE_USABLE"):
    line = next(l for l in SRC.splitlines() if l.startswith(name + " = "))
    exec(line, ns)  # noqa: S102
start = SRC.index("def pick_batch(")
exec(SRC[start:SRC.index("\n\n\n", start)], ns)  # noqa: S102
pick_batch = ns["pick_batch"]


def test_measured_values_from_rtx5080() -> None:
    # 実測（RTX 5080、2026-10-07）: v4-Large はバッチ 4 で 8.82GB、8 で 11.84GB。上限 15.1GB の 85% は 12.8GB
    assert pick_batch({"batch": 4, "setup_gib": 4.95, "peak_gib": 8.82, "cap_gib": 15.1}) == 8
    assert pick_batch({"batch": 8, "setup_gib": 4.95, "peak_gib": 11.84, "cap_gib": 15.1}) == 8


def test_smaller_gpu_gets_smaller_batch() -> None:
    # 同じ 1本あたりの量（約 0.97GB）で、上限が 11.2GB（12GB の GPU）なら 4、7.4GB（8GB の GPU）なら 1
    m = {"batch": 4, "setup_gib": 4.95, "peak_gib": 8.82}
    assert pick_batch(dict(m, cap_gib=11.2)) == 4
    assert pick_batch(dict(m, cap_gib=7.4)) == 1


def test_ran_but_over_margin_steps_down() -> None:
    # 動きはしたが、上限の 85% を超えた → 1つ小さいバッチ
    assert pick_batch({"batch": 8, "setup_gib": 4.95, "peak_gib": 14.5, "cap_gib": 15.1}) == 4


def test_nothing_fits() -> None:
    assert pick_batch({"batch": 1, "setup_gib": 7.0, "peak_gib": 7.9, "cap_gib": 7.4}) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
