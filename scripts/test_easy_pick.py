"""チェックポイントの選び方（かんたん学習の採用・学習タブのおすすめ）。

全チェックポイントから「似ている度 − 0.5 × 読み間違い」が最も高いもの。
サーバー無しで走る。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from easy_eval import combined_score, pick_by_score  # noqa: E402


def test_early_checkpoint_can_win() -> None:
    """前半が一番良ければ前半を選ぶ（コハルは 1300 中 500 が最良だった）。"""
    rows = [
        {"name": "checkpoint_0500", "step": 500, "sim": 0.70, "cer": 0.04},
        {"name": "checkpoint_1300", "step": 1300, "sim": 0.71, "cer": 0.15},
    ]
    assert pick_by_score(rows)["name"] == "checkpoint_0500"


def test_misreads_outweigh_small_similarity_gain() -> None:
    """読み間違いが5ポイント多ければ、似ている度が 0.02 高くても負ける。"""
    a = {"name": "a", "step": 100, "sim": 0.50, "cer": 0.05}
    b = {"name": "b", "step": 200, "sim": 0.52, "cer": 0.10}
    assert combined_score(a) > combined_score(b)
    assert pick_by_score([a, b])["name"] == "a"


def test_similarity_decides_when_misreads_are_close() -> None:
    a = {"name": "a", "step": 100, "sim": 0.50, "cer": 0.050}
    b = {"name": "b", "step": 200, "sim": 0.53, "cer": 0.055}
    assert pick_by_score([a, b])["name"] == "b"


def test_tie_goes_to_later_step() -> None:
    a = {"name": "a", "step": 100, "sim": 0.5, "cer": 0.1}
    b = {"name": "b", "step": 200, "sim": 0.5, "cer": 0.1}
    assert pick_by_score([a, b])["name"] == "b"


def test_empty_raises() -> None:
    try:
        pick_by_score([])
    except ValueError:
        return
    raise AssertionError("空なら ValueError のはず")


if __name__ == "__main__":
    test_early_checkpoint_can_win()
    test_misreads_outweigh_small_similarity_gain()
    test_similarity_decides_when_misreads_are_close()
    test_tie_goes_to_later_step()
    test_empty_raises()
    print("OK")
