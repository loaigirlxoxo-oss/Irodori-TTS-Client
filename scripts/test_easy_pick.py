"""チェックポイントの選び方。

総ステップの60%未満は見ない。SIM/CER で選ぶと時々学習の浅いものが当たる
ため、構造的に防ぐ。サーバー無しで走る。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from easy_eval import pick_checkpoint  # noqa: E402

ROWS = [
    {"name": "checkpoint_0200", "step": 200, "sim": 0.95, "cer": 0.03},
    {"name": "checkpoint_1400", "step": 1400, "sim": 0.87, "cer": 0.041},
    {"name": "checkpoint_1600", "step": 1600, "sim": 0.91, "cer": 0.038},
    {"name": "checkpoint_1800", "step": 1800, "sim": 0.93, "cer": 0.225},
    {"name": "checkpoint_2000", "step": 2000, "sim": 0.89, "cer": 0.044},
]


def test_ignores_early_checkpoints() -> None:
    """SIM が最高でも、60%未満なら選ばない。"""
    got = pick_checkpoint(ROWS, max_steps=2000)
    assert got["name"] != "checkpoint_0200", "60%未満を選んでいる"


def test_picks_highest_sim_after_cer_filter() -> None:
    """CER 足切りの後、SIM が最大のものを選ぶ。"""
    got = pick_checkpoint(ROWS, max_steps=2000)
    assert got["name"] == "checkpoint_1600", f"選ばれたのは {got['name']}"
    assert got["fallback"] is False


def test_cer_limit_rejects_broken_reading() -> None:
    """SIM が最高でも CER が閾値超えなら落とす（1800 は sim 0.93 だが cer 22.5%）。"""
    got = pick_checkpoint(ROWS, max_steps=2000)
    assert got["name"] != "checkpoint_1800", "読みが崩れたものを選んでいる"


def test_falls_back_to_final_when_all_rejected() -> None:
    """全部足切りに落ちたら最終を使い、そのことを伝える。"""
    rows = [dict(r, cer=0.5) for r in ROWS]
    got = pick_checkpoint(rows, max_steps=2000)
    assert got["name"] == "checkpoint_2000", f"最終に落ちていない: {got['name']}"
    assert got["fallback"] is True


def test_uses_all_when_nothing_is_late() -> None:
    """save_every が粗くて60%以降が1つも無いときに、何も選べなくならない。"""
    rows = [{"name": "checkpoint_0100", "step": 100, "sim": 0.8, "cer": 0.05}]
    got = pick_checkpoint(rows, max_steps=2000)
    assert got["name"] == "checkpoint_0100"


def test_empty_rows() -> None:
    try:
        pick_checkpoint([], max_steps=2000)
    except ValueError:
        pass
    else:
        raise AssertionError("空のときに何か返している")


if __name__ == "__main__":
    test_ignores_early_checkpoints()
    test_picks_highest_sim_after_cer_filter()
    test_cer_limit_rejects_broken_reading()
    test_falls_back_to_final_when_all_rejected()
    test_uses_all_when_nothing_is_late()
    test_empty_rows()
    print("OK")
