"""「分割しない」で 30 秒を超える音声の切れ方（server_audio._cap_segments）のテスト。モデルも GPU も使わない。
  python scripts/test_split_none_cap.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
APP = Path(__file__).resolve().parent.parent / "APP"
sys.path.insert(0, str(APP))
os.chdir(APP)
import server_audio as sa  # noqa: E402

SR = 16000


def speech_with_pauses(total_sec: float, talk_sec: float = 4.0, pause_sec: float = 0.5) -> np.ndarray:
    """talk_sec 話して pause_sec 黙る、を total_sec まで繰り返す音（発言の切れ目がある台詞の代わり）。"""
    rng = np.random.default_rng(0)
    out = np.zeros(int(total_sec * SR), dtype=np.float32)
    t = 0.0
    while t < total_sec:
        a, b = int(t * SR), int(min(total_sec, t + talk_sec) * SR)
        out[a:b] = rng.normal(0, 0.2, b - a)
        t += talk_sec + pause_sec
    return out


def lengths(total_sec: float, piece_max: float) -> list[float]:
    audio = speech_with_pauses(total_sec)
    segs = sa._cap_segments([(0.0, total_sec)], audio, SR, sa.CLIP_MAX_SEC, piece_max, 45.0)
    assert abs(segs[0][0]) < 1e-6 and abs(segs[-1][1] - total_sec) < 1e-6, segs          # 何も捨てない
    assert all(abs(a[1] - b[0]) < 1e-6 for a, b in zip(segs, segs[1:])), segs             # すき間なくつながる
    return [round(e - s, 1) for s, e in segs]


def test_under_30_is_kept_whole() -> None:
    assert lengths(29.0, sa.CLIP_MAX_SEC) == [29.0]


def test_none_cuts_only_what_exceeds_30() -> None:
    # 45 秒: 「分割しない」は 30 秒を上限に切る → 2 つ。どちらも 30 秒以下（切れ目は発言の直前なので、ちょうど 30 秒にはならない）
    got = lengths(45.0, sa.CLIP_MAX_SEC)
    assert len(got) == 2 and all(x <= sa.CLIP_MAX_SEC for x in got), got


def test_old_behaviour_cut_at_max_sec() -> None:
    # 直す前は「最長」の 20 秒で刻んでいた → 3 つ
    got = lengths(45.0, 20.0)
    assert len(got) == 3 and all(x <= 20.0 for x in got), got


def test_split_passes_30_when_method_is_none() -> None:
    src = (APP / "server_audio.py").read_text(encoding="utf-8")
    assert 'piece_max = req.max_sec if req.method in ("vad", "level") else CLIP_MAX_SEC' in src


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
