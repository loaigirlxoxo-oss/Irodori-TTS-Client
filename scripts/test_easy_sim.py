"""話者類似度（SIM）。

同一音声で 1.0 に近く、別人で低く出ること。モデルを読むので時間がかかる。
  python scripts/test_easy_sim.py <a.wav> <b.wav>
  （a と b は別人の音声）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from easy_eval import speaker_similarity  # noqa: E402


def test_same_file_is_near_one(a: str) -> None:
    score = speaker_similarity(a, a)
    assert score > 0.99, f"同一音声なのに {score:.3f}"


def test_symmetric(a: str, b: str) -> None:
    """順番を入れ替えても同じ値。コサイン類似度なので対称のはず。"""
    ab = speaker_similarity(a, b)
    ba = speaker_similarity(b, a)
    assert abs(ab - ba) < 1e-4, f"非対称: {ab:.4f} vs {ba:.4f}"


def test_different_speaker_is_lower(a: str, b: str) -> None:
    same = speaker_similarity(a, a)
    diff = speaker_similarity(a, b)
    assert diff < same, f"別人 {diff:.3f} が同一 {same:.3f} を下回らない"


def test_range(a: str, b: str) -> None:
    """コサイン類似度なので -1〜1 の範囲に収まること。

    float32 の丸めで 1.0 をわずかに超えることがあるので少しだけ許容する。
    """
    eps = 1e-5
    for x, y in ((a, a), (a, b)):
        score = speaker_similarity(x, y)
        assert -1.0 - eps <= score <= 1.0 + eps, f"範囲外: {score}"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("使い方: test_easy_sim.py <a.wav> <b.wav>（別人の音声2本）")
    a, b = sys.argv[1], sys.argv[2]
    test_same_file_is_near_one(a)
    test_symmetric(a, b)
    test_different_speaker_is_lower(a, b)
    test_range(a, b)
    print("OK")
