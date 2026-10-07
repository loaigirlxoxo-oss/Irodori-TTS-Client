"""かんたん学習の素材の選別（easy_eval.screen_clips）と、評価の似ている度（score_lines）のテスト。

GPU もモデルも使わない。似ている度と書き起こしは差し替える。
  python scripts/test_easy_screen.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))
import easy_eval  # noqa: E402
from easy_eval import CLIP_KEEP_MIN, combined_score, score_lines, screen_clips  # noqa: E402


def run(sims: list[float], cers: list[float], broken: set[int] = frozenset()):
    """i 本目の似ている度と読み間違いを sims[i]・cers[i] にして選別する。broken の本は測定に失敗させる。"""
    clips = [{"path": str(i), "text": "あ" * 10} for i in range(len(sims))]

    def fake_sim(_ref, path):
        if int(path) in broken:
            raise RuntimeError("測れない")
        return sims[int(path)]

    def fake_cer(_text, heard):
        return cers[int(heard)]

    saved = easy_eval.speaker_similarity, easy_eval.transcribe, easy_eval.cer
    easy_eval.speaker_similarity, easy_eval.transcribe, easy_eval.cer = fake_sim, (lambda path: path), fake_cer
    try:
        return screen_clips(clips, "ref.wav")
    finally:      # ほかのテストが本物の cer を使えるように戻す
        easy_eval.speaker_similarity, easy_eval.transcribe, easy_eval.cer = saved


def test_enough_pass() -> None:
    # 10本のうち 8本が基準を通る → 通った 8本だけ。足さない
    kept, rep = run([0.8] * 8 + [0.5, 0.5], [0.0] * 10)
    assert len(kept) == 8 and rep["kept"] == 8 and rep["topped_up"] == 0, rep


def test_top_up_to_70_percent_not_all() -> None:
    # 10本のうち 4本しか通らない → 全部（10本）ではなく、上位 7本（通った 4本＋落ちた中で点数の高い 3本）
    sims = [0.8] * 10
    cers = [0.0] * 4 + [0.30, 0.90, 0.40, 0.80, 0.50, 0.70]     # 5本目以降は読み間違いで落ちる
    kept, rep = run(sims, cers)
    want = math.ceil(10 * CLIP_KEEP_MIN)
    assert len(kept) == want == 7 and rep["kept"] == 7 and rep["topped_up"] == 3, rep
    assert [c["path"] for c in kept] == ["0", "1", "2", "3", "4", "6", "8"], [c["path"] for c in kept]   # 元の順のまま
    worst_kept = min(combined_score(c) for c in kept)
    assert all(combined_score(c) <= worst_kept for c in run(sims, cers)[0] if c not in kept)


def test_unmeasured_clips_are_never_added() -> None:
    # 測れなかった本は、足りなくても足さない。割合は測れた本に対して数える
    sims = [0.8] * 10
    cers = [0.0] * 2 + [0.5] * 8
    kept, rep = run(sims, cers, broken={9})
    assert rep["failed"] == 1 and "9" not in [c["path"] for c in kept], rep
    assert len(kept) == math.ceil(9 * CLIP_KEEP_MIN), rep


def test_similarity_uses_short_lines_too() -> None:
    # 評価の似ている度は、短い文も含めた全部の平均（20字未満を外さない）
    rows = [{"tag": "😊", "body": "えっ！", "heard": "えっ！", "sim": 0.2},
            {"tag": "😊", "body": "あ" * 40, "heard": "あ" * 40, "sim": 0.8}]
    sim, _ = score_lines(rows)
    assert abs(sim - 0.5) < 1e-9, sim


def test_measure_no_longer_skips_short_lines() -> None:
    src = (Path(__file__).resolve().parent.parent / "APP" / "server_easy.py").read_text(encoding="utf-8")
    assert "SIM_MIN_CHARS" not in src and "sim = speaker_similarity(ref_wav, wav)\n" in src.replace("\r\n", "\n")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
