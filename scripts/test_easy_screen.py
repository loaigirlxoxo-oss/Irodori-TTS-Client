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


def run(sims: list[float], cers: list[float], broken: set[int] = frozenset(), extra: list[dict] | None = None,
        scored_out: list | None = None):
    """i 本目の似ている度と読み間違いを sims[i]・cers[i] にして選別する。broken の本は測定に失敗させる。

    extra[i] は、i 本目の本に足す項目（枠の印など）。
    """
    clips = [dict({"path": str(i), "text": "あ" * 10}, **(extra[i] if extra else {})) for i in range(len(sims))]

    def fake_sim(_ref, path):
        if int(path) in broken:
            raise RuntimeError("測れない")
        return sims[int(path)]

    def fake_cer(_text, heard):
        return cers[int(heard)]

    saved = easy_eval.clip_similarity, easy_eval.transcribe, easy_eval.cer
    easy_eval.clip_similarity, easy_eval.transcribe, easy_eval.cer = fake_sim, (lambda path: path), fake_cer
    try:
        return screen_clips(clips, "ref.wav", scored_out=scored_out)
    finally:      # ほかのテストが本物の cer を使えるように戻す
        easy_eval.clip_similarity, easy_eval.transcribe, easy_eval.cer = saved


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


def test_floor_is_per_frame() -> None:
    """声の近さの足切りは枠ごと。音だけの枠は全体より低く出るが、その枠の中で外れていなければ残す。"""
    sims = [0.75] * 10 + [0.30] * 9 + [0.10]          # 怒りの枠 10 本、舐める音の枠 10 本（うち 1 本だけ外れ）
    extra = [{"frame": "😠 怒り"}] * 10 + [{"frame": "👅 舐める音"}] * 10
    kept, rep = run(sims, [0.0] * 20, extra=extra)
    assert [c["path"] for c in kept] == [str(i) for i in range(19)], [c["path"] for c in kept]
    assert rep["dropped_sim"] == 1 and rep["topped_up"] == 0, rep
    assert abs(rep["floors"]["👅 舐める音"] - (0.30 - easy_eval.CLIP_SIM_MARGIN)) < 1e-9, rep["floors"]
    # 枠の印が無ければ、これまでどおり全体で 1 本の線になり、舐める音の 10 本がまとめて落ちる
    _, flat = run(sims, [0.0] * 20)
    assert flat["dropped_sim"] == 10, flat


def test_small_frames_share_a_floor_by_kind() -> None:
    """本数の少ない枠は、同じ種類の少ない枠とまとめて中央値を取る。"""
    sims = [0.70] * 5 + [0.72] * 4 + [0.40] + [0.30] * 5        # 言葉の枠が 2 つ（5 本ずつ）、音の枠が 1 つ（5 本）
    extra = [{"frame": "🫶 優しく"}] * 5 + [{"frame": "😏 からかう"}] * 5 + [{"frame": "💋 リップノイズ"}] * 5
    kept, rep = run(sims, [0.0] * 15, extra=extra)
    assert rep["floors"]["🫶 優しく"] == rep["floors"]["😏 からかう"] != rep["floors"]["💋 リップノイズ"], rep["floors"]
    assert "9" not in [c["path"] for c in kept] and len(kept) == 14, [c["path"] for c in kept]
    assert easy_eval.frame_kind("🥵 喘ぎ") == "喘ぎ" and easy_eval.frame_kind("👃 嗅ぐ音") == "音" and easy_eval.frame_kind("😠 怒り") == "言葉"


def test_short_clips_skip_the_reading_check() -> None:
    """短い行は、読み間違いの判定を外す（書き起こしが幻聴を起こしても落とさない）。"""
    extra = [{"short": True}] * 3 + [{}] * 7
    kept, rep = run([0.8] * 10, [0.9] * 3 + [0.0] * 7, extra=extra)
    assert len(kept) == 10 and rep["dropped_cer"] == 0, rep
    assert all(c["cer"] == 0.0 for c in kept[:3])


def test_moan_and_lip_frames_skip_the_reading_check() -> None:
    """喘ぎとリップノイズの枠は、読み間違いの判定を外す。ほかの枠は外さない。"""
    extra = [{"frame": "🥵 喘ぎ"}] * 8 + [{"frame": "💋 リップノイズ"}] * 8 + [{"frame": "😠 怒り"}] * 8
    kept, rep = run([0.6] * 24, [0.9] * 24, extra=extra)
    assert rep["dropped_cer"] == 8, rep
    assert {c["frame"] for c in kept if c["cer"] == 0.0} == {"🥵 喘ぎ", "💋 リップノイズ"}, [c["frame"] for c in kept]
    assert easy_eval.skips_reading_check({"frame": "👅 舐める音"}) is False and easy_eval.skips_reading_check({}) is False


def test_measured_clips_are_not_measured_again() -> None:
    """sim と cer を持っている本は測り直さない（落ちた本だけ作り直して、もう一度選別するため）。"""
    scored: list = []
    run([0.8] * 9 + [0.2], [0.0] * 10, scored_out=scored)
    assert len(scored) == 10 and all("sim" in c for c in scored)
    scored[9] = {"path": "9", "text": "あ" * 10}                  # 落ちた 1 本を作り直した、という想定

    calls: list[str] = []

    def counting_sim(_ref, path):
        calls.append(path)
        return 0.8

    saved = easy_eval.clip_similarity, easy_eval.transcribe, easy_eval.cer
    easy_eval.clip_similarity, easy_eval.transcribe, easy_eval.cer = counting_sim, (lambda path: path), (lambda _t, _h: 0.0)
    try:
        kept, rep = screen_clips(scored, "ref.wav")
    finally:
        easy_eval.clip_similarity, easy_eval.transcribe, easy_eval.cer = saved
    assert calls == ["9"], f"作り直していない本まで測っている: {calls}"
    assert len(kept) == 10 and rep["dropped_sim"] == 0, rep


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
