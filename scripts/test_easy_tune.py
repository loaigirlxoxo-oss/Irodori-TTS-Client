"""参照ボイスの微調整が、狙い通りに効いているか。

サーバー不要。WORLD で分解して戻すだけなので数秒で終わる。
  python scripts/test_easy_tune.py <wav>
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from easy_tune import is_identity, tune  # noqa: E402


def _f0_series(path: str) -> np.ndarray:
    """フレームごとの基本周波数。無声は 0。"""
    import parselmouth

    from easy_tune import PITCH_CEIL, PITCH_FLOOR

    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEIL)
    return np.asarray(pitch.selected_array["frequency"])


def _pitch_ratio(src: str, dst: str) -> float:
    """高さが何倍になったか。

    単純に平均を比べてはいけない。再合成で有声と判定される区間が増えると、
    加工していなくても平均が動く。元が有声だったフレームだけを突き合わせる。
    """
    a, b = _f0_series(src), _f0_series(dst)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    both = (a > 0) & (b > 0)
    assert both.sum() > 20, "比較できる有声フレームが少なすぎる"
    return float(np.median(b[both] / a[both]))


def _centroid(path: str) -> float:
    """スペクトル重心。明るさの代理指標。"""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    return float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())


def test_identity_guard() -> None:
    assert is_identity(), "全部ゼロなら加工しない判定のはず"
    assert not is_identity(pitch_semitones=1.0)


def test_pitch_goes_up(src: str, tmp: Path) -> None:
    """+4半音で F0 がおよそ 1.26 倍（2^(4/12)）になること。"""
    out = tune(src, str(tmp / "pitch.wav"), pitch_semitones=4.0)
    ratio = _pitch_ratio(src, out)
    print(f"  高さ  : {ratio:.3f}倍（理論 1.260）")
    assert 1.20 < ratio < 1.32, f"狙いから外れている: {ratio:.3f}"


def test_roundtrip_keeps_pitch(src: str, tmp: Path) -> None:
    """何も指定しなければ高さは変わらない（分解して戻すだけ）。"""
    out = tune(src, str(tmp / "same.wav"))
    ratio = _pitch_ratio(src, out)
    print(f"  往復  : {ratio:.4f}倍（1.0 が理想）")
    assert 0.99 < ratio < 1.01, f"往復で高さが動いている: {ratio:.4f}"


def test_brightness_raises_centroid(src: str, tmp: Path) -> None:
    lo = _centroid(tune(src, str(tmp / "dark.wav"), brightness=-5))
    hi = _centroid(tune(src, str(tmp / "bright.wav"), brightness=5))
    print(f"  明るさ: 重心 {lo:.0f} -> {hi:.0f} Hz")
    assert hi > lo, f"明るさが効いていない: {lo:.0f} vs {hi:.0f}"


def test_formant_changes_timbre_not_pitch(src: str, tmp: Path) -> None:
    """フォルマントを動かしても高さは変わらないこと。ここが位相ボコーダとの差。"""
    out = tune(src, str(tmp / "formant.wav"), formant=5.0)
    ratio = _pitch_ratio(src, out)
    print(f"  声色  : F0 {ratio:.4f}倍（1.0 のままが正しい）")
    assert 0.99 < ratio < 1.01, f"フォルマント操作で高さが動いている: {ratio:.4f}"


def test_quality_is_kept(src: str, tmp: Path) -> None:
    """加工しても本人性が保たれること。

    参照ボイスは402本すべての基準になるので、ここでの劣化は LoRA に乗る。
    WORLD 実装は無加工ですら 0.86 まで落ちたので、その水準には戻さない。
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))
    from easy_eval import speaker_similarity

    same = speaker_similarity(src, tune(src, str(tmp / "q0.wav")))
    up = speaker_similarity(src, tune(src, str(tmp / "q15.wav"), pitch_semitones=1.5))
    print(f"  本人性: 無加工 {same:.4f} / +1.5半音 {up:.4f}")
    assert same > 0.95, f"無加工なのに劣化している: {same:.4f}"
    assert up > 0.90, f"+1.5半音で崩れている: {up:.4f}"


def test_no_clipping(src: str, tmp: Path) -> None:
    out = tune(src, str(tmp / "loud.wav"), pitch_semitones=3, breathiness=5, brightness=5)
    y, _ = sf.read(out, always_2d=False)
    peak = float(np.max(np.abs(y)))
    print(f"  ピーク: {peak:.3f}")
    assert peak <= 1.0, f"クリップしている: {peak:.3f}"


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else ""
    assert src, "wav のパスを引数で渡す"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_identity_guard()
        test_roundtrip_keeps_pitch(src, tmp)
        test_pitch_goes_up(src, tmp)
        test_formant_changes_timbre_not_pitch(src, tmp)
        test_brightness_raises_centroid(src, tmp)
        test_no_clipping(src, tmp)
        test_quality_is_kept(src, tmp)
    print("OK")
