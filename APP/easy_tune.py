"""参照ボイスの微調整。

VoiceDesign はキャプション（文章）でしか声を指定できず、狙った高さや声色に
一発で当たるとは限らない。ここは生成した音声そのものに手を入れて、最後の
詰めをするための層。

実装は Praat（parselmouth）の Change gender を使う。高さとフォルマントを
別々に動かせて、なにより再合成の劣化が小さい。WORLD でも同じことはできるが、
分解して戻すだけで本人性が落ちる。ECAPA で測った実測値:

                  WORLD    Praat
    無加工        0.8641   0.9844
    +1.5半音      0.8235   0.9520
    +3.0半音      0.8020   0.9003

WORLD は無加工ですら Praat の +3半音より悪い。参照ボイスは402本すべての
基準になるので、ここでの劣化は LoRA にそのまま乗る。

注意: それでも加工は無料ではない。まずキャプションと引き直しで近づけて、
最後の詰めにだけ使う。
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

# Praat の解析範囲。日本語話者の声はこの中に収まる。
PITCH_FLOOR = 60.0
PITCH_CEIL = 1000.0


# 加工前に確保しておく余裕。フィルタやピッチ操作は瞬間的に振幅を持ち上げる
# ので、0dBFS に張り付いたまま通すと頭打ちになってパチパチ鳴る。
HEADROOM_PEAK = 0.7


def _load_mono(path: str) -> tuple[np.ndarray, int, float]:
    """モノラルの float64 と、元に戻すための倍率を返す。

    Change gender はモノラルしか受け取らない。あわせて、加工で溢れないよう
    あらかじめ下げておく。元ファイルが 0dBFS に張り付いていることがあり
    （実測: 参照ボイスがピーク 1.000）、そのまま通すと確実に割れる。
    """
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = np.ascontiguousarray(wav, dtype=np.float64)

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    scale = (HEADROOM_PEAK / peak) if peak > HEADROOM_PEAK else 1.0
    return wav * scale, int(sr), scale


def _apply_tilt(wav: np.ndarray, sr: int, brightness: float) -> np.ndarray:
    """高域の棚を上げ下げして明るさを動かす。

    ハイシェルビングを双二次フィルタで掛け、前後方向に通して位相のずれを
    打ち消す（filtfilt）。音声全体を1回の FFT で処理すると時間変化を無視
    するうえ、端が不連続になる。
    """
    from scipy import signal

    gain_db = float(np.clip(brightness, -5.0, 5.0)) * 1.2  # 1段あたり 1.2dB
    if abs(gain_db) < 0.01:
        return wav

    # 3kHz を境に、上を持ち上げ下げする
    f0 = 3000.0
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / 0.9 - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha

    b = np.array([
        a * ((a + 1) + (a - 1) * cos_w0 + two_sqrt_a_alpha),
        -2 * a * ((a - 1) + (a + 1) * cos_w0),
        a * ((a + 1) + (a - 1) * cos_w0 - two_sqrt_a_alpha),
    ])
    aa = np.array([
        (a + 1) - (a - 1) * cos_w0 + two_sqrt_a_alpha,
        2 * ((a - 1) - (a + 1) * cos_w0),
        (a + 1) - (a - 1) * cos_w0 - two_sqrt_a_alpha,
    ])
    return signal.filtfilt(b / aa[0], aa / aa[0], wav)


def _add_breath(wav: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """息の成分を足す。

    一様なノイズを足すと「サーッ」と乗るだけで息には聞こえないので、元の
    振幅に追従させる。ただし追従を強くしすぎると、ノイズが出たり消えたりして
    途切れて聞こえる。下駄を履かせて、無音でも完全には消えないようにする。

    ノイズの正規化に最大値を使うと、乱数の外れ値（実測 peak 4.1）に引きずられて
    実効レベルが 1% まで落ちる。RMS で合わせる。
    """
    from scipy import signal

    if amount <= 0:
        return wav

    win = max(1, int(sr * 0.03))  # 30ms。短いと出入りが細かくなりすぎる
    env = np.convolve(np.abs(wav), np.ones(win) / win, mode="same")
    peak_env = float(env.max()) or 1.0
    env = env / peak_env
    # 完全に 0 にしない。無音区間で途切れるのを避ける。
    env = 0.35 + 0.65 * env

    noise = np.random.default_rng(0).normal(0.0, 1.0, wav.size)
    # 2.5kHz 以上を残す。バターワースを前後方向に通して位相を崩さない。
    sos = signal.butter(4, 2500.0, btype="highpass", fs=sr, output="sos")
    noise = signal.sosfiltfilt(sos, noise)
    rms = float(np.sqrt((noise ** 2).mean())) or 1.0
    noise = noise / rms

    # 元の実効レベルに対する比で足す。1 段あたり約 3%。
    src_rms = float(np.sqrt((wav ** 2).mean())) or 1.0
    return wav + noise * env * (src_rms * amount * 0.03)


def tune(
    src_path: str,
    dst_path: str,
    *,
    pitch_semitones: float = 0.0,
    formant: float = 0.0,
    breathiness: float = 0.0,
    brightness: float = 0.0,
) -> str:
    """参照ボイスに微調整をかけて別ファイルに書く。

    pitch_semitones: 声の高さ。半音単位。
    formant:         声色。-5〜+5 程度。正で細く、負で太くなる。
    breathiness:     息の多さ。0〜5 程度。
    brightness:      明るさ。-5〜+5 程度。
    """
    import parselmouth
    from parselmouth.praat import call

    wav, sr, scale = _load_mono(src_path)

    if pitch_semitones or formant:
        snd = parselmouth.Sound(wav, sampling_frequency=sr)
        pitch = call(snd, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEIL)
        median = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")

        # new_pitch_median に 0 を渡すと「変えない」。倍率で動かしたいので
        # いまの中央値に掛けて絶対値で渡す。
        if pitch_semitones and median and not np.isnan(median):
            new_median = median * (2.0 ** (pitch_semitones / 12.0))
        else:
            new_median = 0.0

        # -5〜+5 を 0.94〜1.06 倍に写す。実測で 0.03/段（±15%）は動きすぎで、
        # 「太さ 2」で本人性が 0.76 まで落ちた。微調整の幅に収める。
        formant_ratio = 1.0 + formant * 0.012

        shifted = call(
            snd, "Change gender",
            PITCH_FLOOR, PITCH_CEIL, formant_ratio, new_median, 1.0, 1.0,
        )
        wav = np.asarray(shifted.values).reshape(-1).astype(np.float64)

    if brightness:
        wav = _apply_tilt(wav, sr, brightness)
    if breathiness:
        wav = _add_breath(wav, sr, breathiness)

    # 下げたぶんを戻す。ただし加工で伸びていたら、そこは戻しきらない。
    if scale != 1.0:
        wav = wav / scale
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.99:
        wav = wav * (0.99 / peak)

    sf.write(dst_path, wav.astype(np.float32), sr)
    return dst_path


def is_identity(
    pitch_semitones: float = 0.0,
    formant: float = 0.0,
    breathiness: float = 0.0,
    brightness: float = 0.0,
) -> bool:
    """全部ゼロなら加工しない。通すだけで品質が落ちる。"""
    return not any((pitch_semitones, formant, breathiness, brightness))
