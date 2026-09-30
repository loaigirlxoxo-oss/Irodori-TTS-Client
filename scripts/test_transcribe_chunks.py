"""書き起こしの前に、音声を声の塊に切る処理（server_audio._sound_chunks）。

モデルは使わないので、サーバー無しで走る。合成した音（鳴っている区間と
無音）で、どこで切れてどこで切れないかを確かめる。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from server_audio import (  # noqa: E402
    CHUNK_MAX_SEC,
    _join_chunk_texts,
    _sound_chunks,
    _tag_nonverbal,
    _trim_silence,
)

SR = 16000


def _tone(sec: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)


def test_short_pause_stays_one_chunk() -> None:
    audio = np.concatenate([_tone(1.0), _silence(0.3), _tone(1.0)])
    assert len(_sound_chunks(audio, SR)) == 1


def test_long_pause_splits() -> None:
    # 2文目の前の長い間。ここで anime-whisper が止まっていた
    audio = np.concatenate([_tone(1.5), _silence(0.9), _tone(2.0)])
    chunks = _sound_chunks(audio, SR)
    assert len(chunks) == 2
    assert abs(chunks[1][0] - 2.4) < 0.02


def test_quiet_phrase_is_kept() -> None:
    # ピークより 30dB 小さい小声も塊として拾う（-45dB まで）
    audio = np.concatenate([_tone(1.0), _silence(1.0), _tone(1.0, amp=0.3 * 10 ** (-30 / 20))])
    assert len(_sound_chunks(audio, SR)) == 2


def test_click_is_dropped() -> None:
    audio = np.concatenate([_tone(1.0), _silence(1.0), _tone(0.05)])
    assert len(_sound_chunks(audio, SR)) == 1


def test_long_run_is_cut_under_encoder_window() -> None:
    audio = _tone(70.0)
    chunks = _sound_chunks(audio, SR)
    assert len(chunks) >= 3
    assert all(e - s <= CHUNK_MAX_SEC + 1e-6 for s, e in chunks)
    assert abs(chunks[-1][1] - 70.0) < 0.02


def test_silence_gives_no_chunk() -> None:
    assert _sound_chunks(_silence(2.0), SR) == []


def test_quiet_recording_is_still_transcribed() -> None:
    # 全体が小さい録音（ピーク約 -80dBFS）も、無音扱いで消さない
    quiet = 10 ** (-77 / 20)
    audio = np.concatenate([_tone(1.0, amp=quiet), _silence(1.0), _tone(1.0, amp=quiet)])
    assert len(_sound_chunks(audio, SR)) == 2


def test_hiss_next_to_speech_is_not_a_chunk() -> None:
    # 声のあるファイルでは、-70dBFS 未満のノイズは塊にしない
    hiss = (np.random.default_rng(0).standard_normal(int(2.0 * SR)) * 10 ** (-85 / 20)).astype(np.float32)
    audio = np.concatenate([_tone(1.0), hiss])
    assert len(_sound_chunks(audio, SR)) == 1


def test_join_adds_comma_only_between_unpunctuated() -> None:
    assert _join_chunk_texts(["お待ちしておりました", "何なりと"]) == "お待ちしておりました、何なりと"
    assert _join_chunk_texts(["はい。", "どうぞ"]) == "はい。どうぞ"
    assert _join_chunk_texts(["", "えっと…", "", "その"]) == "えっと…その"


def test_trim_keeps_margin_at_both_ends() -> None:
    audio = np.concatenate([_silence(0.8), _tone(1.0), _silence(0.8)])
    out = _trim_silence(audio, SR, floor_db=45.0, keep_sec=0.1, max_inner_sec=0.0)
    assert abs(len(out) / SR - 1.2) < 0.03


def test_trim_caps_inner_pause_only_when_asked() -> None:
    audio = np.concatenate([_tone(1.0), _silence(2.0), _tone(1.0)])
    kept = _trim_silence(audio, SR, floor_db=45.0, keep_sec=0.0, max_inner_sec=0.0)
    capped = _trim_silence(audio, SR, floor_db=45.0, keep_sec=0.0, max_inner_sec=0.5)
    assert abs(len(kept) / SR - 4.0) < 0.03
    assert abs(len(capped) / SR - 2.5) < 0.03


def test_trim_leaves_silent_clip_alone() -> None:
    audio = _silence(1.0)
    assert len(_trim_silence(audio, SR, 45.0, 0.1, 0.0)) == len(audio)


def test_tags_sighs_laughs_humming() -> None:
    assert _tag_nonverbal("ふぅ…") == "😮‍💨ふぅ…"
    assert _tag_nonverbal("はい、ふふっ") == "はい、🤭ふふっ"
    assert _tag_nonverbal("ふんふんふんふーん") == "🎵ふんふんふんふーん"
    # 相槌の「ふんふん」、語の途中の「ふう」には付けない。二度かけても増えない
    assert _tag_nonverbal("ふんふん") == "ふんふん"
    assert _tag_nonverbal("工夫う") == "工夫う"
    assert _tag_nonverbal(_tag_nonverbal("ふぅ…")) == "😮‍💨ふぅ…"


if __name__ == "__main__":
    test_trim_keeps_margin_at_both_ends()
    test_trim_caps_inner_pause_only_when_asked()
    test_trim_leaves_silent_clip_alone()
    test_tags_sighs_laughs_humming()
    test_short_pause_stays_one_chunk()
    test_long_pause_splits()
    test_quiet_phrase_is_kept()
    test_click_is_dropped()
    test_long_run_is_cut_under_encoder_window()
    test_silence_gives_no_chunk()
    test_quiet_recording_is_still_transcribed()
    test_hiss_next_to_speech_is_not_a_chunk()
    test_join_adds_comma_only_between_unpunctuated()
    print("OK")
