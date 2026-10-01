"""書き起こしの前に、音声を声の塊に切る処理（server_audio._sound_chunks）。

モデルは使わないので、サーバー無しで走る。合成した音（鳴っている区間と
無音）で、どこで切れてどこで切れないかを確かめる。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

import torch  # noqa: E402

from server_audio import (  # noqa: E402
    CHUNK_MAX_SEC,
    _contained,
    _join_chunk_texts,
    _RewindEscape,
    _sound_chunks,
    _transcribe_whole_then_tail,
    _trim_silence,
    _cap_segments,
    CLIP_MAX_SEC,
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
    out = _trim_silence(audio, SR, floor_db=45.0, keep_sec=0.1)
    assert abs(len(out) / SR - 1.2) < 0.03


def test_trim_keeps_inner_pause() -> None:
    audio = np.concatenate([_tone(1.0), _silence(2.0), _tone(1.0)])
    kept = _trim_silence(audio, SR, floor_db=45.0, keep_sec=0.0)
    assert abs(len(kept) / SR - 4.0) < 0.03


def test_cap_cuts_long_clip_at_pauses_without_loss() -> None:
    # 70 s: 7 s of voice, 1 s of quiet, repeated
    audio = np.concatenate([np.concatenate([_tone(7.0), _silence(1.0)]) for _ in range(9)])[: SR * 70]
    out = _cap_segments([(0.0, 70.0)], audio, SR, 20.0)
    assert len(out) >= 4
    assert all(e - s <= 20.0 + 1e-6 for s, e in out)
    assert out[0][0] == 0.0 and out[-1][1] == 70.0
    assert all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(out, out[1:]))
    # cut points fall in the quiet stretches
    assert all((s % 8.0) >= 7.0 - 0.02 for s, _ in out[1:])


def test_cap_leaves_short_segments_alone() -> None:
    audio = _tone(40.0)
    segs = [(0.0, 12.0), (12.0, 12.0 + CLIP_MAX_SEC)]
    assert _cap_segments(segs, audio, SR, 20.0) == segs


def test_trim_leaves_silent_clip_alone() -> None:
    audio = _silence(1.0)
    assert len(_trim_silence(audio, SR, 45.0, 0.1)) == len(audio)


def _fake_run(by_length: dict):
    """書き起こしの代わり。渡された音声の長さに一番近い鍵（秒）の文を返す。"""
    return lambda piece: by_length[min(by_length, key=lambda k: abs(k - len(piece) / SR))]


def test_whole_clip_kept_when_it_covers_everything() -> None:
    audio = np.concatenate([_tone(1.5), _silence(0.9), _tone(2.0)])
    spans = _sound_chunks(audio, SR)
    run = _fake_run({4.4: "お待ちしておりました。何なりとお申し付けください", 2.1: "何なりとお申し付けください"})
    segs = _transcribe_whole_then_tail(run, audio, SR, spans)
    assert [s["text"] for s in segs] == ["お待ちしておりました。何なりとお申し付けください"]


def test_dropped_second_sentence_is_added_back() -> None:
    # 丸ごとだと 1 文目で止まる（anime-whisper の癖）。2 つ目の塊だけ足す
    audio = np.concatenate([_tone(1.5), _silence(0.9), _tone(2.0)])
    spans = _sound_chunks(audio, SR)
    run = _fake_run({4.4: "お待ちしておりました", 2.1: "コハルに、何なりとお申し付けください", 1.6: "お待ちしておりました"})
    text = _join_chunk_texts([s["text"] for s in _transcribe_whole_then_tail(run, audio, SR, spans)])
    assert text == "お待ちしておりました、コハルに、何なりとお申し付けください"


def test_short_reply_after_a_pause_is_added() -> None:
    # 間のあとの短い返事（「え?」）も、丸ごとの書き起こしに無ければ足す
    audio = np.concatenate([_tone(1.5), _silence(1.0), _tone(0.3)])
    spans = _sound_chunks(audio, SR)
    run = _fake_run({2.8: "オーナー様", 0.4: "え?", 1.6: "オーナー様"})
    text = _join_chunk_texts([s["text"] for s in _transcribe_whole_then_tail(run, audio, SR, spans)])
    assert text == "オーナー様、え?"


def test_chunk_with_nothing_read_is_skipped() -> None:
    audio = np.concatenate([_tone(1.5), _silence(1.0), _tone(0.3)])
    spans = _sound_chunks(audio, SR)
    run = _fake_run({2.8: "オーナー様", 0.4: "…", 1.6: "オーナー様"})
    assert [s["text"] for s in _transcribe_whole_then_tail(run, audio, SR, spans)] == ["オーナー様"]


def test_contained_ignores_spelling() -> None:
    assert _contained("コハルに", "こはるに、何なりと")
    assert not _contained("何なりとお申し付けください", "お待ちしておりました")


def test_rewind_escape_bans_the_ninth_repeat() -> None:
    proc = _RewindEscape()
    prompt = [1, 2, 3, 4]
    scores = torch.zeros(1, 10)
    proc(torch.tensor([prompt]), scores.clone())  # first call fixes the prompt length
    looped = proc(torch.tensor([prompt + [7] * 8]), scores.clone())
    assert looped[0, 7] == -float("inf")
    fine = proc(torch.tensor([prompt + [7] * 7]), scores.clone())
    assert fine[0, 7] == 0


if __name__ == "__main__":
    test_whole_clip_kept_when_it_covers_everything()
    test_dropped_second_sentence_is_added_back()
    test_short_reply_after_a_pause_is_added()
    test_chunk_with_nothing_read_is_skipped()
    test_contained_ignores_spelling()
    test_rewind_escape_bans_the_ninth_repeat()
    test_trim_keeps_margin_at_both_ends()
    test_trim_keeps_inner_pause()
    test_trim_leaves_silent_clip_alone()
    test_cap_cuts_long_clip_at_pauses_without_loss()
    test_cap_leaves_short_segments_alone()
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
