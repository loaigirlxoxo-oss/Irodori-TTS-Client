"""素材づくりで /synthesize に足す項目（説明文、演技 LoRA、強さ）が、台本と選択から正しく決まること。

サーバーを起動しなくても走る。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from server_easy import (  # noqa: E402
    ACTING_CFG_SPEAKER, ACTING_SPEAKER_KV, CER_MIN_LEN, GenerateRequest, generation_extra, load_script, script_file,
    strip_style_tags,
)

LORA = "演技LoRA"
VOICE = "少し高い声の少女。"


def _row(script: str, frame_word: str) -> dict:
    return next(r for r in load_script(script_file(script)) if frame_word in r["frame"])


def test_nothing_is_added_without_script_and_lora() -> None:
    """型も演技 LoRA も選ばなければ、これまでと同じリクエストになる。"""
    req = GenerateRequest(ref_wav="x.wav")
    for row in load_script()[:20]:
        assert generation_extra(row, req) == {}, f"何かが足されている: {generation_extra(row, req)}"


def test_caption_is_voice_plus_frame() -> None:
    req = GenerateRequest(ref_wav="x.wav", script="kachiki", voice_caption=VOICE)
    row = _row("kachiki", "怒り")
    assert generation_extra(row, req) == {"caption": VOICE + "怒りを込めて、強い口調で話している。"}


def test_acting_lora_and_strength() -> None:
    req = GenerateRequest(ref_wav="x.wav", script="kachiki", acting_lora=LORA)
    plain, moan = generation_extra(_row("kachiki", "怒り"), req), generation_extra(_row("kachiki", "喘ぎ"), req)
    assert plain["lora_name"] == LORA and plain["lora_scale"] == 1.0
    assert moan["lora_name"] == LORA and moan["lora_scale"] == 2.0, "喘ぎの枠の強さが 2.0 で渡っていない"
    for extra in (plain, moan):
        assert extra["speaker_kv_scale"] == ACTING_SPEAKER_KV and extra["cfg_scale_speaker"] == ACTING_CFG_SPEAKER


def test_short_lines_exist_and_are_marked_by_length() -> None:
    """型ごとの台本には、読み間違いの判定を外す短い行がある（判定の境目は CER_MIN_LEN）。"""
    rows = load_script(script_file("cool"))
    short = [r for r in rows if len(strip_style_tags(r["body"])) < CER_MIN_LEN]
    assert 0 < len(short) < len(rows), f"短い行の数がおかしい: {len(short)} / {len(rows)}"


def test_redo_replaces_only_dropped_clips() -> None:
    """落ちた本だけを、同じ設定で作り直して同じ場所に置きかえ、評価値を外す。"""
    import tempfile

    import server_easy

    folder = Path(tempfile.mkdtemp(prefix="easy_redo_test_"))
    scored = []
    for i in range(3):
        path = folder / f"{i:04d}.wav"
        path.write_bytes(b"old")
        scored.append({"path": str(path), "text": f"😊せりふ{i}", "sim": 0.5, "cer": 0.0, "eval_failed": False,
                       "extra": {"caption": f"説明文{i}"}})
    kept = [scored[0], scored[2]]
    calls: list[tuple] = []

    def fake_synth(text, ref_wav, model_type, lora=None, seed=None, watermark=False, extra=None):
        calls.append((text, ref_wav, model_type, watermark, extra))
        fresh = folder / "fresh.wav"
        fresh.write_bytes(b"new")
        return str(fresh)

    saved = server_easy._synthesize_one
    server_easy._synthesize_one = fake_synth
    try:
        count = server_easy._redo_dropped(scored, kept, GenerateRequest(ref_wav="ref.wav", model_type="v4_large"))
    finally:
        server_easy._synthesize_one = saved
    assert count == 1 and calls == [("😊せりふ1", "ref.wav", "v4_large", False, {"caption": "説明文1"})], calls
    assert Path(scored[1]["path"]).read_bytes() == b"new" and Path(scored[0]["path"]).read_bytes() == b"old"
    assert "sim" not in scored[1] and "cer" not in scored[1], "作り直した本に前の評価値が残っている"
    assert scored[0]["sim"] == 0.5, "通った本の評価値まで外している"


if __name__ == "__main__":
    test_redo_replaces_only_dropped_clips()
    test_nothing_is_added_without_script_and_lora()
    test_caption_is_voice_plus_frame()
    test_acting_lora_and_strength()
    test_short_lines_exist_and_are_marked_by_length()
    print("OK")
