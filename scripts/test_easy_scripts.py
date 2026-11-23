"""性格の型ごとの台本（APP/presets/easy_scripts）が読めること。

サーバーを起動しなくても走る。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from server_easy import (  # noqa: E402
    MIN_LEN, SCRIPT_TYPES, LineError, load_lines, load_script, script_file, strip_style_tags,
)

COUNTS = {"kachiki": 313, "ottori": 277, "genki": 271, "cool": 251, "amaenbo": 241, "oneesan": 243, "mesugaki": 267}


def _load_text(text: str) -> list[dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        return load_script(tmp)
    finally:
        os.unlink(tmp)


def _expect_error(text: str, word: str) -> None:
    try:
        _load_text(text)
    except LineError as exc:
        assert word in str(exc), f"別の理由で落ちている: {exc}"
    else:
        raise AssertionError(f"弾けていない: {text!r}")


def test_every_type_loads() -> None:
    assert {key for key, _, _ in SCRIPT_TYPES} == set(COUNTS), "型の一覧と台本の数が合わない"
    for key, count in COUNTS.items():
        rows = load_script(script_file(key))
        assert len(rows) == count, f"{key}: {count} 行のはずが {len(rows)} 行"
        for row in rows:
            assert row["frame"] and row["caption"], f"{key}: 枠か説明文が無い: {row['text'][:20]}"
            assert strip_style_tags(row["tag"]) == "", f"{key}: タグに絵文字以外が入っている: {row['tag']}"
            assert row["text"].startswith(row["tag"]), f"{key}: 文がタグで始まっていない: {row['text'][:20]}"
            assert len(row["body"]) >= MIN_LEN


def test_moan_frame_uses_strength_two() -> None:
    """演技 LoRA の強さ 2.0 は、喘ぎの枠だけ（2026-10-10 に決めた）。"""
    for key in COUNTS:
        rows = load_script(script_file(key))
        moan = [r for r in rows if "喘ぎ" in r["frame"]]
        assert moan, f"{key}: 喘ぎの枠が無い"
        assert all(r["lora_scale"] == 2.0 for r in moan), f"{key}: 喘ぎの枠の強さが 2.0 でない"
        assert all(r["lora_scale"] == 1.0 for r in rows if "喘ぎ" not in r["frame"]), f"{key}: 喘ぎ以外の枠の強さが変わっている"


def test_spaces_inside_a_line_stay_in_the_body() -> None:
    """「😲百点！？ きみが！？」のように文の途中に空白がある行で、空白までをタグにしない。"""
    rows = _load_text("## 😲 驚き\n> 驚いている。\n😲百点！？ きみが！？ すごい、ほんとに！？\n")
    assert rows[0]["tag"] == "😲" and rows[0]["body"] == "百点！？ きみが！？ すごい、ほんとに！？", rows[0]
    assert rows[0]["text"] == "😲百点！？ きみが！？ すごい、ほんとに！？", "TTS に渡す文が、書いたとおりでない"


def test_old_format_reads_as_before() -> None:
    """見出しと「タグ 本文」だけの台本（easy_train_lines.txt の書式）は、前と同じ文になる。"""
    rows = _load_text("## 👂 囁き\n👂 ねえ、もう少しだけこっちに寄って。この声、あなたにだけ届けばいいの。\n")
    assert rows[0]["text"] == "👂 ねえ、もう少しだけこっちに寄って。この声、あなたにだけ届けばいいの。"
    assert rows[0]["caption"] == "" and rows[0]["lora_scale"] == 1.0
    assert load_lines()[0][0], "型を選ばないときの台本が読めない"


def test_frame_settings() -> None:
    rows = _load_text("## 🥵 喘ぎ\n> 喘いでいる。\n>> 説明文=素材づくり用の説明文。\n>> LoRA強さ=1.5\n🥵あっ、あ、あっ……だめ、そこ……っ\n"
                      "## 😊 楽しげ\n> 楽しげに話している。\n😊今日はいい天気だね。\n")
    assert rows[0]["caption"] == "素材づくり用の説明文。" and rows[0]["lora_scale"] == 1.5
    assert rows[1]["caption"] == "楽しげに話している。" and rows[1]["lora_scale"] == 1.0, "前の枠の設定が次の枠に残っている"


def test_rejects_broken_settings() -> None:
    _expect_error("## 🥵 喘ぎ\n>> つよさ=2\n🥵あっ、あ、あっ……だめ\n", "読めない設定")
    _expect_error("## 🥵 喘ぎ\n>> LoRA強さ=つよく\n🥵あっ、あ、あっ……だめ\n", "数ではありません")
    _expect_error("## 🥵 喘ぎ\n>> LoRA強さ=9\n🥵あっ、あ、あっ……だめ\n", "3 以下")


def test_rejects_unknown_type() -> None:
    try:
        script_file("../secret")
    except LineError as exc:
        assert "型がありません" in str(exc)
    else:
        raise AssertionError("知らない型を弾けていない")
    assert script_file("") is None and script_file(None) is None


if __name__ == "__main__":
    test_every_type_loads()
    test_moan_frame_uses_strength_two()
    test_spaces_inside_a_line_stay_in_the_body()
    test_old_format_reads_as_before()
    test_frame_settings()
    test_rejects_broken_settings()
    test_rejects_unknown_type()
    print("OK")
