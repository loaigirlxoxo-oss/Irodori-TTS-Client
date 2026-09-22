"""かんたん学習のセリフプリセットが正しく読めること。

サーバーを起動しなくても走る。壊れたプリセットを生成前に弾けるかを見る。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from server_easy import LineError, load_eval_lines, load_lines  # noqa: E402


def test_loads_402_lines() -> None:
    lines = load_lines()
    assert len(lines) == 402, f"402本のはずが {len(lines)} 本"


def test_every_line_has_tag_and_body() -> None:
    for tag, body in load_lines():
        assert tag, "タグが空"
        assert 20 <= len(body) <= 200, f"{len(body)}字: {body[:30]}"


def test_rejects_foreign_characters() -> None:
    """日本語を書くときにハングルやキリルが紛れることがある。読ませる前に弾く。"""
    # 1文字目だけハングル（운）にした行。残りは普通の日本語。
    broken = "## test\n\U0001f600 운動不足を痛感している。これはテスト用の行です。\n"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write(broken)
        tmp = f.name
    try:
        load_lines(tmp)
    except LineError as exc:
        assert "他言語" in str(exc), f"別の理由で落ちている: {exc}"
    else:
        raise AssertionError("他言語混入を検出できていない")
    finally:
        os.unlink(tmp)


def test_rejects_short_line() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write("## test\n\U0001f600 みじかい\n")
        tmp = f.name
    try:
        load_lines(tmp)
    except LineError as exc:
        assert "範囲外" in str(exc), f"別の理由で落ちている: {exc}"
    else:
        raise AssertionError("短すぎる行を検出できていない")
    finally:
        os.unlink(tmp)


def test_rejects_wide_space_spelling() -> None:
    """全角スペースで一文字ずつ区切る表記は変に鳴る（実機確認済み）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write("## test\n\U0001f600 だ　い　じ　ょ　う　ぶ。あ　わ　て　な　く　て、い　い　か　ら。\n")
        tmp = f.name
    try:
        load_lines(tmp)
    except LineError as exc:
        assert "全角スペース" in str(exc), f"別の理由で落ちている: {exc}"
    else:
        raise AssertionError("全角スペース区切りを検出できていない")
    finally:
        os.unlink(tmp)


def test_eval_lines_count_and_spread() -> None:
    """評価文は10本。長さを3階層に散らす。"""
    rows = load_eval_lines()
    assert len(rows) == 10, f"10本のはずが {len(rows)} 本"
    lens = sorted(len(body) for _, body in rows)
    short = [n for n in lens if n < 60]
    mid = [n for n in lens if 60 <= n < 120]
    long_ = [n for n in lens if n >= 120]
    assert (len(short), len(mid), len(long_)) == (3, 4, 3), (
        f"配分が違う 短{len(short)} 中{len(mid)} 長{len(long_)}: {lens}"
    )


def test_eval_lines_differ_from_train() -> None:
    """学習に使った文で評価すると、覚えた文を読ませることになる。"""
    train = {body for _, body in load_lines()}
    for _, body in load_eval_lines():
        assert body not in train, f"学習用と重複: {body[:30]}"


if __name__ == "__main__":
    test_loads_402_lines()
    test_every_line_has_tag_and_body()
    test_rejects_foreign_characters()
    test_rejects_short_line()
    test_rejects_wide_space_spelling()
    test_eval_lines_count_and_spread()
    test_eval_lines_differ_from_train()
    print("OK")
