"""CER（文字誤り率）の計算と、書き起こしの設定。

CER は純粋関数なのでサーバー無しで走る。書き起こしは設定の確認だけする
（実際に鳴らすのは通しテストの担当）。
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))

from easy_eval import cer, transcribe  # noqa: E402


def test_identical_is_zero() -> None:
    assert cer("こんにちは", "こんにちは") == 0.0


def test_one_substitution() -> None:
    # 5文字中1文字違い
    assert abs(cer("こんにちは", "こんにちば") - 0.2) < 1e-9


def test_one_deletion() -> None:
    assert abs(cer("こんにちは", "こんにち") - 0.2) < 1e-9


def test_one_insertion() -> None:
    assert abs(cer("こんにちは", "こんにちはは") - 0.2) < 1e-9


def test_completely_different() -> None:
    """全部違えば 1.0 以上。幻聴で長く返ると 1.0 を超える。"""
    assert cer("あいう", "かきく") == 1.0
    assert cer("あいう", "かきくけこさしす") > 1.0


def test_empty_reference() -> None:
    """元テキストが空なら比較にならない。0 除算で落とさない。"""
    assert cer("", "なにか") == 1.0
    assert cer("", "") == 0.0


def test_ignores_spaces_and_punctuation() -> None:
    """読点や空白の有無で誤差が出ると、読みの崩れを見誤る。"""
    assert cer("こんにちは、元気ですか", "こんにちは元気ですか") == 0.0
    assert cer("あい うえお", "あいうえお") == 0.0


def test_transcribe_pins_no_repeat_ngram() -> None:
    """既定の 0 だと幻聴で CER が 57% まで悪化した実績がある。4 に固定する。"""
    src = inspect.getsource(transcribe)
    assert "no_repeat_ngram_size" in src, "設定を渡していない"
    assert "4" in src, "4 に固定していない"


if __name__ == "__main__":
    test_identical_is_zero()
    test_one_substitution()
    test_one_deletion()
    test_one_insertion()
    test_completely_different()
    test_empty_reference()
    test_ignores_spaces_and_punctuation()
    test_transcribe_pins_no_repeat_ngram()
    print("OK")
