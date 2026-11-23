"""参照音声の上限を、モデルに入っている値に任せていることのテスト。モデルも GPU も使わない。
  python scripts/test_ref_max_seconds.py

以前はサーバーが 30 秒を固定で渡していて、v4 系（モデルの値は 120 秒）でも 30 秒を超える参照音声が切られていた。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "APP"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
os.chdir(APP)

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from irodori_tts import inference_runtime as rt  # noqa: E402


class _Captured(Exception):
    """生成の依頼を組み立てた所で止めるための印。"""


def _request_built_by(path: str, body: dict) -> dict:
    """サーバーが生成のために組み立てた依頼の中身を返す。モデルは読まず、生成もしない。"""
    seen: dict = {}

    def fake_request(**kwargs):
        seen.update(kwargs)
        raise _Captured

    original = (server.resolve_and_load_model, server.SamplingRequest)
    server.resolve_and_load_model = lambda *a, **k: (object(), None)
    server.SamplingRequest = fake_request
    try:
        TestClient(server.app, raise_server_exceptions=False).post(path, json=body)
    finally:
        server.resolve_and_load_model, server.SamplingRequest = original
    assert seen, f"{path} が生成の依頼を組み立てる所まで進まなかった"
    return seen


def test_engine_uses_model_value_when_cap_is_not_given() -> None:
    assert rt._default_max_ref_seconds({"ref_max_seconds": 120.0}) == 120.0     # v4 系のモデルに入っている値


def test_engine_falls_back_to_30_for_models_without_value() -> None:
    assert rt._default_max_ref_seconds({}) == 30.0                              # v2・v3
    assert rt._default_max_ref_seconds(None) == 30.0


def test_synthesize_does_not_fix_the_cap() -> None:
    built = _request_built_by("/api/v1/synthesize/", {"text": "テスト", "model_type": "v4_large"})
    assert built["max_ref_seconds"] is None, built["max_ref_seconds"]


def test_openai_speech_does_not_fix_the_cap() -> None:
    built = _request_built_by("/v1/audio/speech", {"input": "テスト", "model": "v4_large"})
    assert built["max_ref_seconds"] is None, built["max_ref_seconds"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
