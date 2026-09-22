"""生成した音声をデータセットにするところまで。

サーバーを起動してから実行する。
  IRODORI_PORT=<port> python scripts/test_easy_dataset.py <ref.wav>

テスト用のデータセットは毎回作り直す。残しておくと次の実行で
「すでにある」で落ちるため。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

PORT = os.environ.get("IRODORI_PORT", "8080")
API = f"http://127.0.0.1:{PORT}/api/v1"
# "_" 始まりはステージング用に予約されていて 400 になる。
DATASET = "easy-test-tmp"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def delete(path: str) -> None:
    req = urllib.request.Request(API + path, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise


def wait(job_id: str, limit_sec: int = 900) -> dict:
    deadline = time.time() + limit_sec
    while time.time() < deadline:
        time.sleep(3)
        s = get(f"/easy/jobs/{job_id}")
        if s["state"] in ("done", "failed"):
            return s
    raise AssertionError("時間内に終わらなかった")


def test_creates_dataset_from_generated(ref: str) -> None:
    delete(f"/datasets/{DATASET}")
    job = post("/easy/generate", {
        "ref_wav": ref, "limit": 3, "model_type": "v4_1", "dataset": DATASET,
    })
    s = wait(job["job_id"])
    assert s["state"] == "done", f"state={s['state']} error={s.get('error')}"
    assert s.get("dataset") == DATASET, f"dataset が記録されていない: {s.get('dataset')}"

    ds = get(f"/datasets/{DATASET}")
    got = ds["meta"]["num_clips"]
    assert got == 3, f"3クリップのはずが {got}"
    assert len(ds["clips"]) == 3, f"clips が {len(ds['clips'])} 件"


def test_transcript_matches_lines(ref: str) -> None:
    """書き起こしを通さず、生成に使ったテキストがそのまま入ること。"""
    ds = get(f"/datasets/{DATASET}")
    clips = ds.get("clips") or []
    assert clips, "clips が空"
    for c in clips:
        text = c.get("text", "")
        assert len(text) >= 20, f"本文が短い: {text!r}"
        assert not text.startswith("\U0001f4d6"), "タグが本文に混ざっている"


def test_dataset_is_trainable(ref: str) -> None:
    """auto_config が推奨値を出せる＝学習に渡せる形になっていること。"""
    cfg = get(f"/datasets/{DATASET}/auto_config")
    rec = cfg.get("recommended") or {}
    assert rec.get("max_steps", 0) > 0, f"推奨ステップが出ない: {rec}"


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else ""
    assert ref, "参照音声のパスを引数で渡す"
    test_creates_dataset_from_generated(ref)
    test_transcript_matches_lines(ref)
    test_dataset_is_trainable(ref)
    print("OK")
