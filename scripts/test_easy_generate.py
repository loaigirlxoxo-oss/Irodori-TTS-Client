"""参照音声を固定した一括生成。

サーバーを起動してから実行する。
  IRODORI_PORT=<port> python scripts/test_easy_generate.py <ref.wav>
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

PORT = os.environ.get("IRODORI_PORT", "8080")
API = f"http://127.0.0.1:{PORT}/api/v1"


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


def wait(job_id: str, limit_sec: int = 900) -> dict:
    deadline = time.time() + limit_sec
    while time.time() < deadline:
        time.sleep(3)
        s = get(f"/easy/jobs/{job_id}")
        if s["state"] in ("done", "failed"):
            return s
    raise AssertionError("時間内に終わらなかった")


def test_generates_requested_count(ref: str) -> None:
    job = post("/easy/generate", {"ref_wav": ref, "limit": 3, "model_type": "v4_1"})
    s = wait(job["job_id"])
    assert s["state"] == "done", f"state={s['state']} error={s.get('error')}"
    assert s["done"] == 3, f"3本のはずが {s['done']} 本"
    assert len(s["clips"]) == 3, f"clips が {len(s['clips'])} 件"


def test_reports_progress(ref: str) -> None:
    """進捗が total と done で取れること。UI の進捗バーがこれを使う。"""
    job = post("/easy/generate", {"ref_wav": ref, "limit": 2, "model_type": "v4_1"})
    s = get(f"/easy/jobs/{job['job_id']}")
    assert s["total"] == 2, f"total が {s['total']}"
    wait(job["job_id"])


def test_rejects_missing_ref() -> None:
    """扱えない場所の参照音声を弾くこと。

    403 が返る。outputs / voices / datasets / inputs の外は、存在するか
    どうかを確かめる前に断っている（CORS を全開にしているので、外の
    ページに「そこにファイルがあるか」を教えないため）。
    """
    try:
        post("/easy/generate", {"ref_wav": "Z:/does/not/exist.wav", "limit": 1})
    except urllib.error.HTTPError as e:
        assert e.code == 403, f"403 のはずが {e.code}"
    else:
        raise AssertionError("扱えない場所の参照音声を弾いていない")


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else ""
    assert ref, "参照音声のパスを引数で渡す"
    test_rejects_missing_ref()
    test_generates_requested_count(ref)
    test_reports_progress(ref)
    print("OK")
