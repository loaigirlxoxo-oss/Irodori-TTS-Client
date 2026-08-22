"""Phase A end-to-end: split -> transcribe -> create dataset -> CRUD -> delete."""
from __future__ import annotations
import json
import sys
import time
import urllib.request

API = "http://127.0.0.1:8080/api/v1"
LONG_REF = "D:/Irodori-TTS/APP/references/voice_a3c763fd.wav"
DATASET_NAME = "PhaseA_smoke"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def put(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(API + path) as r:
        return json.loads(r.read().decode("utf-8"))


def delete(path: str) -> dict:
    req = urllib.request.Request(API + path, method="DELETE")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    print("=== 1) Split long reference ===")
    t0 = time.time()
    split_res = post("/audio/split", {"path": LONG_REF, "min_sec": 1.5, "max_sec": 15})
    chunks = split_res["chunks"]
    print(f"  {len(chunks)} chunks in {time.time()-t0:.1f}s")
    for c in chunks:
        print(f"    [{c['index']}] {c['duration']}s  {c['path']}")

    print("\n=== 2) Transcribe each chunk ===")
    clips_payload = []
    for c in chunks:
        t0 = time.time()
        tr = post("/audio/transcribe", {"path": c["path"]})
        text = (tr["text"] or "").strip()
        print(f"  [{c['index']}] {time.time()-t0:.1f}s  {text!r}")
        clips_payload.append({"path": c["path"], "text": text})

    print("\n=== 3) Create dataset ===")
    res = post("/datasets", {
        "name": DATASET_NAME,
        "clips": clips_payload,
        "source_files": [LONG_REF],
        "notes": "Phase A smoke test",
        "overwrite": True,
    })
    print(f"  {res}")

    print("\n=== 4) List datasets ===")
    res = get("/datasets")
    for d in res["datasets"]:
        marker = " <-- ours" if d["name"] == DATASET_NAME else ""
        print(f"  {d['name']}  clips={d['num_clips']}{marker}")

    print("\n=== 5) Get dataset detail ===")
    res = get(f"/datasets/{DATASET_NAME}")
    for c in res["clips"]:
        print(f"  [{c['index']}] {c['duration']}s  text={c['text']!r}")

    if res["clips"]:
        first_idx = res["clips"][0]["index"]
        print(f"\n=== 6) Update clip {first_idx} text ===")
        put(f"/datasets/{DATASET_NAME}/clips/{first_idx}", {"text": "EDITED_BY_TEST"})
        check = get(f"/datasets/{DATASET_NAME}")
        for c in check["clips"]:
            if c["index"] == first_idx:
                print(f"  -> {c['text']!r}")

        last_idx = res["clips"][-1]["index"]
        print(f"\n=== 7) Delete clip {last_idx} ===")
        delete(f"/datasets/{DATASET_NAME}/clips/{last_idx}")
        check = get(f"/datasets/{DATASET_NAME}")
        print(f"  num_clips after delete: {check['meta']['num_clips']}")

    print("\n=== 8) Delete dataset ===")
    res = delete(f"/datasets/{DATASET_NAME}")
    print(f"  {res}")

    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
