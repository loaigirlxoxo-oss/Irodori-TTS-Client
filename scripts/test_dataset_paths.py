"""Verify dataset save honors target_dir AND copies source files."""
from __future__ import annotations
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8080/api/v1"
LONG_REF = "D:/Irodori-TTS/APP/references/voice_a3c763fd.wav"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
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
    print("=== Splitting + transcribing 3 clips from long ref ===")
    sr = post("/audio/split", {"path": LONG_REF, "min_sec": 1.5, "max_sec": 15})
    chunks = sr["chunks"][:3]
    clips = []
    for c in chunks:
        tr = post("/audio/transcribe", {"path": c["path"]})
        clips.append({"path": c["path"], "text": (tr.get("text") or "stub").strip() or "stub"})
    print(f"  prepared {len(clips)} clips")

    # === Case A: external target_dir ===
    custom_root = Path(tempfile.gettempdir()) / "irodori_ds_custom_test"
    if custom_root.exists():
        shutil.rmtree(custom_root)
    custom_root.mkdir()
    print(f"\n=== A) Save with target_dir={custom_root} ===")
    res = post("/datasets", {
        "name": "CustomLocTest",
        "clips": clips,
        "source_files": [LONG_REF],
        "notes": "custom-dir test",
        "target_dir": str(custom_root),
        "overwrite": True,
    })
    print("  response:", res)
    saved_at = Path(res["location"])
    print(f"  saved_at: {saved_at}")
    print(f"  exists?  {saved_at.exists()}")
    print(f"  source_audio/ contents:")
    for f in (saved_at / "source_audio").iterdir():
        print(f"    - {f.name}  size={f.stat().st_size}")
    print(f"  clips/ contents (first 5):")
    for f in sorted((saved_at / "clips").iterdir())[:5]:
        print(f"    - {f.name}  size={f.stat().st_size}")

    print("\n=== List datasets - should show CustomLocTest with its custom location ===")
    listing = get("/datasets")
    for d in listing["datasets"]:
        marker = " <-- ours" if d["name"] == "CustomLocTest" else ""
        print(f"  {d['name']:25} clips={d['num_clips']:3} location={d['location']}{marker}")

    print("\n=== Detail check (clips loadable from external location) ===")
    detail = get("/datasets/CustomLocTest")
    print(f"  num_clips: {detail['meta']['num_clips']}")
    for c in detail["clips"]:
        print(f"    [{c['index']}] dur={c['duration']}s text={c['text']!r}")

    # === Case B: default (no target_dir) ===
    print("\n=== B) Save with no target_dir (default app data) ===")
    res2 = post("/datasets", {
        "name": "DefaultLocTest",
        "clips": clips,
        "source_files": [LONG_REF],
        "notes": "default-dir test",
        "overwrite": True,
    })
    print(f"  saved_at: {res2['location']}")
    default_saved = Path(res2["location"])
    print(f"  source_audio exists? {(default_saved / 'source_audio').exists()}")

    # === Cleanup ===
    print("\n=== Cleanup ===")
    print(" delete CustomLocTest:", delete("/datasets/CustomLocTest"))
    print(" delete DefaultLocTest:", delete("/datasets/DefaultLocTest"))
    shutil.rmtree(custom_root, ignore_errors=True)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
