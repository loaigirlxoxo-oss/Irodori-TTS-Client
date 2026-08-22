"""Phase B smoke: build a tiny dataset + invoke manifest builder directly.

Avoids a full training subprocess; verifies that:
- A dataset's clips can be discovered
- DACVAE loads and encodes each clip
- manifest.jsonl + latents/ are produced
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "APP"
sys.path.insert(0, str(APP_DIR))

import data_paths
import server_train

REF_DIR = Path(__file__).resolve().parent.parent / "APP" / "references"
DATASET_NAME = "_smoke_manifest_test"


def main() -> int:
    ds_root = data_paths.datasets_dir() / DATASET_NAME
    if ds_root.exists():
        shutil.rmtree(ds_root)

    clips_dir = ds_root / "clips"
    clips_dir.mkdir(parents=True)

    # Take the first 3 reference wavs, give them dummy text
    src_wavs = sorted(REF_DIR.glob("voice_*.wav"))[:3]
    if len(src_wavs) < 3:
        print(f"need at least 3 reference wavs, found {len(src_wavs)}")
        return 1

    for i, src in enumerate(src_wavs, start=1):
        dst_wav = clips_dir / f"{i:04d}.wav"
        dst_txt = clips_dir / f"{i:04d}.txt"
        shutil.copy2(src, dst_wav)
        dst_txt.write_text(f"テストクリップ {i} です。", encoding="utf-8")
        print(f"  prepared clip {i}: {src.name}")

    meta = {
        "name": DATASET_NAME,
        "created_at": "2026-05-16T00:00:00+00:00",
        "num_clips": len(src_wavs),
        "source_files": [str(s) for s in src_wavs],
        "notes": "smoke",
    }
    (ds_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"dataset staged at {ds_root}")

    t0 = time.time()
    print("\n>> Building manifest…")
    manifest = server_train.build_manifest(DATASET_NAME, log=lambda m: print("  " + m))
    print(f"\nmanifest written: {manifest}")
    print(f"elapsed: {time.time()-t0:.1f}s")

    # Verify
    lines = manifest.read_text(encoding="utf-8").strip().split("\n")
    print(f"manifest entries: {len(lines)}")
    for line in lines:
        entry = json.loads(line)
        ok = Path(entry["latent_path"]).exists()
        print(f"  text={entry['text']!r}  frames={entry['num_frames']}  latent_exists={ok}")

    print("\nCleanup: removing test dataset")
    shutil.rmtree(ds_root)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
