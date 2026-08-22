"""Pre-download all model weights for offline use.

Downloads to default HF cache (~/.cache/huggingface/hub). Subsequent inference
calls will resolve from cache without network access.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

TARGETS = [
    ("Aratako/Irodori-TTS-500M-v2", "model.safetensors"),
    ("Aratako/Irodori-TTS-500M-v2-VoiceDesign", "model.safetensors"),
    ("Aratako/Irodori-TTS-500M-v3", "model.safetensors"),
    ("Aratako/Semantic-DACVAE-Japanese-32dim", "weights.pth"),
]


def main() -> int:
    failed: list[tuple[str, str, str]] = []
    for repo_id, filename in TARGETS:
        label = f"{repo_id}::{filename}"
        print(f"[download] {label}", flush=True)
        start = time.time()
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            size_mb = Path(path).stat().st_size / 1024 / 1024
            elapsed = time.time() - start
            print(f"  -> {path}  ({size_mb:.1f} MB, {elapsed:.1f}s)", flush=True)
        except Exception as exc:
            failed.append((repo_id, filename, str(exc)))
            print(f"  FAILED: {exc}", flush=True)

    if failed:
        print("\n--- FAILURES ---", flush=True)
        for repo_id, filename, msg in failed:
            print(f"  {repo_id}::{filename} -> {msg}", flush=True)
        return 1
    print("\nAll downloads complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
