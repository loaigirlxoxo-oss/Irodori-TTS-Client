"""Compare transcription quality under different conditions.

Hypotheses:
  H1: short clips (<5s) hallucinate
  H2: tight VAD padding cuts context
  H3: whisper params (beam_size, prompt, vad_filter) matter
  H4: model is just unstable for edge cases
"""
from __future__ import annotations
import json
import sys
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8080/api/v1"
REF = "D:/Irodori-TTS/APP/voices/voice_a3c763fd.wav"  # 39.5s known-good


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def transcribe(path: str, **kw) -> str:
    body = {"path": path}
    body.update(kw)
    r = post("/audio/transcribe", body)
    return (r.get("text") or "").strip()


def run(label: str, fn):
    print(f"\n=== {label} ===")
    try:
        result = fn()
        if isinstance(result, list):
            for i, t in enumerate(result, 1):
                print(f"  [{i}] {t!r}")
        else:
            print(f"  {result!r}")
    except Exception as e:
        print(f"  ERROR: {e}")


def main() -> int:
    # Reference: transcribe full file (gold standard)
    run("REFERENCE - full file, default params",
        lambda: transcribe(REF))

    # Reference: full file with anime context prompt
    run("REFERENCE - full file + anime prompt",
        lambda: transcribe(REF, initial_prompt="アニメ風の対話"))

    # H1: chunks at min_sec=1.5 (current default)
    split15 = post("/audio/split", {"path": REF, "min_sec": 1.5, "max_sec": 15})["chunks"]
    run(f"H1.a - {len(split15)} short chunks (min=1.5s)",
        lambda: [transcribe(c["path"]) for c in split15])

    # H1: chunks at min_sec=5 (longer chunks)
    split5 = post("/audio/split", {"path": REF, "min_sec": 5, "max_sec": 20})["chunks"]
    run(f"H1.b - {len(split5)} medium chunks (min=5s, max=20s)",
        lambda: [transcribe(c["path"]) for c in split5])

    # H2: more padding around speech
    split_pad500 = post("/audio/split", {"path": REF, "min_sec": 1.5, "max_sec": 15, "speech_pad_ms": 500})["chunks"]
    run(f"H2 - {len(split_pad500)} chunks with 500ms VAD padding",
        lambda: [transcribe(c["path"]) for c in split_pad500])

    # H3.a: short chunks with anime prompt
    run(f"H3.a - short chunks + anime prompt",
        lambda: [transcribe(c["path"], initial_prompt="アニメ風の対話") for c in split15])

    # H3.b: short chunks with beam_size=10
    run(f"H3.b - short chunks + beam=10",
        lambda: [transcribe(c["path"], beam_size=10) for c in split15])

    return 0


if __name__ == "__main__":
    sys.exit(main())
