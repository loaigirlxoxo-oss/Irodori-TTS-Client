"""Official transformers pipeline path with raw-array input (bypass torchcodec)."""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

REF = Path("D:/Irodori-TTS/APP/voices/voice_a3c763fd.wav")
print(f"REF: {REF}", flush=True)

audio, sr = sf.read(str(REF), always_2d=False)
if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = np.asarray(audio, dtype=np.float32)
print(f"audio: {audio.shape} @ {sr}Hz, dur={len(audio)/sr:.1f}s", flush=True)

print("loading pipeline (litagin/anime-whisper)...", flush=True)
t0 = time.time()
pipe = pipeline(
    "automatic-speech-recognition",
    model="litagin/anime-whisper",
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
    chunk_length_s=30.0,
    batch_size=1,
)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

# Full file
print("\n=== Full file ===", flush=True)
t0 = time.time()
r = pipe(
    {"raw": audio, "sampling_rate": sr},
    generate_kwargs={"language": "Japanese", "no_repeat_ngram_size": 0, "repetition_penalty": 1.0},
)
print(f"  ({time.time()-t0:.1f}s) {r['text']!r}", flush=True)

# Manual slices
print("\n=== Manual 4-5s slices ===", flush=True)
for start, end in [(0, 4), (8, 12), (20, 25), (30, 39)]:
    clip = audio[int(start*sr):int(end*sr)]
    t0 = time.time()
    r = pipe(
        {"raw": clip, "sampling_rate": sr},
        generate_kwargs={"language": "Japanese", "no_repeat_ngram_size": 0, "repetition_penalty": 1.0},
    )
    print(f"  [{start}-{end}s ({end-start}s)] ({time.time()-t0:.1f}s) {r['text']!r}", flush=True)
