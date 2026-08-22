"""Compare faster-whisper backend vs official transformers pipeline.

Tests same clip with both backends to identify if our backend choice is
the root cause of poor accuracy.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

REF = Path("D:/Irodori-TTS/APP/voices/voice_a3c763fd.wav")

print(f"Testing on: {REF}")
print(f"Duration: ", end="")
import soundfile as sf
info = sf.info(str(REF))
print(f"{info.duration:.1f}s, {info.samplerate}Hz, {info.channels}ch")
print()

# ============================================================
# Backend A: faster-whisper with flyfront/anime-whisper-faster
# ============================================================
print("=== A) faster-whisper / flyfront/anime-whisper-faster ===")
t0 = time.time()
from faster_whisper import WhisperModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model_fw = WhisperModel("flyfront/anime-whisper-faster", device=device, compute_type="float16")
print(f"  loaded in {time.time()-t0:.1f}s")

t0 = time.time()
segs, _ = model_fw.transcribe(str(REF), language="ja", beam_size=5)
text_fw = "".join((s.text or "").strip() for s in segs)
print(f"  transcribed in {time.time()-t0:.1f}s")
print(f"  text: {text_fw!r}")
del model_fw
torch.cuda.empty_cache()
print()

# ============================================================
# Backend B: transformers pipeline with litagin/anime-whisper (OFFICIAL)
# ============================================================
print("=== B) transformers pipeline / litagin/anime-whisper (OFFICIAL) ===")
t0 = time.time()
from transformers import pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="litagin/anime-whisper",
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
    chunk_length_s=30.0,
    batch_size=1,
)
print(f"  loaded in {time.time()-t0:.1f}s")

t0 = time.time()
result = pipe(
    str(REF),
    generate_kwargs={"language": "Japanese", "no_repeat_ngram_size": 0, "repetition_penalty": 1.0},
)
text_hf = result["text"].strip()
print(f"  transcribed in {time.time()-t0:.1f}s")
print(f"  text: {text_hf!r}")
print()

# ============================================================
# B on a short chunk via in-memory split
# ============================================================
print("=== C) Official pipeline on 3-5s slices ===")
import numpy as np
audio, sr = sf.read(str(REF), always_2d=False)
if audio.ndim == 2:
    audio = audio.mean(axis=1)
audio = np.asarray(audio, dtype=np.float32)

# Take 3 manual slices: 0-4s, 8-12s, 20-25s
for start, end in [(0, 4), (8, 12), (20, 25)]:
    clip = audio[int(start*sr):int(end*sr)]
    t0 = time.time()
    r = pipe(
        {"raw": clip, "sampling_rate": sr},
        generate_kwargs={"language": "Japanese", "no_repeat_ngram_size": 0, "repetition_penalty": 1.0},
    )
    print(f"  [{start}-{end}s ({end-start}s)] ({time.time()-t0:.1f}s) {r['text'].strip()!r}")
