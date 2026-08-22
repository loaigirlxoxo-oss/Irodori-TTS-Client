"""Audio processing endpoints: VAD splitting + Anime-whisper transcription.

Used by the Dataset tab to turn a raw audio recording into a series of
short clips with text transcripts. Heavy models (Silero VAD,
faster-whisper) are loaded lazily on first use and cached in module state
to avoid per-request startup cost.
"""
from __future__ import annotations

import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Allow `from irodori_tts...` even when imported under APP/ cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from data_paths import datasets_dir  # noqa: E402

router = APIRouter()

# === Whisper backend config ===
# Per the official anime-whisper model card (litagin/anime-whisper on HF),
# the recommended path is the transformers `pipeline` with these params:
#   language="Japanese"  (spelled out, NOT "ja")
#   no_repeat_ngram_size=0
#   repetition_penalty=1.0
#   chunk_length_s=30.0
#   batch_size=64
# Initial prompts are explicitly discouraged (causes hallucinations).
WHISPER_REPO = "litagin/anime-whisper"

# === Silero VAD config ===
VAD_SAMPLE_RATE = 16000  # Silero VAD requirement

# === Module-level caches (lazy) ===
_vad_model = None
_vad_lock = threading.Lock()
_whisper_model = None
_whisper_lock = threading.Lock()


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                from silero_vad import load_silero_vad
                _vad_model = load_silero_vad()
                print("[audio] silero-vad loaded", flush=True)
    return _vad_model


def _get_whisper_model():
    """Load anime-whisper as raw WhisperProcessor + model (no pipeline).

    transformers 4.57's ASR pipeline now imports ``torchcodec`` inside its
    ``preprocess`` method, and torchcodec wants FFmpeg shared libs that we
    don't have on Windows. Calling the underlying model directly sidesteps
    that entirely and is what the model card's pipeline example reduces
    to: feature extraction + ``generate`` + ``batch_decode``.
    """
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                processor = WhisperProcessor.from_pretrained(WHISPER_REPO)
                model = WhisperForConditionalGeneration.from_pretrained(
                    WHISPER_REPO, torch_dtype=dtype
                ).to(device)
                model.eval()
                _whisper_model = {
                    "processor": processor,
                    "model": model,
                    "device": device,
                    "dtype": dtype,
                }
                print(
                    f"[audio] whisper loaded ({WHISPER_REPO}, {device}/{dtype}) - direct model path",
                    flush=True,
                )
    return _whisper_model


# ---------- Helpers ----------

def _staging_dir() -> Path:
    path = datasets_dir() / "_staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_wav(path: str) -> Path:
    p = Path(path).expanduser()
    try:
        p = p.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise HTTPException(400, f"audio file not found: {path}")
    if not p.is_file():
        raise HTTPException(400, f"not a file: {p}")
    return p


def _load_mono_audio(wav_path: Path, target_sr: int) -> torch.Tensor:
    """Load wav, downmix to mono, resample. Returns (T,) float32 in [-1, 1].

    Uses soundfile to avoid the torchaudio->torchcodec FFmpeg dependency,
    which is a hassle on Windows. wav/flac coverage is enough for the
    dataset workflow; long-format support (mp3) can be added later.
    """
    audio, sr = sf.read(str(wav_path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    tensor = torch.from_numpy(audio)
    if sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, sr, target_sr)
    return tensor.contiguous().float()


def _save_wav(out_path: Path, mono: torch.Tensor, sample_rate: int) -> None:
    sf.write(str(out_path), mono.cpu().numpy(), int(sample_rate))


# ---------- Endpoints ----------

class SplitRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a wav/audio file")
    min_sec: float = Field(3.0, description="Minimum chunk length in seconds (Anime-whisper hallucinates below ~3s)")
    max_sec: float = Field(20.0, description="Maximum chunk length; longer segments are split")
    speech_pad_ms: int = Field(400, description="Padding around detected speech, in ms")
    min_silence_ms: int = Field(500, description="Min silence (ms) before treating it as a boundary; 100 is too aggressive for dense dialogue")
    output_sample_rate: int = Field(48000, description="Output wav sample rate (Hz)")


@router.post("/api/v1/audio/split")
def split_audio(req: SplitRequest) -> JSONResponse:
    """Split a long audio file into VAD-detected speech chunks.

    Returns a list of chunk descriptors. Chunks are written to a staging
    directory under the dataset root; the caller is responsible for finalizing
    them into a named dataset via POST /api/v1/datasets.

    Defaults are tuned for Anime-whisper compatibility: min_silence_ms=500
    avoids splitting on breath, min_sec=3 filters out fragments below the
    model's reliable transcription range.
    """
    src = _resolve_wav(req.path)
    vad = _get_vad_model()

    from silero_vad import get_speech_timestamps  # imported lazily

    # Silero needs 16kHz mono float tensor.
    audio_16k = _load_mono_audio(src, VAD_SAMPLE_RATE)
    timestamps = get_speech_timestamps(
        audio_16k,
        vad,
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=int(req.min_sec * 1000),
        max_speech_duration_s=req.max_sec,
        min_silence_duration_ms=int(req.min_silence_ms),
        speech_pad_ms=int(req.speech_pad_ms),
        return_seconds=True,
    )

    if not timestamps:
        return JSONResponse(content={"chunks": [], "source": str(src)})

    # Re-load at requested output sample rate for high-quality chunks.
    out_sr = int(req.output_sample_rate)
    audio_out = _load_mono_audio(src, out_sr)
    session = uuid.uuid4().hex[:8]
    out_dir = _staging_dir() / f"{src.stem}_{session}"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for i, ts in enumerate(timestamps):
        start_sec = float(ts["start"])
        end_sec = float(ts["end"])
        start_sample = int(start_sec * out_sr)
        end_sample = int(end_sec * out_sr)
        clip = audio_out[start_sample:end_sample]
        chunk_path = out_dir / f"chunk_{i:04d}.wav"
        _save_wav(chunk_path, clip, out_sr)
        chunks.append({
            "index": i,
            "path": str(chunk_path),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(end_sec - start_sec, 3),
        })

    return JSONResponse(content={
        "chunks": chunks,
        "source": str(src),
        "staging_dir": str(out_dir),
    })


class TranscribeRequest(BaseModel):
    """Anime-whisper / transformers pipeline params, per the official model card.

    The defaults match the documented recommendation exactly. Tune
    no_repeat_ngram_size to 5-10 and repetition_penalty above 1.0 only when
    you actually observe hallucinated repetition.
    """
    path: str = Field(..., description="Absolute path to a wav file")
    language: str = Field("Japanese", description="Full language name; 'Japanese', not 'ja'")
    no_repeat_ngram_size: int = Field(0, description="Official baseline 0; raise to 5-10 only if repetition appears")
    repetition_penalty: float = Field(1.0, description="Official baseline 1.0; raise above 1.0 only if repetition appears")


def _load_audio_for_pipeline(wav_path: Path) -> tuple:
    """Load wav, downmix to mono, resample to 16k. Returns (np.ndarray, 16000).

    Pre-resampling avoids transformers' internal torchaudio path which
    touches torchcodec / FFmpeg DLLs we don't have on Windows.
    """
    audio, sr = sf.read(str(wav_path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        tensor = torch.from_numpy(audio)
        tensor = torchaudio.functional.resample(tensor, sr, 16000)
        audio = tensor.numpy().astype(np.float32)
    return audio, 16000


@router.post("/api/v1/audio/transcribe")
def transcribe(req: TranscribeRequest) -> JSONResponse:
    """Transcribe one audio file with Anime-whisper (direct model path)."""
    src = _resolve_wav(req.path)
    state = _get_whisper_model()
    processor = state["processor"]
    model = state["model"]
    device = state["device"]
    dtype = state["dtype"]

    audio_16k, sr = _load_audio_for_pipeline(src)
    duration_sec = float(len(audio_16k)) / float(sr)

    features = processor(audio_16k, sampling_rate=sr, return_tensors="pt").input_features
    features = features.to(device=device, dtype=dtype)

    with torch.inference_mode():
        predicted_ids = model.generate(
            features,
            language=req.language,
            task="transcribe",
            no_repeat_ngram_size=int(req.no_repeat_ngram_size),
            repetition_penalty=float(req.repetition_penalty),
        )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    return JSONResponse(content={
        "text": text,
        "segments": [],
        "duration": round(duration_sec, 3),
        "language": "ja",
        "language_probability": 1.0,
    })
