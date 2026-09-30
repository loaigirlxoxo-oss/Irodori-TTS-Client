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

# 書き起こしが何本走っているか。かんたん学習は工程の区切りでこのモデルを
# 手放して VRAM を返すが、走っている最中に手放すと、次の書き起こしが
# 「キャッシュが空」と見て2つ目をGPUへ載せてしまう（一時的に倍載る）。
# 誰も使っていないときだけ手放せるように数える。
_whisper_users = 0
_whisper_users_lock = threading.Lock()


def whisper_busy() -> bool:
    with _whisper_users_lock:
        return _whisper_users > 0


def release_whisper_if_idle() -> bool:
    """誰も使っていなければモデルを手放す。

    確認と破棄を同じロックの中でやる。分けると、確認の直後に始まった
    書き起こしがモデルを取り出したあとでキャッシュが空になり、その次の
    書き起こしが2つ目をGPUへ載せる。
    書き起こしの側は、このロックを取って数を増やしてからモデルを取りに行く。
    """
    global _whisper_model

    with _whisper_users_lock:
        if _whisper_users > 0:
            return False
        _whisper_model = None
        return True


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
                # 取りに行かずキャッシュだけで解決する（server.py の
                # resolve_checkpoint と同じ方針）。fetch_models.py は
                # シンボリックリンクを張らない形でキャッシュへ入れるので、
                # ETag を照合できず、あっても毎回 HEAD を投げてしまう。
                # かんたん学習は工程の区切りでこのモデルを手放して読み直す
                # ため、そのたびに問い合わせると遅いうえ、実際に失敗した。
                try:
                    processor = WhisperProcessor.from_pretrained(
                        WHISPER_REPO, local_files_only=True)
                    model = WhisperForConditionalGeneration.from_pretrained(
                        WHISPER_REPO, torch_dtype=dtype, local_files_only=True
                    ).to(device)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"{WHISPER_REPO} が見つかりません。"
                        "setup.bat を再実行してモデルを取得してください。"
                    ) from exc
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
    global _whisper_users

    src = _resolve_wav(req.path)
    with _whisper_users_lock:
        _whisper_users += 1
    try:
        return _transcribe_inner(req, src)
    finally:
        with _whisper_users_lock:
            _whisper_users -= 1


# === Sound-chunked transcription ===
# Anime-whisper was trained on one game line per file, so a pause inside a
# clip reads to it as "the line is over": it stops and drops everything after
# the pause. Measured on a 1154-clip dataset: 103 clips (9%) had their second
# sentence missing, and a LoRA trained on it learned to keep talking after the
# text ends. Its encoder also only sees 30 s, so longer files lost their tail.
# We therefore cut the audio into sounding chunks by level and transcribe each
# chunk on its own. A clip without a long pause stays one chunk, as before.
CHUNK_FRAME_SEC = 0.01
CHUNK_FLOOR_DB = 45.0      # below peak; low enough to keep quiet phrases and sighs
# Hiss below this is not a chunk when the file also has louder sound. A file
# that is quiet throughout (low-gain recording) keeps the relative rule only,
# so it is still transcribed; pure digital silence gives no chunk at all.
CHUNK_ABS_FLOOR_DBFS = -70.0
CHUNK_DIGITAL_SILENCE_DBFS = -120.0
# Shorter pauses stay inside one chunk. Swept 0.35/0.5/0.6/0.8 s: 0.6 gave the
# lowest CER on untouched clips (0.025), while 0.8 already missed dropped tails
# again (CER 0.043 -> 0.092 on the clips that had lost their second sentence).
CHUNK_JOIN_GAP_SEC = 0.6
CHUNK_MIN_SEC = 0.10       # shorter blips (clicks) are not transcribed
CHUNK_MAX_SEC = 28.0       # stay inside the 30 s encoder window
CHUNK_PAD_SEC = 0.05
_SENTENCE_END = ("、", "。", "…", "!", "?", "！", "？", "♪")


def _frame_levels_db(audio_16k: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * CHUNK_FRAME_SEC)
    n = len(audio_16k) // hop
    if n == 0:
        return np.zeros(0)
    frames = audio_16k[: n * hop].reshape(n, hop)
    return 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def _split_long(start: int, end: int, levels: np.ndarray, max_frames: int) -> list[tuple[int, int]]:
    """Cut a run longer than max_frames at its quietest frame, recursively."""
    if end - start <= max_frames:
        return [(start, end)]
    lo, hi = start + max_frames // 2, start + max_frames
    cut = lo + int(np.argmin(levels[lo:hi]))
    return [(start, cut)] + _split_long(cut, end, levels, max_frames)


def _sound_chunks(audio_16k: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """(start_sec, end_sec) of each sounding chunk, in order."""
    levels = _frame_levels_db(audio_16k, sr)
    if levels.size == 0:
        return []
    peak = float(levels.max())
    if peak <= CHUNK_DIGITAL_SILENCE_DBFS:
        return []
    sounding = levels > peak - CHUNK_FLOOR_DB
    if peak > CHUNK_ABS_FLOOR_DBFS:
        sounding &= levels > CHUNK_ABS_FLOOR_DBFS
    runs: list[list[int]] = []
    start = None
    for i, on in enumerate(list(sounding) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if runs and start - runs[-1][1] < CHUNK_JOIN_GAP_SEC / CHUNK_FRAME_SEC:
                runs[-1][1] = i
            else:
                runs.append([start, i])
            start = None
    max_frames = int(CHUNK_MAX_SEC / CHUNK_FRAME_SEC)
    chunks = []
    for s, e in runs:
        if e - s < CHUNK_MIN_SEC / CHUNK_FRAME_SEC:
            continue
        for cs, ce in _split_long(s, e, levels, max_frames):
            chunks.append((cs * CHUNK_FRAME_SEC, ce * CHUNK_FRAME_SEC))
    return chunks


def _join_chunk_texts(texts: list[str]) -> str:
    """Join chunk transcripts; a pause between chunks becomes a comma."""
    out = ""
    for t in texts:
        if not t:
            continue
        if out and not out.endswith(_SENTENCE_END):
            out += "、"
        out += t
    return out


def _transcribe_inner(req: "TranscribeRequest", src) -> JSONResponse:
    state = _get_whisper_model()
    processor = state["processor"]
    model = state["model"]
    device = state["device"]
    dtype = state["dtype"]

    audio_16k, sr = _load_audio_for_pipeline(src)
    duration_sec = float(len(audio_16k)) / float(sr)

    def run(piece: np.ndarray) -> str:
        features = processor(piece, sampling_rate=sr, return_tensors="pt").input_features
        features = features.to(device=device, dtype=dtype)
        with torch.inference_mode():
            predicted_ids = model.generate(
                features,
                language=req.language,
                task="transcribe",
                no_repeat_ngram_size=int(req.no_repeat_ngram_size),
                repetition_penalty=float(req.repetition_penalty),
            )
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    segments = []
    for start, end in _sound_chunks(audio_16k, sr):
        a = int(max(0.0, start - CHUNK_PAD_SEC) * sr)
        b = int(min(duration_sec, end + CHUNK_PAD_SEC) * sr)
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": run(audio_16k[a:b])})

    return JSONResponse(content={
        "text": _join_chunk_texts([s["text"] for s in segments]),
        "segments": segments,
        "duration": round(duration_sec, 3),
        "language": "ja",
        "language_probability": 1.0,
    })
