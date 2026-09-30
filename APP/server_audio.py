"""Audio processing endpoints: VAD splitting + Anime-whisper transcription.

Used by the Dataset tab to turn a raw audio recording into a series of
short clips with text transcripts. Heavy models (Silero VAD,
faster-whisper) are loaded lazily on first use and cached in module state
to avoid per-request startup cost.
"""
from __future__ import annotations

import difflib
import re
import sys
import threading
import unicodedata
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

# === Qwen3-ASR (optional second transcriber) ===
# Better than anime-whisper on proper nouns and takes a vocabulary hint, but
# drops sighs, laughs and fillers (measured on 452 chunks: it returned nothing
# for はあ… / うふふっ / えへへ…). Not fetched at setup; the Dataset tab offers a
# download when the user picks it.
QWEN_ASR_REPO = "Qwen/Qwen3-ASR-1.7B-hf"
QWEN_ASR_SIZE_GB = 4.1  # model.safetensors is 4,076,193,080 bytes (measured)
_qwen_model = None
_qwen_lock = threading.Lock()
_qwen_download = {"state": "idle", "error": None}
_qwen_download_lock = threading.Lock()


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
    global _whisper_model, _qwen_model

    with _whisper_users_lock:
        if _whisper_users > 0:
            return False
        _whisper_model = None
        # Qwen3-ASR goes through the same transcribe endpoint and counter.
        _qwen_model = None
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


def _qwen_supported() -> tuple[bool, str]:
    """Whether the installed transformers can run Qwen3-ASR at all."""
    try:
        from transformers import AutoModelForMultimodalLM  # noqa: F401
        from transformers import Qwen3ASRForConditionalGeneration  # noqa: F401
    except Exception:  # noqa: BLE001
        import transformers
        return False, f"transformers {transformers.__version__} は Qwen3-ASR に対応していません"
    return True, ""


# Files the processor and model read. Checked one by one: snapshot_download
# with local_files_only also demands README.md etc., and online it creates
# symlinks, which fail on Windows without developer mode (WinError 1314).
QWEN_ASR_FILES = ("config.json", "model.safetensors", "processor_config.json",
                  "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json")


def _qwen_local_snapshot() -> Optional[str]:
    """Directory holding every file Qwen3-ASR needs, or None if any is missing."""
    from huggingface_hub import hf_hub_download
    folder = None
    try:
        for name in QWEN_ASR_FILES:
            folder = str(Path(hf_hub_download(QWEN_ASR_REPO, name, local_files_only=True)).parent)
    except Exception:  # noqa: BLE001
        return None
    return folder


def _get_qwen_model():
    global _qwen_model
    if _qwen_model is None:
        with _qwen_lock:
            if _qwen_model is None:
                ok, why = _qwen_supported()
                if not ok:
                    raise RuntimeError(why)
                path = _qwen_local_snapshot()
                if path is None:
                    raise RuntimeError(f"{QWEN_ASR_REPO} がまだ取得されていません")
                from transformers import AutoModelForMultimodalLM, AutoProcessor
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if device == "cuda" else torch.float32
                processor = AutoProcessor.from_pretrained(path, local_files_only=True)
                model = AutoModelForMultimodalLM.from_pretrained(
                    path, torch_dtype=dtype, local_files_only=True).to(device)
                model.eval()
                _qwen_model = {"processor": processor, "model": model}
                print(f"[audio] qwen3-asr loaded ({device}/{dtype})", flush=True)
    return _qwen_model


def _download_qwen() -> None:
    # Same path as setup (fetch_models): one file at a time, no symlinks, with
    # its retries. The Hub dropped the connection (WinError 10054) several
    # times in testing; a finished file is not fetched again.
    from fetch_models import fetch_with_retry

    try:
        fetch_with_retry(QWEN_ASR_REPO, None)
        ok = _qwen_local_snapshot() is not None
        state = ({"state": "done", "error": None} if ok else
                 {"state": "error", "error": "取得したファイルが揃っていません。もう一度押してください"})
    except Exception as exc:  # noqa: BLE001
        print(f"[audio] qwen3-asr download failed: {exc}", flush=True)
        state = {"state": "error",
                 "error": "ネットワークから取得できませんでした。回線を確かめて、もう一度押してください"}
    with _qwen_download_lock:
        _qwen_download.update(state)


@router.get("/api/v1/audio/asr_models")
def list_asr_models() -> JSONResponse:
    """Transcribers the Dataset tab can offer, and whether each can run now."""
    ok, why = _qwen_supported()
    with _qwen_download_lock:
        dl = dict(_qwen_download)
    qwen_ready = ok and _qwen_local_snapshot() is not None
    return JSONResponse(content={"models": [
        {"id": "anime-whisper", "label": "Anime Whisper", "ready": True,
         "note": "吐息・笑い・フィラーまで拾う"},
        {"id": "qwen3-asr", "label": "Qwen3-ASR", "ready": qwen_ready, "supported": ok,
         "reason": why, "size_gb": QWEN_ASR_SIZE_GB, "download": dl,
         "note": "固有名詞に強い。語彙のヒントを渡せる。吐息・笑いは落とす"},
    ]})


@router.post("/api/v1/audio/asr_models/qwen3-asr/download")
def download_qwen_asr() -> JSONResponse:
    ok, why = _qwen_supported()
    if not ok:
        raise HTTPException(409, why)
    with _qwen_download_lock:
        if _qwen_download["state"] == "running":
            return JSONResponse(content=dict(_qwen_download))
        _qwen_download.update(state="running", error=None)
    threading.Thread(target=_download_qwen, daemon=True).start()
    return JSONResponse(content={"state": "running", "error": None})


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

SPLIT_METHODS = ("vad", "level", "none")


class SplitRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a wav/audio file")
    # vad: Silero VAD (default, as before). level: cut at quiet stretches by
    # loudness; Silero missed about half the speech of one high-pitched anime
    # voice we measured. none: the whole file is one clip (one line per file).
    method: str = Field("vad", description="vad | level | none")
    min_sec: float = Field(3.0, ge=0.0, description="Minimum chunk length in seconds (Anime-whisper hallucinates below ~3s)")
    max_sec: float = Field(20.0, gt=0.0, description="Maximum chunk length; longer segments are split")
    speech_pad_ms: int = Field(400, ge=0, description="Padding around detected speech, in ms")
    min_silence_ms: int = Field(500, ge=0, description="Min silence (ms) before treating it as a boundary; 100 is too aggressive for dense dialogue")
    level_floor_db: float = Field(45.0, gt=0.0, description="level method: below peak minus this is quiet")
    # Trimming is applied to every output clip, whatever the method.
    trim: bool = Field(False, description="Cut silence at the head and tail of each clip")
    trim_keep_ms: int = Field(100, ge=0, description="Silence left in front of / after the voice")
    trim_floor_db: float = Field(45.0, gt=0.0, description="Below peak minus this counts as silence")
    max_inner_silence_ms: int = Field(0, ge=0, description="Shorten pauses inside a clip to this; 0 keeps them")
    output_sample_rate: int = Field(48000, description="Output wav sample rate (Hz)")


def _level_segments(audio: np.ndarray, sr: int, req: SplitRequest) -> list[tuple[float, float]]:
    """Speech spans by loudness, padded and held within min/max length."""
    spans = _sound_chunks(audio, sr, floor_db=req.level_floor_db,
                          join_gap_sec=req.min_silence_ms / 1000.0,
                          min_sec=req.min_sec, max_sec=req.max_sec)
    pad = req.speech_pad_ms / 1000.0
    duration = len(audio) / sr
    out: list[tuple[float, float]] = []
    for start, end in spans:
        start, end = max(0.0, start - pad), min(duration, end + pad)
        # Padding must not reach into the neighbour.
        if out and start < out[-1][1]:
            mid = (out[-1][1] + start) / 2.0
            out[-1] = (out[-1][0], mid)
            start = mid
        out.append((start, end))
    return out


def _vad_segments(src: Path, req: SplitRequest) -> list[tuple[float, float]]:
    from silero_vad import get_speech_timestamps  # imported lazily

    # Silero needs 16kHz mono float tensor.
    audio_16k = _load_mono_audio(src, VAD_SAMPLE_RATE)
    timestamps = get_speech_timestamps(
        audio_16k,
        _get_vad_model(),
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=int(req.min_sec * 1000),
        max_speech_duration_s=req.max_sec,
        min_silence_duration_ms=int(req.min_silence_ms),
        speech_pad_ms=int(req.speech_pad_ms),
        return_seconds=True,
    )
    return [(float(ts["start"]), float(ts["end"])) for ts in timestamps]


def _trim_silence(clip: np.ndarray, sr: int, floor_db: float, keep_sec: float,
                  max_inner_sec: float) -> np.ndarray:
    """Cut silence at both ends, keeping keep_sec; optionally cap inner pauses.

    Silence is judged against the clip's own peak, so a quiet recording is
    trimmed the same way as a loud one. A clip with no sound is returned as is.
    """
    hop = int(sr * CHUNK_FRAME_SEC)
    levels = _frame_levels_db(clip, sr)
    if levels.size == 0:
        return clip
    sounding = levels > levels.max() - floor_db
    idx = np.flatnonzero(sounding)
    if idx.size == 0:
        return clip
    keep = int(keep_sec * sr)
    head = max(0, idx[0] * hop - keep)
    tail = min(len(clip), (idx[-1] + 1) * hop + keep)
    if max_inner_sec <= 0:
        return clip[head:tail]
    # Shorten each quiet run inside the voice to max_inner_sec, cutting its middle.
    limit = int(max_inner_sec / CHUNK_FRAME_SEC)
    pieces, cursor = [], head
    run_start = None
    for i in range(idx[0], idx[-1] + 1):
        if not sounding[i]:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None and i - run_start > limit:
            left = (run_start + limit // 2) * hop
            right = (i - (limit - limit // 2)) * hop
            pieces.append(clip[cursor:left])
            cursor = right
        run_start = None
    pieces.append(clip[cursor:tail])
    return np.concatenate(pieces)


@router.post("/api/v1/audio/split")
def split_audio(req: SplitRequest) -> JSONResponse:
    """Split an audio file into clips and write them to a staging directory.

    Returns a list of chunk descriptors; the caller finalizes them into a
    named dataset via POST /api/v1/datasets. "duration" is the length after
    trimming, "raw_duration" before it.

    VAD defaults are tuned for Anime-whisper compatibility: min_silence_ms=500
    avoids splitting on breath, min_sec=3 filters out fragments below the
    model's reliable transcription range.
    """
    if req.method not in SPLIT_METHODS:
        raise HTTPException(400, f"method must be one of {SPLIT_METHODS}")
    if req.max_sec <= req.min_sec and req.method != "none":
        raise HTTPException(400, "max_sec must be larger than min_sec")
    src = _resolve_wav(req.path)
    out_sr = int(req.output_sample_rate)
    audio_out = _load_mono_audio(src, out_sr).numpy()
    duration = len(audio_out) / out_sr

    if req.method == "vad":
        segments = _vad_segments(src, req)
    elif req.method == "level":
        segments = _level_segments(audio_out, out_sr, req)
    else:
        segments = [(0.0, duration)]

    if not segments:
        return JSONResponse(content={"chunks": [], "source": str(src)})

    session = uuid.uuid4().hex[:8]
    out_dir = _staging_dir() / f"{src.stem}_{session}"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for i, (start_sec, end_sec) in enumerate(segments):
        clip = audio_out[int(start_sec * out_sr):int(end_sec * out_sr)]
        raw_len = len(clip) / out_sr
        if req.trim:
            clip = _trim_silence(clip, out_sr, req.trim_floor_db, req.trim_keep_ms / 1000.0,
                                 req.max_inner_silence_ms / 1000.0)
        chunk_path = out_dir / f"chunk_{i:04d}.wav"
        sf.write(str(chunk_path), clip, out_sr)
        chunks.append({
            "index": i,
            "path": str(chunk_path),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(len(clip) / out_sr, 3),
            "raw_duration": round(raw_len, 3),
        })

    return JSONResponse(content={
        "chunks": chunks,
        "source": str(src),
        "staging_dir": str(out_dir),
    })


class ProbeRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1, max_length=5000)
    # False: lengths only (cheap; the tab calls this whenever sources change)
    recommend: bool = Field(True, description="Also measure and recommend a split method")


# One line per file (game voice sets) is recognised by length: with several
# files and a median at or under this, splitting would only cut lines apart.
PROBE_PER_FILE_MEDIAN_SEC = 15.0
# ...and only when no file is long: a single long recording among short lines
# would otherwise be kept whole (an hour-long clip is useless for training).
PROBE_PER_FILE_MAX_SEC = 30.0
# When long recordings are split next to short lines, keep lines this short.
PROBE_MIXED_MIN_SEC = 1.0
# Silero coverage of the loud parts below this means it is missing speech
# (it caught about half of one anime voice we measured), so cut by level.
PROBE_VAD_COVERAGE_MIN = 0.7
PROBE_VAD_SAMPLE_SEC = 120.0


# Measured on RTX 5080 transcribing a 10.6 s clip (GPU-wide nvidia-smi delta):
# anime-whisper 1.9 GB, Qwen3-ASR 4.3 GB, both loaded 5.9 GB.
ASR_VRAM_GB = {"anime-whisper": 1.9, "qwen3-asr": 4.3, "both": 5.9}
# Room left for the CUDA context of other apps, the desktop and the TTS model.
ASR_VRAM_MARGIN_GB = 2.0
# Free VRAM already excludes what other apps hold now, so the free-memory
# warning only adds a small buffer for allocator slack, not the 2 GB above.
ASR_FREE_HEADROOM_GB = 0.5


def _gpu_info() -> dict:
    """Card name, total VRAM, and VRAM free right now (None when unknown).

    Free memory comes from nvidia-smi: torch.cuda.mem_get_info on Windows was
    6.7 GB off from what the driver reports for the whole card.
    """
    info = {"cuda": False, "name": None, "total_gb": None, "free_gb": None}
    try:
        if not torch.cuda.is_available():
            return info
        props = torch.cuda.get_device_properties(0)
        info.update(cuda=True, name=props.name, total_gb=round(props.total_memory / 1024 ** 3, 1))
    except Exception:  # noqa: BLE001
        return info
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.splitlines()
        total_mib, used_mib = (float(x) for x in out[0].split(","))
        info["free_gb"] = round((total_mib - used_mib) / 1024, 1)
    except Exception:  # noqa: BLE001 - not an NVIDIA card, or no nvidia-smi
        pass
    return info


def _recommend_asr(gpu: dict) -> tuple[str, str]:
    """Transcriber to recommend for this machine, with the reason shown to the user."""
    if not gpu["cuda"]:
        return "anime-whisper", "GPU が見つからないので Anime Whisper を CPU で使います（時間がかかります）"
    need_both = ASR_VRAM_GB["both"] + ASR_VRAM_MARGIN_GB
    ok, _ = _qwen_supported()
    qwen_ready = ok and _qwen_local_snapshot() is not None
    if gpu["total_gb"] >= need_both and qwen_ready:
        return "both", (f"VRAM {gpu['total_gb']}GB に 2 つのモデル（約 {ASR_VRAM_GB['both']}GB）が載るので、"
                        "両方で書き起こして食い違いを見ます")
    if gpu["total_gb"] >= need_both and ok:
        return "anime-whisper", (f"VRAM {gpu['total_gb']}GB なら、Qwen3-ASR を取得すると"
                                 "「両方」で食い違いを見られます")
    return "anime-whisper", f"VRAM {gpu['total_gb']}GB なので、Anime Whisper だけで書き起こします"


def _vad_coverage(src: Path) -> Optional[float]:
    """Share of the loud audio that Silero calls speech, on the first 2 minutes."""
    from silero_vad import get_speech_timestamps  # imported lazily

    audio = _load_mono_audio(src, VAD_SAMPLE_RATE)[: int(PROBE_VAD_SAMPLE_SEC * VAD_SAMPLE_RATE)]
    loud = sum(e - s for s, e in _sound_chunks(audio.numpy(), VAD_SAMPLE_RATE, floor_db=30.0,
                                               join_gap_sec=0.2, min_sec=0.1))
    if loud <= 0:
        return None
    ts = get_speech_timestamps(audio, _get_vad_model(), sampling_rate=VAD_SAMPLE_RATE,
                               return_seconds=True)
    speech = sum(float(t["end"]) - float(t["start"]) for t in ts)
    return min(1.0, speech / loud)


@router.post("/api/v1/audio/probe")
def probe_sources(req: ProbeRequest) -> JSONResponse:
    """Lengths of the source files and the recommended split method."""
    files, bad = [], []
    for raw in req.paths:
        try:
            src = _resolve_wav(raw)
            files.append({"path": str(src), "duration": round(float(sf.info(str(src)).duration), 3)})
        except Exception:  # noqa: BLE001
            bad.append(raw)
    durations = sorted(f["duration"] for f in files)
    total = float(sum(durations))
    if not files:
        return JSONResponse(content={"files": [], "unreadable": bad, "total_duration": 0.0})

    median = durations[len(durations) // 2]
    if not req.recommend:
        return JSONResponse(content={"files": files, "unreadable": bad,
                                     "total_duration": round(total, 3), "median_duration": median})
    coverage = None
    longest_sec = durations[-1]
    per_file = len(files) >= 2 and median <= PROBE_PER_FILE_MEDIAN_SEC
    min_sec = None
    if per_file and longest_sec <= PROBE_PER_FILE_MAX_SEC:
        method, why = "none", f"{len(files)} 本の音声が短い（中央値 {median:.1f} 秒）ので、1ファイル＝1クリップ"
    else:
        if per_file:
            # Short lines mixed with long recordings: split, but keep short lines.
            min_sec = PROBE_MIXED_MIN_SEC
        # Measure the longest file; that is where splitting matters.
        longest = max(files, key=lambda f: f["duration"])
        coverage = _vad_coverage(Path(longest["path"]))
        if coverage is not None and coverage < PROBE_VAD_COVERAGE_MIN:
            method, why = "level", f"VAD が声の {coverage:.0%} しか拾えないので、音量で切る"
        else:
            shown = "測れず" if coverage is None else f"{coverage:.0%}"
            method, why = "vad", f"VAD が声を拾えている（{shown}）ので、VAD で切る"
        if min_sec is not None:
            length = f"{longest_sec:.0f} 秒" if longest_sec < 120 else f"{longest_sec / 60:.0f} 分"
            why += (f"。{length}の長い音源が混ざっているので分割し、"
                    f"短い台詞が消えないよう最短を {min_sec:g} 秒にする")
    return JSONResponse(content={
        "files": files,
        "unreadable": bad,
        "total_duration": round(total, 3),
        "median_duration": median,
        "vad_coverage": None if coverage is None else round(coverage, 3),
        "recommended": {"method": method, "reason": why, "min_sec": min_sec, **_asr_recommendation()},
    })


def _asr_recommendation() -> dict:
    gpu = _gpu_info()
    model, reason = _recommend_asr(gpu)
    warning = ""
    need = round(ASR_VRAM_GB[model] + ASR_FREE_HEADROOM_GB, 1)
    if gpu["free_gb"] is not None and gpu["free_gb"] < need:
        warning = (f"いま空いている VRAM は {gpu['free_gb']}GB で、書き起こしに約 {need}GB 要ります。"
                   "生成のモデルが載ったままなら、アプリを再起動してから始めると速く終わります")
    return {"asr_model": model, "asr_reason": reason, "asr_warning": warning, "gpu": gpu}


@router.get("/api/v1/audio/asr_recommendation")
def asr_recommendation() -> JSONResponse:
    """Recommended transcriber for this machine (GPU and VRAM based)."""
    return JSONResponse(content=_asr_recommendation())


class TranscribeRequest(BaseModel):
    """Anime-whisper / transformers pipeline params, per the official model card.

    The defaults match the documented recommendation exactly. Tune
    no_repeat_ngram_size to 5-10 and repetition_penalty above 1.0 only when
    you actually observe hallucinated repetition.
    """
    path: str = Field(..., description="Absolute path to a wav file")
    language: str = Field("Japanese", description="Full language name; 'Japanese', not 'ja'")
    no_repeat_ngram_size: int = Field(0, ge=0, le=20, description="Official baseline 0; raise to 5-10 only if repetition appears")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="Official baseline 1.0; raise above 1.0 only if repetition appears")
    model: str = Field("anime-whisper", description="anime-whisper | qwen3-asr")
    # Same values as CHUNK_JOIN_GAP_SEC / CHUNK_FLOOR_DB below (see the sweep there).
    chunk_gap_sec: float = Field(0.6, ge=0.1, le=5.0, description="Pauses at least this long split the audio before transcription")
    floor_db: float = Field(45.0, ge=10.0, le=80.0, description="Sound quieter than peak minus this is not transcribed")
    vocabulary: str = Field("", max_length=2000, description="qwen3-asr only: names and words to prefer")
    tag_nonverbal: bool = Field(False, description="Put the app's emoji tags before sighs, laughs and humming")


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


def _sound_chunks(audio: np.ndarray, sr: int, *, floor_db: float = CHUNK_FLOOR_DB,
                  join_gap_sec: float = CHUNK_JOIN_GAP_SEC, min_sec: float = CHUNK_MIN_SEC,
                  max_sec: float = CHUNK_MAX_SEC) -> list[tuple[float, float]]:
    """(start_sec, end_sec) of each sounding chunk, in order."""
    levels = _frame_levels_db(audio, sr)
    if levels.size == 0:
        return []
    peak = float(levels.max())
    if peak <= CHUNK_DIGITAL_SILENCE_DBFS:
        return []
    sounding = levels > peak - floor_db
    if peak > CHUNK_ABS_FLOOR_DBFS:
        sounding &= levels > CHUNK_ABS_FLOOR_DBFS
    runs: list[list[int]] = []
    start = None
    for i, on in enumerate(list(sounding) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if runs and start - runs[-1][1] < join_gap_sec / CHUNK_FRAME_SEC:
                runs[-1][1] = i
            else:
                runs.append([start, i])
            start = None
    max_frames = int(max_sec / CHUNK_FRAME_SEC)
    chunks = []
    for s, e in runs:
        if e - s < min_sec / CHUNK_FRAME_SEC:
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


# Emoji tags from the app's palette (irodori_tts/gradio_emoji_palette.py).
TAG_SIGH, TAG_LAUGH, TAG_HUM = "😮‍💨", "🤭", "🎵"
_TAG_BOUNDARY = r"(?:(?<=^)|(?<=[、。…!?！？」]))"
_SIGH_RE = re.compile(_TAG_BOUNDARY + r"(?<!" + TAG_SIGH + r")((?:は[ぁあ]|ふ[ぅう]|ふー|はー)[ぁぅあうー]*)(?=[…、。!?！？ー]|$)")
_LAUGH_RE = re.compile(_TAG_BOUNDARY + r"(?<!" + TAG_LAUGH + r")((?:う?ふふ+|えへ+|あはは+|くすっ|てへ)っ?)")
# Humming only when it is sung (ふーん...); a bare ふんふん can be a nod.
_HUM_RE = re.compile(r"^(?!" + TAG_HUM + r")(ふんふん(?=.*ふーん))")


def _tag_nonverbal(text: str) -> str:
    """Put the sigh / laugh / humming tags in front of those sounds."""
    text = _SIGH_RE.sub(TAG_SIGH + r"\1", text)
    text = _LAUGH_RE.sub(TAG_LAUGH + r"\1", text)
    return _HUM_RE.sub(TAG_HUM + r"\1", text)


ASR_MODELS = ("anime-whisper", "qwen3-asr")


class _RewindEscape:
    """Stop a repetition loop without cutting the line (CrisperWhisper 2.0 idea).

    When the same n-gram of tokens has just repeated LIMITS[n] times, the
    token that would start one more repeat is banned for this step, so the
    decoder moves on to the rest of the line instead of writing る until the
    448-token limit. Genuine repeats up to the limit are kept.
    Measured on 1360 real clips of 34 speakers: runaway transcripts 10 -> 0;
    on 1154 lines with the game script as reference, CER of moan-heavy lines
    28.9% -> 26.0% while plain lines did not change.
    """

    LIMITS = {1: 8, 2: 8, 3: 4, 4: 4}

    def __init__(self):
        self.prompt_len = None

    def __call__(self, input_ids, scores):
        if self.prompt_len is None:  # first step: only the forced prompt is there
            self.prompt_len = input_ids.shape[1]
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:].tolist()
            for n, limit in self.LIMITS.items():
                if len(gen) < n * limit:
                    continue
                unit = gen[-n:]
                reps, pos = 1, len(gen) - n
                while pos - n >= 0 and gen[pos - n:pos] == unit:
                    reps += 1
                    pos -= n
                if reps >= limit:
                    scores[b, unit[0]] = -float("inf")
        return scores


def _whisper_runner(req: "TranscribeRequest"):
    from transformers import LogitsProcessorList

    state = _get_whisper_model()
    processor, model = state["processor"], state["model"]

    def run(piece: np.ndarray) -> str:
        features = processor(piece, sampling_rate=16000, return_tensors="pt").input_features
        features = features.to(device=state["device"], dtype=state["dtype"])
        with torch.inference_mode():
            predicted_ids = model.generate(
                features,
                language=req.language,
                task="transcribe",
                no_repeat_ngram_size=int(req.no_repeat_ngram_size),
                repetition_penalty=float(req.repetition_penalty),
                logits_processor=LogitsProcessorList([_RewindEscape()]),
            )
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    return run


def _qwen_runner(req: "TranscribeRequest"):
    state = _get_qwen_model()
    processor, model = state["processor"], state["model"]
    vocabulary = req.vocabulary.strip()
    prompt = f"語彙: {vocabulary}" if vocabulary else None

    def run(piece: np.ndarray) -> str:
        kwargs = {"audio": piece, "language": req.language}
        if prompt:
            kwargs["prompt"] = prompt
        inputs = processor.apply_transcription_request(**kwargs).to(model.device, model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=256)
        new_ids = out[:, inputs["input_ids"].shape[1]:]
        return processor.decode(new_ids, return_format="transcription_only")[0].strip()

    return run


def _plain_letters(s: str) -> str:
    """Letters and digits only, katakana as hiragana: for comparing transcripts."""
    s = unicodedata.normalize("NFKC", s or "")
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)
    return "".join(c for c in s if unicodedata.category(c)[0] in "LN")


def _contained(part: str, whole: str) -> bool:
    """Is most of part's text (>= half its letters) found in whole, in order?"""
    p, w = _plain_letters(part), _plain_letters(whole)
    blocks = difflib.SequenceMatcher(None, p, w, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / max(1, len(p)) >= 0.5


def _segment(start: float, end: float, text: str) -> dict:
    return {"start": round(start, 3), "end": round(end, 3), "text": text}


def _transcribe_whole_then_tail(run, audio_16k: np.ndarray, sr: int, spans: list) -> list[dict]:
    """Anime Whisper: the whole clip first, then add back what it left out.

    anime-whisper stops at a long pause and drops the rest (one line per file
    in its training data). Cutting every clip at pauses (1.2.1) fixed that but
    cost context on single-line clips. Here the whole clip is read in one
    pass; then, walking the sounding chunks from the end, each chunk whose own
    transcript is not found in the whole text is added back, until one is.
    Two choices were measured on 3834 clips (22 voices with the game script,
    Koharu, 34 speakers): stopping at the first covered chunk beats checking
    every chunk (a middle chunk missing with the tail present: 29 clips;
    checking all and falling back to chunks: plain 3.24 -> 3.15% but Koharu
    3.38 -> 3.61%, and every chunk transcribed each time), and "contained"
    at half the letters beats stricter cut-offs (0.7 / 0.8 / 1.0 made every
    set worse, e.g. plain lines 3.24 -> 3.33 / 3.54 / 5.89%): spelling
    differences then look like missing speech and lines get added twice.

    Short chunks get no special case: skipping one-letter chunks only ever
    changed 8 Koharu clips (dropping one 手 but also real replies like よ) and
    no clip of the 34 other voices, so it was a rule fitted to one clip.

    Measured against the game script on 1320 real lines of 22 voices and on
    1154 twice-corrected Koharu lines:
      plain lines CER   1.2.0 3.22%   1.2.1 3.42%   this 3.24%
      Koharu CER        1.2.0 9.61%   1.2.1 4.64%   this 3.38% (exact 78.8 / 81.4 / 87.9%)
    """
    duration = len(audio_16k) / sr
    whole = run(audio_16k)
    segments = [_segment(0.0, duration, whole)]
    if len(spans) <= 1:
        return segments
    tail = []
    for start, end in reversed(spans):
        a = int(max(0.0, start - CHUNK_PAD_SEC) * sr)
        b = int(min(duration, end + CHUNK_PAD_SEC) * sr)
        text = run(audio_16k[a:b])
        if not _plain_letters(text):
            continue  # nothing was read in this chunk
        if _contained(text, whole):
            break
        tail.insert(0, _segment(start, end, text))
    return segments + tail


def _transcribe_inner(req: "TranscribeRequest", src) -> JSONResponse:
    if req.model not in ASR_MODELS:
        raise HTTPException(400, f"model must be one of {ASR_MODELS}")
    try:
        run = _whisper_runner(req) if req.model == "anime-whisper" else _qwen_runner(req)
    except RuntimeError as exc:
        # Model not fetched / transformers too old: the tab shows this as is.
        raise HTTPException(409, str(exc)) from exc

    audio_16k, sr = _load_audio_for_pipeline(src)
    duration_sec = float(len(audio_16k)) / float(sr)
    spans = _sound_chunks(audio_16k, sr, floor_db=req.floor_db, join_gap_sec=req.chunk_gap_sec)

    if req.model == "anime-whisper" and spans and duration_sec <= CHUNK_MAX_SEC:
        segments = _transcribe_whole_then_tail(run, audio_16k, sr, spans)
    else:
        # Longer than one 30 s window (or Qwen3-ASR): chunk by pauses.
        # Qwen3-ASR does not stop at pauses (it returned second sentences
        # anime-whisper dropped), and a lone word chunk loses the context it
        # needs for names, so it gets the whole sounding span when that fits.
        if req.model == "qwen3-asr" and spans and spans[-1][1] - spans[0][0] <= CHUNK_MAX_SEC:
            spans = [(spans[0][0], spans[-1][1])]
        segments = []
        for start, end in spans:
            a = int(max(0.0, start - CHUNK_PAD_SEC) * sr)
            b = int(min(duration_sec, end + CHUNK_PAD_SEC) * sr)
            segments.append(_segment(start, end, run(audio_16k[a:b])))

    text = _join_chunk_texts([s["text"] for s in segments])
    if req.tag_nonverbal:
        text = _tag_nonverbal(text)
    return JSONResponse(content={
        "text": text,
        "segments": segments,
        "model": req.model,
        "duration": round(duration_sec, 3),
        "language": "ja",
        "language_probability": 1.0,
    })
