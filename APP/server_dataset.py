"""Dataset CRUD for the Dataset tab.

A "dataset" is a per-character collection of short audio clips paired with
transcripts, ready to be fed into ``prepare_manifest.py`` and ``train.py``.

Layout under userData::

    datasets/
      <name>/
        meta.json
        clips/
          0001.wav
          0001.txt
          0002.wav
          0002.txt
          ...
        manifest.jsonl   (created by Phase B)
        latents/         (created by Phase B)

The Dataset tab orchestrates split + transcribe (server_audio.py) before
calling ``POST /api/v1/datasets`` to finalize the staging chunks into a
named dataset directory.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from data_paths import datasets_dir  # noqa: E402

router = APIRouter()

META_NAME = "meta.json"
CLIPS_SUBDIR = "clips"
TRANSCRIPT_FILE = "transcript.txt"
LOCATIONS_FILE = "_locations.json"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-\. 一-龥ぁ-んァ-ヶー]+$")


# ---------- External-location registry ----------
# Datasets can be stored anywhere on disk via `target_dir`. We keep a small
# JSON map under datasets_dir() so they remain discoverable to the rest of
# the app (training, listing) without scanning the filesystem.

def _locations_path() -> Path:
    return datasets_dir() / LOCATIONS_FILE


def _load_locations() -> dict[str, str]:
    p = _locations_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Drop entries whose target no longer exists.
            return {k: v for k, v in data.items() if Path(v).is_dir()}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_locations(data: dict[str, str]) -> None:
    p = _locations_path()
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _register_location(name: str, abs_path: Path) -> None:
    locs = _load_locations()
    locs[name] = str(abs_path.resolve())
    _save_locations(locs)


def _unregister_location(name: str) -> None:
    locs = _load_locations()
    if name in locs:
        del locs[name]
        _save_locations(locs)


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > 64:
        raise HTTPException(400, "name must be 1-64 characters")
    if name.startswith("_"):
        raise HTTPException(400, "name must not start with '_' (reserved for staging)")
    if not NAME_PATTERN.match(name):
        raise HTTPException(400, "name contains invalid characters")
    return name


def _dataset_dir(name: str) -> Path:
    """Resolve where a dataset lives. External locations win over the default."""
    locs = _load_locations()
    if name in locs:
        return Path(locs[name])
    return datasets_dir() / name


def get_dataset_dir(name: str) -> Path:
    """Public form of _dataset_dir; honors the external-location registry."""
    return _dataset_dir(name)


def _clips_dir(name: str) -> Path:
    p = _dataset_dir(name) / CLIPS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _meta_path(name: str) -> Path:
    return _dataset_dir(name) / META_NAME


def _load_meta(name: str) -> dict:
    mp = _meta_path(name)
    if not mp.exists():
        return {"name": name, "num_clips": 0, "created_at": None, "source_files": [], "notes": ""}
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("meta.json must be an object")
        data.setdefault("name", name)
        return data
    except Exception as exc:
        raise HTTPException(500, f"corrupt meta.json for {name!r}: {exc}")


def _write_meta(name: str, meta: dict) -> None:
    _meta_path(name).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _list_clip_indices(name: str) -> list[int]:
    cdir = _clips_dir(name)
    out = []
    for f in cdir.glob("*.wav"):
        try:
            out.append(int(f.stem))
        except ValueError:
            continue
    return sorted(out)


def _load_transcript(name: str) -> dict[int, str]:
    """transcript.txt → {index: text}。旧形式（個別.txt）にも fallback。"""
    tp = _dataset_dir(name) / TRANSCRIPT_FILE
    result: dict[int, str] = {}
    if tp.exists():
        for line in tp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                try:
                    idx = int(Path(parts[0]).stem)
                    result[idx] = parts[1]
                except ValueError:
                    pass
    return result


def _save_transcript(name: str, clips: list[tuple[int, str]]) -> None:
    tp = _dataset_dir(name) / TRANSCRIPT_FILE
    lines = [f"{i:04d}.wav|{text}" for i, text in sorted(clips)]
    tp.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _measure_total_duration(name: str) -> Optional[float]:
    """クリップ全部の秒数を足す。wav を 1 本ずつ開くので速くない。"""
    import soundfile as sf

    cdir = _clips_dir(name)
    total = 0.0
    found = False
    for idx in _list_clip_indices(name):
        try:
            total += float(sf.info(str(cdir / f"{idx:04d}.wav")).duration)
            found = True
        except Exception:
            continue
    return round(total, 3) if found else None


def _total_duration_cached(name: str, meta: dict) -> Optional[float]:
    """meta.json の total_duration を返す。無ければ 1 度だけ実測して書き足す。

    一覧は開くたびに呼ばれる。ここで毎回 wav を全部開くと、クリップが
    数百あるデータセットが並んだだけでタブの切り替えが目に見えて遅くなる。
    測るのは値が無いときだけにして、以後は meta.json から読む。
    クリップ数が変わったら測り直す（追加・削除された合図）。
    """
    num = meta.get("num_clips")
    cached = meta.get("total_duration")
    if cached is not None and meta.get("duration_for_clips") == num:
        return cached

    dur = _measure_total_duration(name)
    if dur is None:
        return None
    meta["total_duration"] = dur
    meta["duration_for_clips"] = num
    try:
        _write_meta(name, meta)
    except OSError:
        pass  # 書けなくても値は返す（次回また測るだけ）
    return dur


def _list_clips_detail(name: str) -> list[dict]:
    cdir = _clips_dir(name)
    transcript = _load_transcript(name)
    items = []
    for idx in _list_clip_indices(name):
        wav = cdir / f"{idx:04d}.wav"
        # transcript.txt 優先、なければ旧形式の個別.txt にフォールバック
        if idx in transcript:
            text = transcript[idx]
        else:
            txt = cdir / f"{idx:04d}.txt"
            text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        try:
            import soundfile as sf
            info = sf.info(str(wav))
            dur = round(info.duration, 3)
        except Exception:
            dur = None
        items.append({
            "index": idx,
            "wav_path": str(wav.resolve()),
            "text": text,
            "duration": dur,
        })
    return items


# ---------- Endpoints ----------

@router.get("/api/v1/datasets")
def list_datasets() -> JSONResponse:
    root = datasets_dir()
    seen: set[str] = set()
    items = []

    # Datasets stored in the default location.
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        meta = _load_meta(child.name)
        meta["num_clips"] = len(_list_clip_indices(child.name))
        meta["total_duration"] = _total_duration_cached(child.name, meta)
        meta["location"] = str(child.resolve())
        items.append(meta)
        seen.add(child.name)

    # Externally-located datasets (via target_dir).
    for name, ext_path in _load_locations().items():
        if name in seen:
            continue
        if not Path(ext_path).is_dir():
            continue
        meta = _load_meta(name)
        meta["num_clips"] = len(_list_clip_indices(name))
        meta["total_duration"] = _total_duration_cached(name, meta)
        meta["location"] = str(Path(ext_path).resolve())
        items.append(meta)

    return JSONResponse(content={"datasets": items})


@router.get("/api/v1/datasets/{name}")
def get_dataset(name: str) -> JSONResponse:
    name = _validate_name(name)
    if not _dataset_dir(name).is_dir():
        raise HTTPException(404, f"dataset {name!r} not found")
    meta = _load_meta(name)
    clips = _list_clips_detail(name)
    meta["num_clips"] = len(clips)
    return JSONResponse(content={"meta": meta, "clips": clips})


class ClipInput(BaseModel):
    path: str = Field(..., description="Source wav path (typically a staging chunk)")
    text: str = Field("", description="Transcript text")


class CreateDatasetRequest(BaseModel):
    name: str = Field(..., description="Unique dataset name")
    clips: list[ClipInput] = Field(..., min_length=1, description="Initial clips to populate")
    source_files: list[str] = Field(default_factory=list, description="Original source audio paths (recorded in meta only, not copied)")
    notes: str = Field("", description="Free text notes")
    overwrite: bool = Field(False, description="If true, replace any existing dataset with this name")
    staging_dirs: list[str] = Field(default_factory=list, description="Staging directories to delete after save")
    target_dir: Optional[str] = Field(
        None,
        description=(
            "Optional absolute path. If provided, the dataset is created at "
            "<target_dir>/<name>/ and registered so it remains discoverable. "
            "When omitted, the dataset is stored under the app's default "
            "datasets directory."
        ),
    )


@router.post("/api/v1/datasets")
def create_dataset(req: CreateDatasetRequest) -> JSONResponse:
    name = _validate_name(req.name)

    # Determine target location
    if req.target_dir:
        target_root = Path(req.target_dir).expanduser()
        try:
            target_root = target_root.resolve(strict=True)
        except (FileNotFoundError, OSError):
            raise HTTPException(400, f"target_dir does not exist: {req.target_dir}")
        if not target_root.is_dir():
            raise HTTPException(400, f"target_dir is not a directory: {target_root}")
        ddir = target_root / name
        external = True
    else:
        ddir = datasets_dir() / name
        external = False

    if ddir.exists():
        if not req.overwrite:
            raise HTTPException(409, f"dataset {name!r} already exists at {ddir}; pass overwrite=true to replace")
        shutil.rmtree(ddir)
    ddir.mkdir(parents=True)

    if external:
        _register_location(name, ddir)

    # Save clips (wav only; text goes to transcript.txt)
    cdir = ddir / CLIPS_SUBDIR
    cdir.mkdir(parents=True, exist_ok=True)
    saved_clips: list[tuple[int, str]] = []
    try:
        for i, clip in enumerate(req.clips, start=1):
            src = Path(clip.path).expanduser()
            if not src.is_file():
                raise HTTPException(400, f"clip source not found: {clip.path}")
            shutil.copy2(src, cdir / f"{i:04d}.wav")
            saved_clips.append((i, clip.text.strip()))
    except HTTPException:
        shutil.rmtree(ddir, ignore_errors=True)
        if external:
            _unregister_location(name)
        raise
    except OSError as exc:
        shutil.rmtree(ddir, ignore_errors=True)
        if external:
            _unregister_location(name)
        raise HTTPException(500, f"failed to save clip {len(saved_clips) + 1}: {exc}") from exc

    # Write single transcript.txt (0001.wav|text per line)
    _save_transcript(name, saved_clips)

    meta = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_clips": len(saved_clips),
        "source_origins": [str(Path(p).expanduser()) for p in req.source_files],
        "notes": req.notes,
    }
    # 保存し終えた直後はクリップが手元にあるので、ここで測っておく。
    # 一覧で初めて測ると、その 1 回だけ待たされることになる。
    dur = _measure_total_duration(name)
    if dur is not None:
        meta["total_duration"] = dur
        meta["duration_for_clips"] = len(saved_clips)
    _write_meta(name, meta)

    # Clean up staging directories
    for sd in req.staging_dirs:
        try:
            sp = Path(sd)
            if sp.is_dir():
                shutil.rmtree(sp)
        except OSError:
            pass

    return JSONResponse(content={
        "status": "ok",
        "name": name,
        "num_clips": len(saved_clips),
        "location": str(ddir.resolve()),
    })


class UpdateClipRequest(BaseModel):
    text: Optional[str] = Field(None, description="New transcript; None = no change")


@router.put("/api/v1/datasets/{name}/clips/{index}")
def update_clip(name: str, index: int, req: UpdateClipRequest) -> JSONResponse:
    name = _validate_name(name)
    if not _dataset_dir(name).is_dir():
        raise HTTPException(404, f"dataset {name!r} not found")
    cdir = _clips_dir(name)
    wav = cdir / f"{index:04d}.wav"
    if not wav.is_file():
        raise HTTPException(404, f"clip {index} not found in {name!r}")
    if req.text is not None:
        transcript = _load_transcript(name)
        transcript[index] = req.text.strip()
        _save_transcript(name, list(transcript.items()))
    return JSONResponse(content={"status": "ok"})


@router.delete("/api/v1/datasets/{name}/clips/{index}")
def delete_clip(name: str, index: int) -> JSONResponse:
    name = _validate_name(name)
    if not _dataset_dir(name).is_dir():
        raise HTTPException(404, f"dataset {name!r} not found")
    cdir = _clips_dir(name)
    wav = cdir / f"{index:04d}.wav"
    txt = cdir / f"{index:04d}.txt"
    if not wav.is_file():
        raise HTTPException(404, f"clip {index} not found in {name!r}")
    wav.unlink()
    # 個別.txt が残っていれば削除（旧形式互換）
    txt = cdir / f"{index:04d}.txt"
    if txt.exists():
        txt.unlink()
    # transcript.txt からも除去
    transcript = _load_transcript(name)
    transcript.pop(index, None)
    _save_transcript(name, list(transcript.items()))
    meta = _load_meta(name)
    meta["num_clips"] = len(_list_clip_indices(name))
    _write_meta(name, meta)
    return JSONResponse(content={"status": "ok"})


@router.delete("/api/v1/datasets/{name}")
def delete_dataset(name: str) -> JSONResponse:
    name = _validate_name(name)
    ddir = _dataset_dir(name)
    if not ddir.is_dir():
        # Even if files are gone, sweep the registry entry.
        _unregister_location(name)
        raise HTTPException(404, f"dataset {name!r} not found")
    shutil.rmtree(ddir)
    _unregister_location(name)
    return JSONResponse(content={"status": "ok", "name": name})


# ---------- Auto-config (recommended training params per dataset) ----------
# Scale on how much speech the dataset holds, not on how many files it was cut
# into. Clip count alone is misleading: measured across the local datasets the
# average clip runs 2.26s to 16.34s, so two sets with the same count can differ
# 5x in material — Eva and Ravaeje are both 408 clips but 39 vs 106 minutes.
#
# 30 minutes is the anchor: it maps to 1200 steps, which is what
# tts_work/scripts/auto_train.py:steps_for() used for its middle band, the rule
# that actually produced the 84 adapters in this registry.
#
# Square root rather than linear because published guidance puts the useful
# range at roughly 15-60 minutes and calls returns beyond that diminishing;
# scaling linearly would hand a 241-minute set nearly 10k steps.
#   https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning
#
# Bounds come from what has been observed rather than theory: 2000 is the most
# steps any of the 84 runs used, and the floor keeps small sets well clear of
# the ~100 optimizer steps that LoRP-TTS (arXiv 2502.07562) found already
# yields a marked gain in speaker similarity.
_ANCHOR_MINUTES = 30.0
_ANCHOR_STEPS = 1200
_MIN_STEPS = 300
_MAX_STEPS = 2000
# 保存間隔は run の長さで決める。最良のチェックポイントは最後とは限らないので
# 見直せる粒度は要るが、1回 120MB 級の書き出しなので刻みすぎても捨てるだけになる。
#
# クリップ数から出していたときは max_steps も同じ式だったので比率が揃っていた。
# max_steps だけ総尺ベースに変えた結果、5分の素材で 500 ステップ回して
# 50 回保存する、といった噛み合わせになっていた。ステップ数側から決め直す。
#
# 桁が変われば手頃な刻みも変わる（300 ステップに 100 刻みでは粗く、
# 2000 ステップに 10 刻みでは細かすぎる）ので、レンジごとに間隔を持つ。
# 上限で頭打ちにするのではなく、間隔そのものをスケールさせる。
_SAVE_EVERY_BY_STEPS = (
    (500, 25),      # ~300-500 steps   -> 12-20 saves
    (1000, 50),     # ~500-1000        -> 10-20
    (2000, 100),    # ~1000-2000       -> 10-20
    (5000, 250),    # ~2000-5000       -> 8-20
    (None, 500),    # beyond           -> 10+
)


def _save_every_for(max_steps: int) -> int:
    """Checkpoint interval matched to the length of the run."""
    for limit, interval in _SAVE_EVERY_BY_STEPS:
        if limit is None or max_steps <= limit:
            return interval
    return _SAVE_EVERY_BY_STEPS[-1][1]


def _recommend_preset(num_clips: int) -> str:
    """Pick a LoRA preset name that the training endpoint actually accepts.

    The names must match ``server_train.PRESETS`` (speaker_style / full /
    speaker_only / style_only). They are also the values of the Train tab's
    <select>, and the renderer assigns this string to it directly — an
    unknown name silently leaves the field empty and the job is rejected
    with a 400.

    speaker_style (speaker encoder + diffusion) is the default: it captures
    both voice and delivery, and is what the Train tab starts on. Only a
    large dataset justifies the wider "full" target set, which also trains
    the text encoder.
    """
    if num_clips > 3000:
        return "full"
    return "speaker_style"


def _round_to_nice(value: int, base: int) -> int:
    """Round to the nearest multiple of base, with a minimum of 10."""
    if value <= base:
        return max(10, base)
    return max(base, int(round(value / base) * base))


@router.get("/api/v1/datasets/{name}/auto_config")
def auto_config(name: str) -> JSONResponse:
    """Suggest training parameters for a dataset.

    Returns analysis (num_clips, avg_duration, total_duration) plus a
    `recommended` block (max_steps, save_every, preset) computed from the
    research-backed 30-epoch target.
    """
    import soundfile as sf

    name = _validate_name(name)
    ddir = _dataset_dir(name)
    if not ddir.is_dir():
        raise HTTPException(404, f"dataset {name!r} not found")

    clips_dir = ddir / CLIPS_SUBDIR
    if not clips_dir.is_dir():
        raise HTTPException(400, f"dataset {name!r} has no clips/")

    durations: list[float] = []
    for wav in sorted(clips_dir.glob("*.wav")):
        try:
            durations.append(float(sf.info(str(wav)).duration))
        except Exception:
            continue

    num_clips = len(durations)
    if num_clips == 0:
        raise HTTPException(400, f"dataset {name!r} has 0 readable clips")

    total = sum(durations)
    avg = total / num_clips

    total_minutes = total / 60.0
    raw_max_steps = _ANCHOR_STEPS * math.sqrt(total_minutes / _ANCHOR_MINUTES)

    # Clip count deliberately does not scale max_steps. It looked like it
    # should — the same duration spans 25 to 229 passes over the data
    # depending on how it was cut — but no source supports a bound on that.
    # VoxCPM's own worked example (num_iters 1000, batch 16) is flat across
    # dataset sizes, which works out to anywhere from 32 to 3200 passes, and
    # every guide says the same thing instead: treat the step count as a
    # ceiling and let checkpoint evaluation pick the winner. That is what
    # checkpoint_best_val_loss already does, so the count stays duration-led.
    #
    # Friendly rounding: snap to 50/100 buckets so the UI shows clean values.
    if raw_max_steps < 1000:
        max_steps = _round_to_nice(round(raw_max_steps), 50)
    else:
        max_steps = _round_to_nice(round(raw_max_steps), 100)
    max_steps = max(_MIN_STEPS, min(_MAX_STEPS, max_steps))

    # 保存間隔は run の長さ（max_steps）のレンジで決める。
    save_every = _save_every_for(max_steps)
    # Save Every must not exceed max_steps; cap it.
    save_every = min(save_every, max(10, max_steps))

    # 進捗行の間隔も返す。既定の 50 のままだと、推奨 max_steps（100前後）では
    # 更新が2回しか出ず、10分近く止まって見える。
    # 保存より疎になると「保存はされたのに進捗が出ない」状態になるため、
    # save_every を上限にして必ず保存以上の頻度で出す。
    log_every = max(1, min(50, max_steps // 20, save_every))

    # server_train は server_dataset を import しているので、循環を避けてここで読む。
    from server_train import PRESETS

    preset = _recommend_preset(num_clips)
    if preset not in PRESETS:
        # 学習側が受け付けない名前を返すと、UI の <select> に無い値が入って
        # 空文字が送信され、ジョブ開始が 400 で落ちる。ここで気付けるようにする。
        raise HTTPException(
            status_code=500,
            detail=f"internal: recommended preset {preset!r} is not one of {sorted(PRESETS)}",
        )

    return JSONResponse(content={
        "name": name,
        "num_clips": num_clips,
        "avg_duration": round(avg, 3),
        "total_duration": round(total, 3),
        "recommended": {
            "max_steps": int(max_steps),
            "save_every": int(save_every),
            "log_every": int(log_every),
            "preset": preset,
            # 何を根拠に出した数字かを返す。クリップ数ではなく尺で決めている
            # ことが分からないと、同じ本数で違う値が出た理由が伝わらない。
            "basis": "duration",
            "total_minutes": round(total / 60.0, 1),
            "anchor": {"minutes": _ANCHOR_MINUTES, "steps": _ANCHOR_STEPS},
        },
    })
