"""Shared data-path resolution for the desktop app.

All writable runtime data lives under a single root determined by the
``IRODORI_DATA_DIR`` environment variable. ``main.js`` sets it to the
folder containing the launcher (dev: ``APP/``; packaged Electron:
``<install_dir>/data/``). When unset (e.g. ``python server.py`` run
directly), it falls back to ``APP/`` next to this file.

Layout::

    <data_root>/
        loras/<name>/...
        voices/
            metadata.json
            voice_*.wav
        outputs/
            sample_*.wav
        datasets/
            <name>/clips,manifest.jsonl,latents,...
        lora_jobs/
            <id>/status.json,training.log,train_output/...

A one-time migration moves legacy ``references/`` + ``metadata.json`` into
``voices/`` on first run; see :func:`migrate_legacy_voices`.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def data_root() -> Path:
    raw = os.environ.get("IRODORI_DATA_DIR", "").strip()
    base = Path(raw) if raw else Path(__file__).resolve().parent
    base.mkdir(parents=True, exist_ok=True)
    return base


def loras_dir() -> Path:
    path = data_root() / "loras"
    path.mkdir(parents=True, exist_ok=True)
    return path


def voices_dir() -> Path:
    path = data_root() / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def voices_metadata_path() -> Path:
    return voices_dir() / "metadata.json"


def outputs_dir() -> Path:
    path = data_root() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir() -> Path:
    path = data_root() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def lora_jobs_dir() -> Path:
    path = data_root() / "lora_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- One-time migrations ----------

_MIGRATION_DONE = False


def migrate_legacy_voices() -> None:
    """Copy the historical ``APP/references/`` + ``APP/metadata.json`` into
    the new ``voices/`` layout if the new location is empty.

    Idempotent. Safe to call on every server start.
    """
    global _MIGRATION_DONE
    if _MIGRATION_DONE:
        return

    app_dir = Path(__file__).resolve().parent
    old_refs = app_dir / "references"
    old_meta = app_dir / "metadata.json"
    new_voices = voices_dir()
    new_meta = voices_metadata_path()

    if old_refs.exists() and old_refs.resolve() != new_voices.resolve():
        copied = 0
        for src in old_refs.iterdir():
            if not src.is_file():
                continue
            dst = new_voices / src.name
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except OSError:
                    pass
        if copied:
            print(f"[migration] copied {copied} legacy voice files -> {new_voices}", flush=True)

    if old_meta.exists() and not new_meta.exists():
        try:
            data = json.loads(old_meta.read_text(encoding="utf-8"))
            for v in data.get("voices", []):
                if "path" in v:
                    v["path"] = str(new_voices / Path(v["path"]).name)
            new_meta.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[migration] metadata rewritten -> {new_meta}", flush=True)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[migration] metadata copy failed: {exc}", flush=True)

    _MIGRATION_DONE = True
