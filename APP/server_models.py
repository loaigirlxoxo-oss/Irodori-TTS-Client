"""設定画面の「音声モデル」。一覧・取得・削除。

取得は fetch_models.py を別プロセスで走らせる。初回セットアップと同じ取り方
（1ファイルずつ・シンボリックリンクなし・再試行あり）になり、中止はプロセスを
止めるだけで済む。hf_hub_download はファイルの途中で止められないため。

進み具合は、取得中のファイル（*.incomplete）と、取得済みのファイルの大きさを足して出す。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import model_catalog

router = APIRouter()

_unload_runtime = None    # 読み込み済みのモデルを手放す関数
_lock = threading.Lock()
_job: dict | None = None  # {"id", "proc", "state", "error"}


def configure(*, unload_runtime) -> None:
    global _unload_runtime
    _unload_runtime = unload_runtime


def _repo_dir(repo: str) -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def _done_bytes(model_id: str) -> int:
    total = 0
    seen = set()
    for repo, filename in model_catalog.BY_ID[model_id]["files"]:
        p = model_catalog.cached_path(repo, filename)
        if p is not None and p.is_file():
            total += p.stat().st_size
        if repo not in seen:
            seen.add(repo)
            for part in _repo_dir(repo).rglob("*.incomplete"):
                try:
                    total += part.stat().st_size
                except OSError:
                    pass
    return total


def _status_of_job() -> dict | None:
    if _job is None:
        return None
    m = model_catalog.BY_ID[_job["id"]]
    return {"id": _job["id"], "state": _job["state"], "error": _job.get("error"),
            "done_gb": round(_done_bytes(_job["id"]) / 1e9, 2), "total_gb": m["size_gb"]}


@router.get("/api/v1/models/catalog")
def catalog() -> JSONResponse:
    data = model_catalog.catalog_json()
    data["job"] = _status_of_job()
    return JSONResponse(data)


def _watch(proc: subprocess.Popen) -> None:
    global _job
    out, _ = proc.communicate()
    with _lock:
        if _job is None or _job.get("proc") is not proc:
            return
        if _job["state"] == "cancelling":
            _job["state"] = "cancelled"
        elif proc.returncode == 0:
            _job["state"] = "done"
        else:
            _job["state"] = "failed"
            tail = [line for line in (out or "").splitlines() if line.strip()][-4:]
            _job["error"] = "\n".join(tail) or f"exit {proc.returncode}"


@router.post("/api/v1/models/{model_id}/download")
def download(model_id: str) -> JSONResponse:
    global _job
    if model_id not in model_catalog.BY_ID:
        return JSONResponse(status_code=404, content={"error": f"知らないモデル: {model_id}"})
    with _lock:
        if _job is not None and _job["state"] in ("running", "cancelling"):
            return JSONResponse(status_code=409, content={"error": "ほかのモデルを取得中です。"})
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.pop("HF_HUB_OFFLINE", None)  # 取りに行くので、オフライン指定は外す
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("fetch_models.py")), "--models", model_id, "--no-common"],
            cwd=str(Path(__file__).parent), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _job = {"id": model_id, "proc": proc, "state": "running", "error": None}
    threading.Thread(target=_watch, args=(proc,), daemon=True).start()
    return JSONResponse({"job": _status_of_job()})


@router.post("/api/v1/models/download/cancel")
def cancel() -> JSONResponse:
    with _lock:
        if _job is None or _job["state"] != "running":
            return JSONResponse({"job": _status_of_job()})
        _job["state"] = "cancelling"
        proc = _job["proc"]
    # 自分が起動した取得プロセスだけを止める。途中のファイルは次の取得で続きから取る。
    proc.kill()
    return JSONResponse({"job": _status_of_job()})


@router.delete("/api/v1/models/{model_id}")
def delete(model_id: str) -> JSONResponse:
    if model_id not in model_catalog.BY_ID:
        return JSONResponse(status_code=404, content={"error": f"知らないモデル: {model_id}"})
    with _lock:
        if _job is not None and _job["id"] == model_id and _job["state"] in ("running", "cancelling"):
            return JSONResponse(status_code=409, content={"error": "取得中のモデルは削除できません。"})
    # 読み込んだままだとファイルを掴んでいて消せない。消す前に手放す。
    if _unload_runtime is not None:
        _unload_runtime()
    removed = 0
    for repo, filename in model_catalog.BY_ID[model_id]["files"]:
        # 消すのは本体だけ。トークナイザなどの小さい付属ファイルは、同じリポジトリの
        # ほかのモデル（量子化版の別の種類など）も使うので残す。
        if not filename.endswith("model.safetensors"):
            continue
        p = model_catalog.cached_path(repo, filename)
        if p is not None and p.is_file():
            p.unlink()
            removed += 1
    return JSONResponse({"removed": removed, "downloaded": model_catalog.is_downloaded(model_id)})
