"""LoRA training job management.

Wraps the project's CLI (``train.py``) as a background subprocess and
exposes start/list/status/stop endpoints. Manifest construction (DACVAE
latent precompute) runs in-process before spawning train.py so the user
only needs to press Start once.

Single-job MVP: only one job runs at a time. Concurrent starts return 409.

Job layout on disk::

    <userData>/lora_jobs/<job_id>/
        status.json     (atomically written; polled by frontend)
        training.log    (full stdout/stderr of subprocesses)
        train_output/   (train.py --output-dir; final adapter ends up here)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from data_paths import lora_jobs_dir, loras_dir  # noqa: E402
from data_paths import manual_checkpoint, OPTIONAL_MODELS, missing_optional_model_message
from huggingface_hub import hf_hub_download  # noqa: E402
from server_dataset import get_dataset_dir  # noqa: E402  honors external-location registry

router = APIRouter()

# === Configuration ===

# Maps user-facing base name -> {repo_id, config_file}.
BASE_CONFIGS: dict[str, dict[str, str]] = {
    # v4.1 は v4-Small の duration predictor だけを差し替えたもので、
    # 本体は同じ。上流に v4.1 専用の学習 config は無いので v4 のものを使う。
    "v4_1": {
        "repo_id": "Aratako/Irodori-TTS-v4.1-Small",
        "config_file": "configs/train_v4_small_lora.yaml",
    },
    # v4.1-Small の追加学習モデル。本体の作りは同じなので config も同じ。
    "v4_1_anime": {
        "repo_id": "phasefield-audio/Irodori-TTS-v4.1-Anime",
        "config_file": "configs/train_v4_small_lora.yaml",
    },
    "v4": {
        "repo_id": "Aratako/Irodori-TTS-v4-Small",
        "config_file": "configs/train_v4_small_lora.yaml",
    },
    # v4-Large（ベータ）。上流に LoRA 用 config が無いので Client で作った。
    # model: は公開チェックポイントの config_json の写し、train: は Small と同じ＋勾配チェックポイント。
    "v4_large": {
        "repo_id": "Aratako/Irodori-TTS-v4-Large",
        "config_file": "configs/train_v4_large_lora.yaml",
    },
    "v3": {
        "repo_id": "Aratako/Irodori-TTS-500M-v3",
        "config_file": "configs/train_500m_v3_lora.yaml",
    },
    "v3_voice_design": {
        "repo_id": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        "config_file": "configs/train_500m_v3_voice_design_lora.yaml",
    },
    "v2": {
        "repo_id": "Aratako/Irodori-TTS-500M-v2",
        "config_file": "configs/train_500m_v2_lora.yaml",
    },
    "voice_design": {
        "repo_id": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        "config_file": "configs/train_500m_v2_voice_design_lora.yaml",
    },
}

# Preset -> lora_target_modules string accepted by train.py.
#
# `speaker_style` is the default because the stock presets never touch
# speaker_encoder: every LoRA trained before this (95 of them) can only pin a
# voice when a reference wav is supplied. Including speaker_encoder is what
# makes --no-ref reproduce the trained speaker.
#
# It is spelled out rather than using `all_attn_mlp` because that preset also
# pulls in ~100 pretrained-backbone (ModernBERT) modules on v4. Adapting the
# backbone is only worth it when the data teaches readings the tokenizer does
# not know -- that is what `full` is for.
_SPEAKER_STYLE = (
    r"^(speaker_encoder\.in_proj"
    r"|speaker_encoder\.blocks\.\d+\.(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
    r"|blocks\.\d+\."
    r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
    r"|mlp\.(w1|w2|w3)))$"
)

PRESETS: dict[str, str] = {
    "speaker_style": _SPEAKER_STYLE,
    "full": "all_attn_mlp",
    "speaker_only": "speaker_attn_mlp",
    "style_only": "diffusion_attn_mlp",
}

# 文章の読み取り部分まで学習する学習範囲。このときは読み取り結果を使い回せない
# （結果を固定すると、その部分が何も学ばない。train.py が実行時に判定して使い回しをやめる）。
READER_TRAINING_PRESETS = ("full",)

CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"


# Regex to parse train.py's progress messages.
# Examples:
#   step=1000 loss=0.123456 rf=0.123456 lr=1.000e-04
#   step=1000 loss=0.123456 rf=0.123456 dur=0.001234 dur_mae=2.34 lr=1.000e-04
_STEP_RE = re.compile(r"^step=(\d+)\s+loss=([\d.eE+\-]+)")

# 検証の行は "valid step=... loss=..." / "valid final step=... loss=..." で、
# 学習の行と同じ形をしている。行頭で区別しないと、検証が走るたびに
# current_loss が val_loss で上書きされ、UI に学習ロスとして検証ロスが出る。
_VALID_STEP_RE = re.compile(r"^valid(?:\s+final)?\s+step=(\d+)\s+loss=([\d.eE+\-]+)")
_OOM_RE = re.compile(r"out of memory", re.IGNORECASE)
_VRAM_CAP_RE = re.compile(r"^VRAM cap: ([\d.]+) GB")
_ERROR_LINE_RE = re.compile(r"^[A-Za-z_.]*(?:Error|Exception)\b.*")


def failure_message(returncode: int, log_lines: list[str], params: dict) -> str:
    """学習が止まったときの、利用者向けの文。何が原因で、次に何をすればよいかを書く。"""
    lines = [line.strip() for line in log_lines]
    if any(_OOM_RE.search(line) for line in lines):
        cap = next((m.group(1) for line in lines if (m := _VRAM_CAP_RE.match(line))), None)
        batch = int(params["batch_size"])
        accum = int(params["gradient_accumulation_steps"])
        tips = []
        if batch > 1:
            tips.append(f"バッチサイズを {batch} から {batch // 2} に減らし、"
                        f"勾配の蓄積を {accum} から {accum * 2} に増やす")
        ref = params.get("ref_max_seconds")
        tips.append("参照音声の上限を短くする" + (f"（いま {float(ref):g}秒）" if ref else ""))
        if params.get("gradient_checkpointing") is False:
            tips.append("勾配チェックポイントを入れる")
        tips.append("ほかのアプリを閉じて、VRAM を空ける")
        head = "VRAM が足りず、学習を止めました" + (f"（使えた量: {cap}GB）" if cap else "")
        return head + "。次のどれかを試してください。\n" + "\n".join("・" + t for t in tips)
    cause = next((line for line in reversed(lines) if _ERROR_LINE_RE.match(line)), "")
    tail = f"\n原因: {cause[:300]}" if cause else "\n下の「記録」の末尾に、止まる直前の出力があります。"
    return f"学習が途中で止まりました（終了コード {returncode}）。" + tail


# Single active job lock (MVP: serial execution).
_active_lock = threading.Lock()
_active_job_id: Optional[str] = None


def active_job_id() -> Optional[str]:
    """学習中ならそのジョブ ID、動いていなければ None。

    生成の口が「いま学習中か」を知るために使う。学習は GPU を占有していて、
    そこへ生成が来るとベースモデルがもう一度載り、8GB 級では学習ごと OOM で
    落ちる。UI は学習中の生成を止めているが、OpenAI 互換の口を外部の
    クライアント（SillyTavern など）から叩かれると素通りしていた。
    """
    return _active_job_id

# Job ids the user asked to stop.
#
# The worker needs to tell "the user stopped this" from "training crashed",
# and both end with a non-zero exit code. It used to infer that from the
# status file still saying "stopping", which meant stop_job had to leave the
# job in that state until the worker noticed — and if the kill failed, the
# job stayed there forever and blocked every later start with a 409.
# Recording the request separately lets stop_job settle the state immediately
# while the worker still knows why the process died.
_stop_requests: set[str] = set()
_stop_lock = threading.Lock()


def _terminate_pid(pid: int) -> Optional[str]:
    """Kill a training process and its children. Returns an error string or None.

    A pid that is already gone counts as success — the point is that it stops
    running, not that we were the one to end it.
    """
    try:
        if sys.platform == "win32":
            proc = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True,
            )
            # 128 = そのPIDが既に無い。
            if proc.returncode not in (0, 128):
                return (proc.stderr or proc.stdout or "").strip() or f"taskkill exit {proc.returncode}"
        else:
            # 子は start_new_session=True で自分のグループを持つ。念のため
            # 自分と同じグループなら killpg せず、そのプロセスだけを狙う。
            # （古い status に残った pid など、想定外の相手を殺さないため）
            pgid = os.getpgid(pid)
            if pgid == os.getpgid(0):
                os.kill(pid, signal.SIGTERM)
            else:
                os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # 既に終了していた
    except Exception as exc:  # noqa: BLE001 - 何が起きても stopping で固まらせない
        return f"{type(exc).__name__}: {exc}"
    return None


def _find_unresolved_job() -> Optional[str]:
    """Return a job whose training process may still be running, if any.

    Only ``stop_failed`` qualifies: it is written when a kill failed or could
    not be verified, meaning a trainer might still hold the GPU. Its process
    is re-checked here so the block clears itself once that process is gone,
    rather than needing a manual status edit.
    """
    for job_dir in lora_jobs_dir().iterdir():
        if not job_dir.is_dir():
            continue
        status = _read_status(job_dir.name)
        if not status or status.get("state") != _STOP_FAILED:
            continue
        pid = status.get("pid")
        if pid and _owns_pid(int(pid), job_dir.name) is False:
            # そのプロセスはもう居ない（別物に置き換わっている）。解除してよい。
            _update_status(
                job_dir.name,
                state="failed",
                error=f"{status.get('error') or ''}; leftover process is gone".strip("; "),
                finished_at=_now_iso(),
            )
            continue
        return job_dir.name
    return None


def _process_command_line(pid: int) -> Optional[str]:
    """Return the command line of a running pid, or None if it cannot be read."""
    attempts: list[list[str]]
    if sys.platform == "win32":
        # wmic は新しい Windows では同梱されなくなったので PowerShell も試す。
        # 実行ファイルごと無い場合は FileNotFoundError になるため、
        # 一つ失敗しても次を試せるよう個別に囲む。
        attempts = [
            ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine"],
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
        ]
    else:
        attempts = [["ps", "-p", str(pid), "-o", "args="]]

    for cmd in attempts:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except Exception:  # noqa: BLE001 - このやり方が使えないだけ。次を試す
            continue
        out = (proc.stdout or "").strip()
        if out:
            return out
    return None


def _owns_pid(pid: int, job_id: str) -> Optional[bool]:
    """Whether this pid is the train.py this job started.

    The recorded pid only means something while the process it referred to is
    still alive; the OS reuses numbers, so after a restart it may belong to
    anything. train.py is spawned with ``--output-dir .../<job_id>/...``, and
    that id is unique to this job, so its presence on the command line is what
    identifies the process as ours — matching on "train.py" alone would also
    match an unrelated project's training run.

    Returns None when the command line cannot be read at all, so the caller can
    tell "not ours" from "cannot tell".
    """
    out = _process_command_line(pid)
    if out is None:
        # コマンドラインが読めない理由は二つある: プロセスがもう居ないか、
        # 問い合わせ手段が使えないか。前者なら「うちのではない」で確定できる
        # ので、生存を別途確かめてから判断する。
        if not _pid_alive(pid):
            return False
        return None
    return job_id in out and "train.py" in out


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check. True when in doubt (don't release the lock)."""
    try:
        if sys.platform == "win32":
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=15,
            )
            return str(pid) in (proc.stdout or "")
        os.kill(pid, 0)  # シグナル 0 は存在確認だけ
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 他ユーザーのプロセス。居ることは確か
    except Exception:  # noqa: BLE001 - 確かめられない。安全側に倒す
        return True
    return True


def _mark_stop_requested(job_id: str) -> None:
    with _stop_lock:
        _stop_requests.add(job_id)


def _was_stop_requested(job_id: str) -> bool:
    with _stop_lock:
        return job_id in _stop_requests


def _clear_stop_request(job_id: str) -> None:
    with _stop_lock:
        _stop_requests.discard(job_id)


# === Status persistence ===

_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _job_dir(job_id: str, *, create: bool = False) -> Path:
    # job_id は URL からそのまま渡ってくる。無検証で連結すると
    # "..\loras" のような値で lora_jobs/ の外を指せてしまう。
    # 生成側と同じ形だけ通す。
    if not _JOB_ID_RE.match(str(job_id)):
        raise HTTPException(400, f"invalid job id: {job_id!r}")
    p = lora_jobs_dir() / job_id
    # 参照しただけで作らない。形式さえ合っていれば存在しない ID でも
    # ディレクトリが生え、status.json の無い記録が一覧に並んでいた。
    # 作るのは実際に書き込む側だけ。
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _status_path(job_id: str, *, create: bool = False) -> Path:
    return _job_dir(job_id, create=create) / "status.json"


def _log_path(job_id: str, *, create: bool = False) -> Path:
    return _job_dir(job_id, create=create) / "training.log"


def _train_output_dir(job_id: str) -> Path:
    p = _job_dir(job_id, create=True) / "train_output"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_status(job_id: str) -> Optional[dict]:
    sp = _status_path(job_id)
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_status(job_id: str, data: dict) -> None:
    """Atomic-ish write: write to tmp then rename.
    Windows では replace 先が開かれていると PermissionError になるのでリトライする。
    """
    import time
    sp = _status_path(job_id, create=True)
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(sp)
            return
        except PermissionError:
            time.sleep(0.05)  # 50ms 待ってリトライ
    # リトライ全滅したら直接上書き（atomicityは諦めるが落とさない）
    sp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# status.json は read-modify-write なので、ワーカーの進捗更新と stop_job の
# 状態遷移が並行すると、先に読んだ側が後から古い state を書き戻せる。
# 停止直後にワーカーが step= 行を1つ処理しただけで "training" に戻りうる。
_status_lock = threading.Lock()


_TERMINAL_STATES = ("done", "failed", "stopped")


def _update_status(job_id: str, **fields) -> dict:
    with _status_lock:
        cur = _read_status(job_id) or {}
        # 一度終端に達したジョブを走行中に戻さない。停止の直後にワーカーが
        # step= 行を1つ処理しただけで "training" に戻ると、_ACTIVE_STATES 判定で
        # 動いていないジョブが居座り、以降の開始要求が 409 になる。
        if cur.get("state") in _TERMINAL_STATES and "state" not in fields:
            fields = {
                k: v for k, v in fields.items()
                if k not in ("current_step", "current_loss", "valid_loss")
            }
            if not fields:
                return cur
        cur.update(fields)
        cur["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_status(job_id, cur)
        return cur


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Non-terminal states. A job in one of these when the server (re)starts is an
# orphan: no worker thread survives a process restart.
_ACTIVE_STATES = ("pending", "preparing", "training", "evaluating", "stopping", "stop_failed")

# 停止を試みたが kill できず、学習プロセスが生き残っている可能性がある状態。
# 「終わった」わけではないので _ACTIVE_STATES に含める。ここを外すと、
# 生きているプロセスと新しいジョブが同時に GPU を掴む。
# 再起動で解消するので recover_orphan_jobs が failed に落とす。
_STOP_FAILED = "stop_failed"


def recover_orphan_jobs() -> None:
    """Mark any job left in a non-terminal state as failed.

    Called once at server startup. Without this, a job interrupted mid-run
    (e.g. server killed during manifest building, which has no subprocess to
    signal) stays 'stopping'/'preparing' forever, and the frontend keeps
    treating it as active — blocking all new training.

    Any subprocess recorded for such a job is killed first. train.py is spawned
    into its own process group, so it outlives the server that started it;
    releasing the job without killing it would let the next run share the GPU
    with a training process nobody is watching. This matters most for
    ``stop_failed``, where a kill is already known to have failed once.
    """
    jobs_root = lora_jobs_dir()
    recovered = 0
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        status = _read_status(job_dir.name)
        if not status or status.get("state") not in _ACTIVE_STATES:
            continue

        job_id = job_dir.name
        if status.get("state") == "evaluating":
            # 学習は終わっていて、測っている途中で止まっただけ。チェックポイントは
            # 揃っているので失敗にはしない。trainer も残っていない。
            ev = dict(status.get("evaluation") or {}, state="interrupted")
            _update_status(job_id, state="done", finished_at=_now_iso(), evaluation=ev)
            recovered += 1
            continue
        note = "interrupted by server restart"
        pid = status.get("pid")
        survivor = False  # 生きた trainer が残っている可能性があるか

        if pid:
            owned = _owns_pid(int(pid), job_id)
            if owned is True:
                kill_error = _terminate_pid(int(pid))
                if kill_error:
                    survivor = True
                    note = f"{note}; leftover pid {pid} could not be killed: {kill_error}"
                    print(f"[train] WARNING: {note}", flush=True)
                else:
                    print(f"[train] killed leftover trainer pid {pid} from {job_id}", flush=True)
            elif owned is None:
                # コマンドラインを読めなかった。生きているのか、別物なのかも
                # 分からないので殺さない。ただし「無事に終わった」とも言えない。
                survivor = True
                note = f"{note}; could not verify pid {pid}"
                print(f"[train] WARNING: {note}", flush=True)
            # owned is False -> pid は別プロセスに再利用されている。触らない。

        if survivor:
            # 生き残りを否定できないので、走行中扱いのまま残して排他を効かせる。
            # ここで failed にすると開始側のガードが通り、旧 trainer と
            # 新ジョブが GPU を奪い合う。ユーザーが手で片付ける必要がある。
            _update_status(job_id, state=_STOP_FAILED, error=note)
            print(
                f"[train] {job_id} left as {_STOP_FAILED}; "
                "kill the leftover process manually to unblock new jobs",
                flush=True,
            )
        else:
            _update_status(job_id, state="failed", error=note, finished_at=_now_iso())
        recovered += 1
    if recovered:
        print(f"[train] recovered {recovered} orphan job(s) -> failed", flush=True)


# === Manifest builder ===

def _load_transcript_map(dataset_name: str) -> dict[int, str]:
    """transcript.txt → {index: text}。旧形式（個別.txt）にも fallback。"""
    ddir = get_dataset_dir(dataset_name)
    result: dict[int, str] = {}
    tp = ddir / "transcript.txt"
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


def _list_dataset_clips(dataset_name: str) -> list[tuple[int, Path, str]]:
    """Return list of (index, wav, text) for all clips in the dataset."""
    ddir = get_dataset_dir(dataset_name) / "clips"
    if not ddir.is_dir():
        raise FileNotFoundError(f"dataset {dataset_name!r} has no clips dir")
    transcript = _load_transcript_map(dataset_name)
    out = []
    for wav in sorted(ddir.glob("*.wav")):
        try:
            idx = int(wav.stem)
        except ValueError:
            continue
        # transcript.txt 優先、なければ旧形式の個別.txt にフォールバック
        if idx in transcript:
            text = transcript[idx]
        else:
            txt = ddir / f"{idx:04d}.txt"
            text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        out.append((idx, wav, text))
    return out


class JobStopped(Exception):
    """Raised when a stop request is detected during the manifest phase."""


def build_manifest(
    dataset_name: str,
    *,
    log: callable,
    overwrite: bool = True,
    should_stop: callable = None,
) -> Path:
    """Build manifest.jsonl + latents for a dataset, encoding wavs via DACVAE.

    Returns path to manifest.jsonl. Uses CUDA when available.

    ``should_stop`` is an optional zero-arg callable checked periodically;
    when it returns True, JobStopped is raised so a stop request mid-encode
    is honored (this phase has no subprocess for the stop endpoint to kill).
    """
    import torch
    import soundfile as sf
    from irodori_tts.codec import DACVAECodec

    dataset_root = get_dataset_dir(dataset_name)
    manifest_path = dataset_root / "manifest.jsonl"
    latents_dir = dataset_root / "latents"
    # 消す前に確認する。ここを通り過ぎると、停止しても既存の latent が
    # 失われた状態で終わる（作り直しに数分〜数十分かかる）。
    if should_stop is not None and should_stop():
        raise JobStopped("stop requested before touching the dataset")
    if overwrite:
        if latents_dir.exists():
            shutil.rmtree(latents_dir)
        if manifest_path.exists():
            manifest_path.unlink()
    latents_dir.mkdir(parents=True, exist_ok=True)

    clips = _list_dataset_clips(dataset_name)
    if not clips:
        raise RuntimeError(f"dataset {dataset_name!r} contains no clips")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[manifest] loading DACVAE on {device}…")
    # DACVAE のロードは数秒〜数十秒かかる。その手前でも一度見る。
    if should_stop is not None and should_stop():
        raise JobStopped("stop requested before loading the codec")
    codec = DACVAECodec.load(repo_id=CODEC_REPO, device=device, dtype=torch.float32)

    written = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as fp:
        for i, (idx, wav, text) in enumerate(clips):
            # Honor a stop request every 10 clips.
            if should_stop is not None and i % 10 == 0 and should_stop():
                raise JobStopped("stop requested during manifest build")
            if not text:
                skipped += 1
                continue
            audio, sr = sf.read(str(wav), always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            waveform = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            with torch.inference_mode():
                latent = codec.encode_waveform(waveform.to(device), sample_rate=int(sr))
            latent_cpu = latent.squeeze(0).contiguous().cpu()  # (T_lat, D)
            latent_path = latents_dir / f"{idx:04d}.pt"
            torch.save(latent_cpu, latent_path)
            entry = {
                "text": text,
                "latent_path": str(latent_path),
                "num_frames": int(latent_cpu.shape[0]),
                "speaker_id": f"local:{dataset_name}",
            }
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            if written % 10 == 0:
                log(f"[manifest] encoded {written}/{len(clips)} clips…")

    # Free codec VRAM.
    del codec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"[manifest] done: {written} entries, {skipped} skipped (empty text)")
    if written == 0:
        raise RuntimeError("manifest has zero entries (all clips have empty text?)")
    return manifest_path


# === Training subprocess ===

# LoRA ジョブの学習率スケジュール。
#   warmup は max_steps の 5%。1000 固定（config の値）だと、推奨データ量
#   （20分前後 ≒ 100クリップ ≒ 110ステップ）では warmup の1割で学習が終わり、
#   学習率が目標の 11% までしか上がらないまま何も学ばずに終了する。
#   warmup 1000 を抜けるには約1067クリップ（3.5時間分）必要になる計算で、
#   通常の使い方はほぼ全域がこれに該当する。
#   残りは stable に充て、WSD の減衰フェーズが必ず来るようにする。
_WARMUP_RATIO = 20  # max_steps // 20 = 5%
_MIN_WARMUP_STEPS = 10


_MIN_VALID_CLIPS = 8
_MAX_VALID_FRACTION = 0.10
_RATIO_DECIMALS = 6


def _valid_ratio_for(num_clips: int, target_clips: Optional[int] = None) -> float:
    """Fraction of the dataset to hold out for validation.

    The configs ship valid_ratio=0.0005, sized for the tens of thousands of
    clips a full training run uses. On a LoRA dataset — usually 100-400 clips —
    that rounds down to a single clip, so val_loss is one sample's noise and
    the val_loss-ranked checkpoint_best_n picks essentially at random.

    train.py turns the ratio back into a count with ``int(n * ratio)``, then
    clamps it to at least 1 and at most n-1, and rejects anything outside
    [0, 1). The ratio is therefore chosen by picking the count first and
    nudging upwards until the truncation lands on it, so what actually gets
    held out matches the intent instead of drifting with the rounding.
    """
    clips = max(1, int(num_clips))
    if clips < 2:
        # train.py は 2 件未満だと分割自体を拒否する。0 を渡して検証を切る。
        return 0.0

    if target_clips is None:
        target = min(_MIN_VALID_CLIPS, max(1, int(clips * _MAX_VALID_FRACTION)))
    elif target_clips <= 0:
        return 0.0
    else:
        target = int(target_clips)
    target = max(1, min(target, clips - 1))

    ratio = round(target / clips, _RATIO_DECIMALS)
    # 丸めで下振れした分を最小刻みで持ち上げる（int() の切り捨て対策）。
    step = 10 ** -_RATIO_DECIMALS
    while ratio < 1.0 and int(clips * ratio) < target:
        ratio = round(ratio + step, _RATIO_DECIMALS)
    return min(ratio, 1.0 - step)


def _lr_schedule_for(max_steps: int, warmup_pct: float = 100.0 / _WARMUP_RATIO,
                     decay_pct: float = 25.0) -> tuple[int, int]:
    """Return (warmup_steps, stable_steps) scaled to this run's length.

    stable_steps is a duration measured after warmup, not a boundary:
    optim.py holds the peak rate while ``step < warmup_steps + stable_steps``
    and decays over whatever remains. Splitting the run 5/70/25 keeps a real
    decay phase no matter how short the job is.
    """
    steps = max(1, int(max_steps))

    # warmup は 5% を基本にしつつ最低 _MIN_WARMUP_STEPS。ただしその下限を
    # そのまま使うと、max_steps が下限付近（受け口は 10 以上を許す）のときに
    # warmup が全体を占め、安定も減衰も無いまま終わる。
    # 全体の半分を超えないよう頭を押さえる。
    warmup = max(_MIN_WARMUP_STEPS, int(steps * warmup_pct / 100.0))
    warmup = min(warmup, max(1, steps // 2))

    remaining = steps - warmup
    if remaining <= 0:
        return warmup, 0
    decay = max(1, int(remaining * decay_pct / 100.0))  # 末尾（既定25%）は必ず減衰に充てる
    stable = remaining - decay
    return warmup, stable


def _build_train_command(
    *,
    config_file: Path,
    manifest_path: Path,
    output_dir: Path,
    init_checkpoint: Path,
    lora_target_modules: str,
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    save_every: int,
    log_every: int,
    num_clips: int,
    learning_rate: Optional[float] = None,
    text_encoder_lr: Optional[float] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    weight_decay: Optional[float] = None,
    min_lr_scale: Optional[float] = None,
    warmup_pct: Optional[float] = None,
    decay_pct: Optional[float] = None,
    valid_clips: Optional[int] = None,
    seed: Optional[int] = None,
    gradient_checkpointing: Optional[bool] = None,
    compile_blocks: Optional[bool] = None,
    ref_max_seconds: Optional[float] = None,
) -> list[str]:
    """Build the train.py command line for a LoRA job.

    The learning-rate schedule has to be derived from max_steps rather than
    left to the config file. Every LoRA config ships a full-training schedule
    (warmup_steps=1000, stable_steps=24000-44000, for runs of 30k-50k steps),
    but a LoRA job is typically 100-2000 steps. Left alone, training ends
    partway through warmup: at 110 steps the rate only ever reaches 11% of the
    target and the adapter learns essentially nothing.
    """
    sched = {k: v for k, v in (("warmup_pct", warmup_pct), ("decay_pct", decay_pct)) if v is not None}
    warmup_steps, stable_steps = _lr_schedule_for(max_steps, **sched)
    # 既定の 50 のままだと、短いジョブ（推奨は100前後）では進捗行が数回しか出ず、
    # 数分から十数分のあいだ止まったように見える。少なくとも20回は出す。
    log_every = max(1, min(int(log_every), max(1, int(max_steps) // 20)))
    cmd = [
        sys.executable,
        "train.py",
        "--config", str(config_file),
        "--manifest", str(manifest_path),
        "--output-dir", str(output_dir),
        "--init-checkpoint", str(init_checkpoint),
        "--max-steps", str(int(max_steps)),
        "--warmup-steps", str(int(warmup_steps)),
        "--stable-steps", str(int(stable_steps)),
        # config の valid_every は 1000（フル学習向け）で、短い LoRA ジョブでは
        # 検証が一度も走らないまま終わる。val_loss を各世代の目安として
        # 一覧に出したいので、save 頻度に合わせて必ず走らせる。
        "--valid-every", str(int(save_every)),
        "--valid-ratio", f"{_valid_ratio_for(num_clips, valid_clips):.6f}",
        # 検証の本を学習から外さない。検証の損失は画面に出すだけで、ステップ選びには使わない。
        "--valid-from-train",
        "--batch-size", str(int(batch_size)),
        "--gradient-accumulation-steps", str(int(gradient_accumulation_steps)),
        "--save-every", str(int(save_every)),
        "--log-every", str(int(log_every)),
        # 保存した世代を1つも捨てない。config の既定（5）だと保持機構が働き、
        # 検証ありの場合は定期チェックポイントが最新1個に上書きされ続けて、
        # 聴き比べる材料が残らない。0 にすると train.py は保持判定ごと止める。
        "--checkpoint-best-n", "0",
        "--lora",
        "--lora-target-modules", lora_target_modules,
        "--no-progress",  # avoid tqdm spam; rely on `step=NNN loss=...` lines
    ]
    # 渡されたものだけ足す。渡さなければ config の値のまま（これまでどおり）。
    optional = (
        ("--lr", learning_rate), ("--pretrained-text-encoder-learning-rate", text_encoder_lr),
        ("--lora-r", lora_r), ("--lora-alpha", lora_alpha), ("--lora-dropout", lora_dropout),
        ("--weight-decay", weight_decay), ("--min-lr-scale", min_lr_scale), ("--seed", seed),
        ("--ref-max-seconds", ref_max_seconds),
    )
    for flag, value in optional:
        if value is not None:
            cmd += [flag, str(value)]
    if gradient_checkpointing is not None:
        cmd.append("--gradient-checkpointing" if gradient_checkpointing else "--no-gradient-checkpointing")
    if compile_blocks is not None:
        cmd.append("--compile-blocks" if compile_blocks else "--no-compile-blocks")
    return cmd


def _free_inference_models() -> None:
    """Best-effort GPU memory release before starting training subprocess."""
    try:
        import server_audio
        server_audio._whisper_model = None
        server_audio._vad_model = None
    except Exception:
        pass
    try:
        from irodori_tts.inference_runtime import clear_cached_runtime
        clear_cached_runtime()
    except Exception:
        pass
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _latest_lora_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint directory or .pt file (LoRA case = dir)."""
    if not output_dir.exists():
        return None
    candidates = list(output_dir.glob("checkpoint_*"))
    if not candidates:
        return None
    # Prefer adapter-only checkpoint dirs (LoRA writes dirs not .pt).
    dirs = [c for c in candidates if c.is_dir()]
    if dirs:
        return max(dirs, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _drop_intermediate_optimizer_state(output_dir: Path) -> int:
    """Delete trainer_state.pt from every checkpoint except the final one.

    全世代を残すようにしたので、そのままだと 1 ジョブで十数 GB になる。
    容量の大半は trainer_state.pt（実測 191MB、adapter 本体は 95MB）で、
    これは学習を再開するときにしか読まれない。再開するなら最後まで進んだ
    地点からなので、checkpoint_final のぶんだけ残せば足りる。
    聴き比べに使う adapter は触らない。
    """
    freed = 0
    try:
        for state in output_dir.glob("checkpoint*/trainer_state.pt"):
            if state.parent.name == "checkpoint_final":
                continue
            size = state.stat().st_size
            state.unlink()
            freed += size
    except OSError as exc:
        # 掃除に失敗しても学習の成果は無事なので、ジョブは失敗にしない。
        print(f"[job] could not clean optimizer state in {output_dir}: {exc}", flush=True)
    return freed


def _safe_lora_dir(lora_name: str) -> Path:
    """Registry path for `lora_name`, refusing anything that escapes it."""
    name = (lora_name or "").strip()
    if not name:
        raise HTTPException(400, "lora_name must not be empty")
    # パス区切りや親参照は名前として認めない。解決後の位置も必ず確かめる
    # （Windows では "a:b" のようなドライブ表記でも外へ出られる）。
    if "/" in name or "\\" in name or name in (".", "..") or ":" in name:
        raise HTTPException(400, f"lora_name must not contain path separators: {name!r}")
    root = loras_dir().resolve()
    dst = (root / name).resolve()
    if dst == root or not dst.is_relative_to(root):
        raise HTTPException(400, f"lora_name resolves outside the registry: {name!r}")
    return dst


def _register_lora(lora_name: str, base: str, source_dir: Path, notes: str) -> None:
    """Copy a freshly-trained adapter into the LoRA registry."""
    from irodori_tts.lora import is_lora_adapter_dir
    if not is_lora_adapter_dir(source_dir):
        raise RuntimeError(f"trained checkpoint is not a valid PEFT adapter dir: {source_dir}")
    # 登録名はそのままディレクトリ名になり、既存なら rmtree する。"../.." を
    # 含む名前を通すと登録先の外を消せてしまうため、ここで必ず確かめる。
    # 学習開始・チェックポイント登録のどちらもこの関数を通るので、
    # 入口を個別に守るのではなく削除の直前で一度だけ検証する。
    dst_root = _safe_lora_dir(lora_name)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    adapter_dst = dst_root / "adapter"
    # trainer_state.pt はオプティマイザ状態で、学習を再開するときにしか使わない。
    # アダプタ本体の倍以上（実測 239MB 対 120MB）あるので登録先には持ち込まない。
    # 元のジョブフォルダには残るので、再開したいときはそちらを使える。
    shutil.copytree(source_dir, adapter_dst, ignore=shutil.ignore_patterns("trainer_state.pt"))
    meta = {
        "name": lora_name,
        "base": base,
        "imported_at": _now_iso(),
        "source": str(source_dir),
        "notes": notes or "trained in-app",
    }
    (dst_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# === Job worker ===

def _run_job(job_id: str, params: dict) -> None:
    """Background worker. Runs in its own thread."""
    global _active_job_id
    log_path = _log_path(job_id, create=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = msg.rstrip("\n")
        log_lines.append(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        # Keep status log_tail bounded.
        tail = log_lines[-60:]
        _update_status(job_id, log_tail=tail)
        print(f"[job {job_id}] {line}", flush=True)

    try:
        # スレッドが走り出す前に停止された場合、ここで打ち切る。これが無いと
        # 状態を preparing に戻したうえで、既存 latents の削除・DACVAE の
        # ロード・manifest 構築まで進んでしまう（最初の停止チェックは
        # build_manifest のクリップ単位ループの中にしかない）。
        if _was_stop_requested(job_id):
            raise JobStopped("stop requested before the job started")

        _update_status(
            job_id,
            state="preparing",
            started_at=_now_iso(),
            current_step=0,
            current_loss=None,
            valid_loss=None,
        )
        _free_inference_models()
        log("[job] freed inference models from GPU")

        # 1) Build manifest. should_stop lets a stop request abort the
        # in-process encode loop (no subprocess exists in this phase).
        dataset = params["dataset"]
        log(f"[job] building manifest for dataset: {dataset}")

        def _stop_requested() -> bool:
            return _was_stop_requested(job_id)

        manifest = build_manifest(dataset, log=log, should_stop=_stop_requested)

        def _abort_if_stopped(where: str) -> None:
            """Bail out of the preparing phase when a stop has been requested.

            build_manifest polls for stops, but the steps after it (checkpoint
            download, path checks) do not. A stop landing in that window used
            to be ignored until the child had already been spawned, so a
            multi-gigabyte download kept running after the user pressed stop.
            """
            if _was_stop_requested(job_id):
                raise JobStopped(f"stop requested {where}")

        _abort_if_stopped("after manifest build")

        # 2) Resolve base checkpoint via HF download
        #
        # hf_hub_download は中断できない。停止を押してもここは最後まで走る。
        # 初回だけ数GB かかるが、以降はキャッシュから即座に返るので、
        # 中断可能にするために別スレッド化するほどの利得はないと判断した。
        # setup.bat がモデルを先に取得しておくのもこのため。
        base = params["base"]
        base_cfg = BASE_CONFIGS[base]
        repo_id = base_cfg["repo_id"]
        log(f"[job] resolving base checkpoint: {repo_id}")
        # 任意モデル（setup.bat が取らないもの）は手で置いた実体だけを見る。
        # 無いときに落としに行くと、生成側は「置いてください」と言うのに学習
        # だけが 2.9GB を勝手に取り始める。しかも初回ダウンロードは停止要求を
        # 受け付けないので、始まると止められない。
        manual = manual_checkpoint(repo_id)
        if manual is not None:
            init_ckpt = Path(manual)
            log("[job] using a manually placed checkpoint")
        elif repo_id in OPTIONAL_MODELS:
            raise RuntimeError(missing_optional_model_message(repo_id))
        else:
            # 警告は実際に取りに行くときだけ出す。出さないと、落とさない経路でも
            # 「止められない」と読めてしまう。
            log("[job] (a first-time download cannot be interrupted; stop takes effect after it)")
            init_ckpt = Path(hf_hub_download(
                repo_id=repo_id, filename="model.safetensors"
            ))
        _abort_if_stopped("after checkpoint download")

        # 上流 main へ統合したので、v4 も v2/v3 と同じツリー・同じ Python で
        # 学習できる。ベースごとに config を選ぶだけでよい。
        train_root = _PROJECT_ROOT
        config_file = train_root / base_cfg["config_file"]
        if not config_file.is_file():
            raise FileNotFoundError(f"training config not found: {config_file}")

        # 3) Build + spawn train.py
        train_output = _train_output_dir(job_id)
        cmd = _build_train_command(
            config_file=config_file,
            manifest_path=manifest,
            output_dir=train_output,
            init_checkpoint=init_ckpt,
            lora_target_modules=PRESETS[params["preset"]],
            max_steps=params["max_steps"],
            batch_size=params["batch_size"],
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            save_every=params["save_every"],
            log_every=params["log_every"],
            num_clips=sum(1 for _ in manifest.open(encoding="utf-8")),
            learning_rate=params.get("learning_rate"),
            **{k: params.get(k) for k in TRAIN_SETTING_KEYS},
        )
        # spawn の直前に最後の確認。ここを通ってから起動までは一瞬なので、
        # そこで来た停止要求は pid 記録後のチェックが拾う。
        _abort_if_stopped("before spawning train.py")

        log(f"[job] spawning: {' '.join(str(c) for c in cmd)}")
        _update_status(job_id, state="training")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Dataset names and manifest paths can be Japanese; the child would
        # otherwise inherit the cp932 console codepage and fail on them.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd,
            cwd=str(train_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            # POSIX でも自分のプロセスグループを持たせる。これが無いと
            # 子はサーバーと同じグループに属し、停止時の killpg が
            # サーバー自身まで巻き込む。
            start_new_session=(sys.platform != "win32"),
        )
        _update_status(job_id, pid=proc.pid)

        # spawn と pid の記録のあいだに停止要求が入ると、stop_job は
        # まだ pid=None の status を読んで「殺す相手が居ない」と判断し、
        # 何もせず成功を返す。manifest 側の停止チェックも既に通過した後なので、
        # 学習だけが max_steps まで走り続けてしまう。ここで拾い直す。
        if _was_stop_requested(job_id):
            log("[job] stop was requested while spawning; terminating child")
            _terminate_pid(proc.pid)

        for line in proc.stdout:  # blocks until child closes stream
            log(line.rstrip("\n"))
            stripped = line.strip()
            m = _STEP_RE.match(stripped)
            if m:
                _update_status(
                    job_id,
                    current_step=int(m.group(1)),
                    current_loss=float(m.group(2)),
                )
                continue
            v = _VALID_STEP_RE.match(stripped)
            if v:
                # 検証ロスは別に持つ。学習ロスと混ぜると、検証が走った瞬間だけ
                # UI の数値が跳ねて「悪化した」ように見える。
                _update_status(job_id, valid_loss=float(v.group(2)))
        proc.wait()
        # 読み取り結果をディスクに置いた学習の一時ファイル。train.py は終わるときに自分で消すが、
        # 停止ボタンで止めたときは消す間が無いので、ここでも消す。
        shutil.rmtree(train_output / "_condition_cache", ignore_errors=True)

        # Detect stopped (terminated externally) vs error vs success.
        # A killed process also exits non-zero, so ask whether a stop was
        # requested rather than reading it out of the status file — stop_job
        # settles the state itself and no longer leaves a "stopping" marker.
        if _was_stop_requested(job_id):
            _update_status(job_id, state="stopped", finished_at=_now_iso())
            log(f"[job] stopped by user (exit {proc.returncode})")
            return

        if proc.returncode != 0:
            _update_status(
                job_id,
                state="failed",
                finished_at=_now_iso(),
                error=failure_message(proc.returncode, log_lines, params),
            )
            log(f"[job] FAILED (exit {proc.returncode})")
            return

        # 4) Make sure the run left something to audition.
        #
        # 登録はここではしない。最終ステップが最良とは限らず、どの世代を採るかは
        # 聴き比べて決めるものなので、採用は試聴（/register）だけの仕事にする。
        # ここで勝手に登録すると、聴かないまま最終回がレジストリに載る。
        ckpt = _latest_lora_checkpoint(train_output)
        if ckpt is None or not ckpt.is_dir():
            _update_status(
                job_id,
                state="failed",
                finished_at=_now_iso(),
                error="no checkpoint dir found after training",
            )
            log("[job] FAILED: no checkpoint output")
            return

        freed = _drop_intermediate_optimizer_state(train_output)
        n_ckpt = len(_list_job_checkpoints(job_id))
        log(
            f"[job] finished; {n_ckpt} checkpoint(s) ready to audition"
            + (f" (freed {freed / (1 << 30):.1f} GB of optimizer state)" if freed else "")
        )
        # かんたん学習は自分で測って選ぶので、ここでは測らない（二重になる）。
        if params.get("evaluate", True) and not params.get("easy_token"):
            _update_status(job_id, state="evaluating")
            log("[job] measuring every checkpoint (SIM / CER on the evaluation lines)")
            try:
                _evaluate_job(job_id, params)
            except Exception as exc:  # noqa: BLE001 - 測れなくても学習の結果は使える
                _update_status(job_id, evaluation={"state": "failed",
                                                   "error": f"{type(exc).__name__}: {exc}"})
                log(f"[job] evaluation failed: {type(exc).__name__}: {exc}")
        _update_status(
            job_id,
            state="done",
            finished_at=_now_iso(),
        )
    except JobStopped as exc:
        _update_status(job_id, state="stopped", finished_at=_now_iso())
        log(f"[job] stopped during manifest build: {exc}")
    except Exception as exc:
        _update_status(
            job_id,
            state="failed",
            finished_at=_now_iso(),
            error=f"{type(exc).__name__}: {exc}",
        )
        log(f"[job] EXCEPTION: {type(exc).__name__}: {exc}")
    finally:
        _clear_stop_request(job_id)
        with _active_lock:
            # 自分がロックを持っているときだけ解放する。無条件に None を入れると、
            # stop の kill が失敗して先にロックを手放し、その後に始まった別ジョブの
            # 排他まで巻き添えで解除してしまう（学習が同時に走る）。
            if _active_job_id == job_id:
                _active_job_id = None


def _evaluate_job(job_id: str, params: dict) -> None:
    """Measure each checkpoint and mark the one to recommend.

    Same tools and rule as the easy tab (SIM against a reference, CER of the
    evaluation lines, pick_by_score over every checkpoint). The reference is
    the dataset's first clip, as the easy tab does for a folder.
    """
    import server_easy
    from easy_eval import combined_score, pick_by_score

    clips = sorted(get_dataset_dir(params["dataset"]).joinpath("clips").glob("*.wav"))
    if not clips:
        raise RuntimeError("データセットに音声がありません")
    ref = str(clips[0])

    def save(st: dict) -> None:
        _update_status(job_id, evaluation={
            "state": "running", "rows": st.get("evaluated", []),
            "current": st.get("evaluating"), "total": st.get("evaluate_total"),
        })

    rows = server_easy._evaluate_checkpoints(
        # 一時登録の名前に入る ID。16進のまま渡す（server_lora.is_preview_lora が16進で
        # 見分ける。前に "tr" を付けたら一時登録と見なされず、測っただけで採用済みになった）。
        job_id, ref, params["base"], int(params["max_steps"]), {}, job_id,
        save=save, should_stop=lambda: _was_stop_requested(job_id))
    stopped = _was_stop_requested(job_id)
    result = {"state": "stopped" if stopped else "done", "rows": rows, "ref": ref}
    if rows:
        for r in rows:
            r["score"] = combined_score(r)
        picked = pick_by_score(rows)
        result.update(picked=picked["name"])
    _update_status(job_id, evaluation=result)


# === Endpoints ===

class StartJobRequest(BaseModel):
    lora_name: str = Field(..., description="Output LoRA name (registered after training)")
    dataset: str = Field(..., description="Existing dataset name")
    base: str = Field("v4", description="One of v4, v3, v2, voice_design")
    preset: str = Field(
        "speaker_style",
        description="One of speaker_style, full, speaker_only, style_only",
    )
    max_steps: int = Field(2000, ge=10, le=200000)
    batch_size: int = Field(4, ge=1, le=128)
    gradient_accumulation_steps: int = Field(8, ge=1, le=64)
    save_every: int = Field(1000, ge=10)
    log_every: int = Field(50, ge=1)
    learning_rate: Optional[float] = Field(None, gt=0)
    # 学習のあと各チェックポイントを測っておすすめに印を付ける（かんたん学習と同じ道具）
    evaluate: bool = Field(True, description="Measure every checkpoint after training")
    # 以下は空なら config の値。画面は /lora/train_defaults の値を最初から入れて見せる。
    text_encoder_lr: Optional[float] = Field(None, gt=0, description="pretrained text encoder LR")
    lora_r: Optional[int] = Field(None, ge=1, le=256)
    lora_alpha: Optional[int] = Field(None, ge=1, le=1024)
    lora_dropout: Optional[float] = Field(None, ge=0.0, lt=1.0)
    weight_decay: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_lr_scale: Optional[float] = Field(None, ge=0.0, le=1.0)
    warmup_pct: Optional[float] = Field(None, ge=0.0, le=50.0)
    decay_pct: Optional[float] = Field(None, ge=1.0, le=100.0)
    valid_clips: Optional[int] = Field(None, ge=0, le=1000, description="0 = no validation")
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)
    gradient_checkpointing: Optional[bool] = None
    # ブロック単位の torch.compile（train._compile_blocks）。初回のステップでコンパイルを待つ
    compile_blocks: Optional[bool] = None
    # 1本ごとに付ける参照音声（同じ話者の別の音声をつないだもの）の最長秒数。
    # 8本ずつ compile するときは 30 秒に下げないと 16GB に収まらない（実測）
    ref_max_seconds: Optional[float] = Field(None, ge=1.0, le=120.0)
    overwrite: bool = Field(
        False,
        description="If true, delete an existing LoRA with the same name before starting",
    )
    # かんたん学習が自分の学習を始めるための合言葉。外から使う口では
    # 渡らないので、かんたん学習中の学習開始を断れる。
    easy_token: str | None = None

    @model_validator(mode="after")
    def _clamp_intervals(self) -> "StartJobRequest":
        """Keep the save/log intervals inside the run.

        Their defaults (1000 / 50) assume a long run. If max_steps is smaller —
        the Train tab's auto config suggests around 100, and the API can be
        called with anything — an interval larger than max_steps means the
        event never happens: no intermediate checkpoint, no validation (which
        is derived from save_every), no progress line. Clamping here covers
        every caller instead of just the UI path.
        """
        if self.save_every > self.max_steps:
            self.save_every = self.max_steps
        if self.log_every > self.max_steps:
            self.log_every = self.max_steps
        # 保存より進捗が疎だと、書けているのに止まって見える。UI の自動設定は
        # この関係を守るが、API を直に叩けば log_every だけ大きく渡せてしまう。
        if self.log_every > self.save_every:
            self.log_every = self.save_every
        return self


TRAIN_SETTING_KEYS = ("text_encoder_lr", "lora_r", "lora_alpha", "lora_dropout", "weight_decay",
                      "min_lr_scale", "warmup_pct", "decay_pct", "valid_clips", "seed",
                      "gradient_checkpointing", "compile_blocks", "ref_max_seconds")


def compile_available() -> bool:
    """ブロック単位の compile が使えるか（CUDA と Triton）。"""
    import importlib.util

    try:
        import torch

        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return False
    except Exception:  # noqa: BLE001
        return False
    return importlib.util.find_spec("triton") is not None


@router.get("/api/v1/lora/train_defaults")
def train_defaults(base: str) -> JSONResponse:
    """What a job uses for each setting when the request leaves it empty.

    Read from the base's training config so the screen shows the real values
    instead of hiding them; the schedule shares and validation count are the
    server's own (see _lr_schedule_for / _valid_ratio_for).
    """
    if base not in BASE_CONFIGS:
        raise HTTPException(400, f"unknown base: {base}")
    import yaml

    cfg = yaml.safe_load((_PROJECT_ROOT / BASE_CONFIGS[base]["config_file"]).read_text(encoding="utf-8"))
    t = cfg.get("train", {}) or {}
    return JSONResponse(content={
        "learning_rate": t.get("learning_rate"),
        "text_encoder_lr": t.get("pretrained_text_encoder_learning_rate"),
        "lora_r": t.get("lora_r"),
        "lora_alpha": t.get("lora_alpha"),
        "lora_dropout": t.get("lora_dropout"),
        "weight_decay": t.get("weight_decay"),
        "min_lr_scale": t.get("min_lr_scale"),
        "warmup_pct": 100.0 / _WARMUP_RATIO,
        "decay_pct": 25.0,
        "valid_clips": _MIN_VALID_CLIPS,
        "seed": t.get("seed", 0),
        "gradient_checkpointing": bool(t.get("gradient_checkpointing", False)),
        # 使えるのは NVIDIA の GPU と Triton があるときだけ。測ったのは v4-Large で、
        # 既定のオンもそこだけにする（他のモデルは測っていない）。
        "compile_blocks": base == "v4_large" and compile_available(),
        "compile_available": compile_available(),
        "ref_max_seconds": t.get("ref_max_seconds"),
        # 読み取り結果の使い回しをする学習か（本体メモリ・ディスクを使う旨の注意書きを出すため）
        "cache_condition_states": bool(t.get("lora_cache_condition_states", False)),
        "cache_skipped_presets": list(READER_TRAINING_PRESETS),
    })


# ===== おまかせ設定のための試し走り（Client 独自） =====
#
# バッチサイズを VRAM の総量の表で決めると、ほかのアプリが VRAM を使っているとき・素材が長いとき・
# 測っていない GPU のときに外れる。学習と同じ条件で数ステップだけ実際に動かし、使った量を測って決める。
_PROBE_CANDIDATES = (8, 4, 2, 1)
_PROBE_EFFECTIVE_BATCH = 32          # バッチ×勾配の蓄積。学習の中身を変えないよう、これは固定する
_PROBE_USABLE = 0.85                 # 上限のうち、ここまでに収まるバッチを選ぶ（おすすめで失敗させないための余裕）
_PROBE_CLIPS = 16                    # 試し走りに使う本数。長い順に取る（いちばん VRAM を使う組み合わせで測る）
_PROBE_STEPS = 2
_PROBE_REF_SECONDS_SHORT = 30.0
_PROBE_RE = re.compile(r"^VRAM probe: setup=([\d.]+) GB peak=([\d.]+) GB")
_probe_active = False
# 試し走りが、いま何をしているか。画面が読んで出す（待ち時間が長いので、止まって見えないようにする）。
_probe_status: dict = {"active": False}


class MeasureBatchRequest(BaseModel):
    dataset: str
    base: str
    preset: str = "speaker_style"
    # 本番と同じ条件で測る。コンパイルすると使う VRAM が減る（v4-Large・バッチ 8 で 11.84GB → 10.33GB、実測）ので、
    # コンパイルなしで測ると、収まるはずのバッチを取りこぼす。空なら、このベースモデルの既定に合わせる。
    compile_blocks: Optional[bool] = None
    # かんたん学習が自分の工程の中から呼ぶときの合言葉（start_job と同じ扱い）
    easy_token: Optional[str] = None


def _probe_manifest(dataset: str, log) -> tuple[Path, int]:
    """長い順に _PROBE_CLIPS 本だけの manifest を作る。latent が無ければ先に作る。"""
    manifest = get_dataset_dir(dataset) / "manifest.jsonl"

    def rows() -> list[dict]:
        return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not manifest.is_file() or any(not Path(r["latent_path"]).is_file() for r in rows()):
        build_manifest(dataset, log=log)
    longest = sorted(rows(), key=lambda r: -int(r.get("num_frames", 0)))[:_PROBE_CLIPS]
    out = lora_jobs_dir() / "_probe" / "manifest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in longest), encoding="utf-8")
    return out, len(longest)


def _probe_once(*, config_file: Path, manifest: Path, num_clips: int, init_ckpt: Path, preset: str,
                batch: int, ref_max_seconds: Optional[float], gradient_checkpointing: Optional[bool],
                compile_blocks: bool) -> dict:
    """train.py を試し走りで 1 回動かし、使った VRAM を返す。"""
    out = lora_jobs_dir() / "_probe" / "out"
    shutil.rmtree(out, ignore_errors=True)
    cmd = _build_train_command(
        config_file=config_file, manifest_path=manifest, output_dir=out, init_checkpoint=init_ckpt,
        lora_target_modules=PRESETS[preset], max_steps=_PROBE_STEPS, batch_size=batch,
        gradient_accumulation_steps=1, save_every=10**6, log_every=1, num_clips=num_clips, valid_clips=0,
        gradient_checkpointing=gradient_checkpointing, compile_blocks=compile_blocks, ref_max_seconds=ref_max_seconds,
    ) + ["--probe"]
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(_PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    # 出力を読みながら、いまの段階を控える。読み込み → コンパイル → 試しの学習、の順に進む。
    _probe_status.update(batch=batch, phase="モデルを読み込んでいます", compile=bool(compile_blocks), run_started=time.time())
    lines = []
    for raw in proc.stdout:
        line = raw.strip()
        lines.append(line)
        if line.startswith("torch.compile per block"):
            _probe_status["phase"] = "コンパイルしています"
        elif _STEP_RE.match(line):
            _probe_status["phase"] = "試しの学習をしています"
        if time.time() - _probe_status["run_started"] > 1800:      # 止まったまま居座らせない
            _terminate_pid(proc.pid)
            break
    proc.wait()
    found = next((m for line in lines if (m := _PROBE_RE.match(line))), None)
    cap = next((float(m.group(1)) for line in lines if (m := _VRAM_CAP_RE.match(line))), None)
    ok = proc.returncode == 0 and found is not None
    params = {"batch_size": batch, "gradient_accumulation_steps": 1, "ref_max_seconds": ref_max_seconds,
              "gradient_checkpointing": gradient_checkpointing}
    return {"batch": batch, "ok": ok, "oom": any(_OOM_RE.search(line) for line in lines),
            "setup_gib": float(found.group(1)) if found else None, "peak_gib": float(found.group(2)) if found else None,
            "cap_gib": cap, "ref_max_seconds": ref_max_seconds, "gradient_checkpointing": gradient_checkpointing,
            "error": "" if ok else failure_message(proc.returncode, lines, params)}


def pick_batch(measured: dict) -> int:
    """測った 1 回から、上限の _PROBE_USABLE に収まる最大のバッチを見積もる。

    学習中に増える量はバッチにほぼ比例するので、(peak - setup) / batch を 1 本あたりの量とみなす。
    """
    per_clip = max(1e-6, (measured["peak_gib"] - measured["setup_gib"]) / measured["batch"])
    room = measured["cap_gib"] * _PROBE_USABLE - measured["setup_gib"]
    return next((b for b in _PROBE_CANDIDATES if per_clip * b <= room), 0)


@router.get("/api/v1/lora/measure_batch/status")
def measure_batch_status() -> JSONResponse:
    """試し走りの進み具合。画面が、待っている間に出す。"""
    s = dict(_probe_status)
    if s.get("active"):
        s["elapsed"] = round(time.time() - s.get("started", time.time()))
    return JSONResponse(content=s)


@router.post("/api/v1/lora/measure_batch")
def measure_batch(req: MeasureBatchRequest) -> JSONResponse:
    """おまかせ設定：この PC・この素材で収まるバッチサイズを、試し走りで測って返す。"""
    global _probe_active
    if req.base not in BASE_CONFIGS:
        raise HTTPException(400, f"unknown base: {req.base}")
    if req.preset not in PRESETS:
        raise HTTPException(400, f"unknown preset: {req.preset}")
    import server_easy  # lazily: server_easy imports this module
    from irodori_tts import vram_guard

    with server_easy.GPU_LOCK, _active_lock:
        easy_busy = req.easy_token != server_easy.INTERNAL_TOKEN and server_easy.active_job_id()
        if _active_job_id is not None or easy_busy or _probe_active:
            raise HTTPException(409, "学習の最中は測れません。終わってから、もう一度押してください。")
        _probe_active = True
    started = time.time()
    runs: list[dict] = []
    _probe_status.clear()
    _probe_status.update(active=True, started=started, run=0, batch=None, phase="素材を用意しています", compile=False)
    try:
        _free_inference_models()
        manifest, num_clips = _probe_manifest(req.dataset, log=lambda m: print(f"[probe] {m}", flush=True))
        base_cfg = BASE_CONFIGS[req.base]
        manual = manual_checkpoint(base_cfg["repo_id"])
        if manual is None and base_cfg["repo_id"] in OPTIONAL_MODELS:
            raise HTTPException(400, missing_optional_model_message(base_cfg["repo_id"]))
        init_ckpt = Path(manual) if manual is not None else Path(
            hf_hub_download(repo_id=base_cfg["repo_id"], filename="model.safetensors"))
        config_file = _PROJECT_ROOT / base_cfg["config_file"]
        defaults = json.loads(train_defaults(req.base).body)
        # v4-Large は参照 30 秒が標準（8本×4回・参照30秒と 4本×8回・参照120秒で、聞き比べに差が無かった）。
        ref = _PROBE_REF_SECONDS_SHORT if req.base == "v4_large" else defaults["ref_max_seconds"]
        grad_ckpt = defaults["gradient_checkpointing"]
        use_compile = defaults["compile_blocks"] if req.compile_blocks is None else bool(req.compile_blocks)
        if use_compile and not compile_available():
            raise HTTPException(400, "この環境では「学習を速くする」を使えません（NVIDIA の GPU と Triton が必要です）。")

        def run(batch: int) -> dict:
            _probe_status["run"] = len(runs) + 1
            result = _probe_once(config_file=config_file, manifest=manifest, num_clips=num_clips, init_ckpt=init_ckpt,
                                 preset=req.preset, batch=batch, ref_max_seconds=ref, gradient_checkpointing=grad_ckpt,
                                 compile_blocks=use_compile)
            runs.append(result)
            print(f"[probe] batch={batch} ref={ref} grad_ckpt={grad_ckpt} compile={use_compile} -> {result}", flush=True)
            if not result["ok"] and not result["oom"]:
                raise HTTPException(500, result["error"])   # VRAM 以外の失敗。小さくしても直らない
            return result

        # 1 回目は、いまの空きから選ぶ（1回の試し走りが v4-Large で2〜3分かかるので、当たりそうな値から始める）。
        # 収まらなければ半分にしていく。動いても余裕が足りなければ、測った値から小さいほうを見積もる。
        free = vram_guard.free_gib()
        batch = 8 if (free or 0) >= 15 else 4 if (free or 0) >= 11 else 2 if (free or 0) >= 7 else 1
        first = run(batch)
        while not first["ok"] and batch > 1:
            batch //= 2
            first = run(batch)
        if not first["ok"] and ref and ref > _PROBE_REF_SECONDS_SHORT:
            ref = _PROBE_REF_SECONDS_SHORT           # バッチ 1 でも収まらない → 参照音声を短くする
            first = run(1)
        if not first["ok"] and not grad_ckpt:
            grad_ckpt = True                         # それでも収まらない → 勾配チェックポイントを入れる
            first = run(1)
        if not first["ok"]:
            raise HTTPException(507, "この PC の空き VRAM では、このモデルの学習が収まりませんでした。"
                                     "ほかのアプリを閉じるか、軽いモデル（v4.1-Small など）を選んでください。")
        chosen = first
        best = pick_batch(first)
        if best > batch:
            bigger = run(best)                       # 見積もりで増やせるときは、実際に動かして確かめる
            if bigger["ok"] and bigger["peak_gib"] <= bigger["cap_gib"] * _PROBE_USABLE:
                chosen = bigger
        elif 0 < best < batch:
            chosen = dict(first, batch=best)         # 動きはしたが余裕が足りない → 小さいほうを選ぶ（測った値より軽い）
        size = chosen["batch"]
        return JSONResponse(content={
            "batch_size": size,
            "gradient_accumulation_steps": max(1, _PROBE_EFFECTIVE_BATCH // size),
            "ref_max_seconds": ref,
            "gradient_checkpointing": bool(grad_ckpt),
            "compile_blocks": use_compile,
            "runs": runs,
            "seconds": round(time.time() - started, 1),
        })
    finally:
        shutil.rmtree(lora_jobs_dir() / "_probe", ignore_errors=True)
        _probe_active = False
        _probe_status.clear()
        _probe_status["active"] = False


@router.post("/api/v1/lora/jobs")
def start_job(req: StartJobRequest) -> JSONResponse:
    global _active_job_id
    if req.compile_blocks and not compile_available():
        # 受けてしまうと、学習の準備まで進んでから compile で落ちる。
        raise HTTPException(400, "学習の高速化はこの環境では使えません（NVIDIA の GPU と Triton が必要です）。")
    if req.base not in BASE_CONFIGS:
        raise HTTPException(400, f"base must be one of {list(BASE_CONFIGS)}")
    if req.preset not in PRESETS:
        raise HTTPException(400, f"preset must be one of {list(PRESETS)}")

    # かんたん学習が動いている間は始めない。同じ GPU で2つ学習すると
    # 8GB 級では両方落ちる。かんたん学習自身がここを呼ぶので、合言葉を
    # 持っているものだけ通す。
    import server_easy

    dataset_root = get_dataset_dir(req.dataset)
    if not dataset_root.is_dir():
        raise HTTPException(404, f"dataset {req.dataset!r} not found")
    if not (dataset_root / "clips").is_dir():
        raise HTTPException(400, f"dataset {req.dataset!r} has no clips/")

    # 名前の重複はここでは見ない。学習しただけでは何も登録されず、レジストリに
    # 触るのは採用のとき（/register）だけなので、開始時点で名前がぶつかっても
    # 実害がない。同名を上書きするかどうかは、採用する人がその場で判断する。
    # req.lora_name は History の表示と、採用時の既定の登録名に使う。

    # 受付はかんたん学習と同じロックの中で決める。外で確認すると、互いに
    # 「相手は動いていない」と見てから互いに登録してしまう。
    with server_easy.GPU_LOCK, _active_lock:
        if (req.easy_token != server_easy.INTERNAL_TOKEN
                and server_easy.active_job_id()):
            raise HTTPException(
                409, "かんたん学習の最中です。終わるまで学習を始められません。")
        if _probe_active:
            raise HTTPException(409, "おまかせ設定の測定中です。終わるまで学習を始められません。")
        if _active_job_id is not None:
            cur = _read_status(_active_job_id) or {}
            # 判定は _ACTIVE_STATES に一本化する。ここにタプルをベタ書きすると
            # 状態を追加したときに片方だけ更新され、ガードが黙って開く。
            if cur.get("state") in _ACTIVE_STATES:
                raise HTTPException(409, f"another job is active: {_active_job_id}")

        # メモリ上のロックはこのプロセス限りで、再起動すると消える。
        # recover_orphan_jobs が「生き残りを否定できない」と判断したジョブは
        # ディスク側に stop_failed で残るので、そちらも見る。これが無いと
        # 再起動しただけで排他が外れ、旧 trainer と GPU を奪い合う。
        blocker = _find_unresolved_job()
        if blocker is not None:
            raise HTTPException(
                409,
                f"job {blocker} may still have a training process running; "
                "stop it manually (Task Manager) before starting a new one",
            )

        job_id = uuid.uuid4().hex[:12]
        _active_job_id = job_id
        # status はロックを持ったまま作る。ここを抜けてから書くと、その隙間に
        # 来た2件目の要求が _read_status で {} を受け取り、_ACTIVE_STATES 判定を
        # 素通りして _active_job_id を上書きする（学習が2本同時に走る）。
        _write_status(job_id, {
            "id": job_id,
            "name": req.lora_name,
            "dataset": req.dataset,
            "base": req.base,
            "preset": req.preset,
            "max_steps": req.max_steps,
            "state": "pending",
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "current_step": 0,
            "current_loss": None,
            "valid_loss": None,
            "log_tail": [],
            "error": None,
            "pid": None,
        })

    params = req.model_dump()
    t = threading.Thread(target=_run_job, args=(job_id, params), daemon=True)
    t.start()

    return JSONResponse(content={"status": "ok", "job_id": job_id})


@router.get("/api/v1/lora/jobs")
def list_jobs() -> JSONResponse:
    items = []
    for p in sorted(lora_jobs_dir().iterdir(), reverse=True):
        if not p.is_dir():
            continue
        s = _read_status(p.name)
        if s:
            items.append({
                "id": s.get("id"),
                "name": s.get("name"),
                "state": s.get("state"),
                # base は試聴時の model_type に使う。欠けると呼び出し側が
                # 既定値へ落ち、v3 で学習したものを v4 で鳴らしてしまう。
                "base": s.get("base"),
                "current_step": s.get("current_step"),
                "max_steps": s.get("max_steps"),
                "current_loss": s.get("current_loss"),
                "created_at": s.get("created_at"),
                "finished_at": s.get("finished_at"),
                # 採用済みかどうかは History の表示と導線を分ける材料になる。
                "registered_as": s.get("registered_as"),
                "registered_checkpoint": s.get("registered_checkpoint"),
            })
    return JSONResponse(content={"jobs": items})


@router.get("/api/v1/lora/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    s = _read_status(job_id)
    if s is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    return JSONResponse(content=s)


@router.post("/api/v1/lora/jobs/{job_id}/stop")
def stop_job(job_id: str) -> JSONResponse:
    s = _read_status(job_id)
    if s is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    if s.get("state") == "evaluating":
        # 学習は終わっている。測るのを次のチェックポイントの手前で止め、
        # 測れた分だけでおすすめを出す。学習プロセスはもう無いので kill しない
        # （記録に残った pid は別のプロセスに再利用されているかもしれない）。
        _mark_stop_requested(job_id)
        return JSONResponse(content={"status": "ok", "job_id": job_id, "state": "evaluating"})
    if s.get("state") not in ("pending", "preparing", "training"):
        return JSONResponse(content={"status": "noop", "state": s.get("state")})

    pid = s.get("pid")
    # ワーカーが「異常終了」と取り違えないよう、kill する前に印を付ける。
    _mark_stop_requested(job_id)
    _update_status(job_id, state="stopping")  # UI 表示用。終端状態はこの後に決まる。

    kill_error = _terminate_pid(pid) if pid else None

    if kill_error:
        # kill に失敗した = 学習プロセスがまだ生きている可能性が高い。
        # ここを "failed" のような終端状態にすると、開始側のガードが
        # 通ってしまい、生き残ったプロセスと新しいジョブが同時に GPU を掴む。
        # 走行中でも終了済みでもない専用の状態にして、ロックも保持する。
        # プロセスが後で終わればワーカーが "stopped" を書いて解放する。
        # 終わらなければサーバー再起動時に recover_orphan_jobs が拾う。
        _update_status(
            job_id,
            state=_STOP_FAILED,
            error=f"stop requested but killing pid {pid} failed: {kill_error}",
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": f"failed to kill pid {pid}: {kill_error}",
                "state": _STOP_FAILED,
                "hint": "training process may still be running; check Task Manager",
            },
        )

    # kill 成功時の終端状態はワーカーに任せる。proc.wait() から復帰した側が
    # _was_stop_requested を見て "stopped" を書き、ロックも解放する。
    # ここで先に書くと二重更新になり、ワーカーの後始末と競合する。
    return JSONResponse(content={"status": "ok", "job_id": job_id, "state": "stopping"})


# train.py writes checkpoint_best_val_loss_<step>_<loss> alongside plain
# checkpoint_<step> and checkpoint_final. Parsing the name is how the step and
# score get back out — nothing else records them per checkpoint.
_CKPT_BEST_RE = re.compile(r"^checkpoint_best_val_loss_(\d+)_(\d+(?:\.\d+)?)$")
_CKPT_STEP_RE = re.compile(r"^checkpoint_(\d+)$")


def _adapter_digest(ckpt_dir: Path) -> Optional[str]:
    """Cheap fingerprint of a checkpoint's weights, or None if unreadable.

    一覧を出すたびに 95MB を読み切るのは重いので、大きさと先頭・末尾だけで
    見分ける。狙って衝突させられる相手がいる場所ではなく、区別したいのは
    「同じ学習が書いた同一ファイル」かどうかだけなので、これで足りる。
    """
    weights = ckpt_dir / "adapter_model.safetensors"
    try:
        size = weights.stat().st_size
        with weights.open("rb") as fh:
            head = fh.read(65536)
            if size > 131072:
                fh.seek(-65536, os.SEEK_END)
                tail = fh.read(65536)
            else:
                tail = b""
    except OSError:
        return None
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(str(size).encode())
    h.update(head)
    h.update(tail)
    return h.hexdigest()


def _val_losses_from_log(job_id: str) -> dict[int, float]:
    """step -> val_loss, read back from the run's log.

    以前は checkpoint_best_val_loss_<step>_<loss> というディレクトリ名から
    読んでいたが、全世代を残すために保持機構を切った（--checkpoint-best-n 0）
    ので、その名前は作られなくなった。検証自体は valid_every ごとに走って
    ログに出ているため、そちらを唯一の出所にする。
    """
    log_path = _job_dir(job_id) / "training.log"
    if not log_path.is_file():
        return {}
    out: dict[int, float] = {}
    try:
        with log_path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _VALID_STEP_RE.match(line.strip())
                if m:
                    try:
                        out[int(m.group(1))] = float(m.group(2))
                    except ValueError:
                        continue
    except OSError:
        return {}
    return out


def _list_job_checkpoints(job_id: str) -> list[dict]:
    """Every usable adapter directory a job left behind, best score first.

    Only one checkpoint is registered when training ends, and it is whichever
    the picker happened to choose. The rest sit on disk unheard, so list them
    all and let the caller audition them.
    """
    from irodori_tts.lora import is_lora_adapter_dir

    out_dir = _job_dir(job_id) / "train_output"
    if not out_dir.is_dir():
        return []
    log_losses = _val_losses_from_log(job_id)
    items: list[dict] = []
    for child in sorted(out_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("checkpoint"):
            continue
        if not is_lora_adapter_dir(child):
            continue
        step: int | None = None
        val_loss: float | None = None
        m = _CKPT_BEST_RE.match(child.name)
        if m:
            step = int(m.group(1))
            try:
                val_loss = float(m.group(2))
            except ValueError:
                val_loss = None
        else:
            m2 = _CKPT_STEP_RE.match(child.name)
            if m2:
                step = int(m2.group(1))
        # 名前から取れないぶんはログから補う。定期チェックポイントの名前には
        # loss が入らないので、これが無いと一覧の val がすべて空になる。
        if val_loss is None and step is not None:
            val_loss = log_losses.get(step)
        elif val_loss is None and child.name == "checkpoint_final" and log_losses:
            # 最終だけは名前に step を持たない。最後に検証した値を当てる。
            last_step = max(log_losses)
            step, val_loss = last_step, log_losses[last_step]
        items.append({
            "name": child.name,
            "path": str(child.resolve()),
            "step": step,
            "val_loss": val_loss,
            "is_final": child.name == "checkpoint_final",
        })
    # max_steps が save_every の倍数だと、最終ステップの定期チェックポイントと
    # checkpoint_final が同じ重みになる。同じものが二つ並ぶと試聴でどちらを
    # 選べばよいか分からないので、片方だけ残す。残すのは checkpoint_final ——
    # 「最終」と分かる名前のほうが選びやすく、再開地点として
    # trainer_state.pt を持っているのもこちらなので。
    #
    # 判定は重みの中身で行う。step が同じでも別物のことがあり（実測で
    # 同 step の 3 件中 1 件は別ハッシュだった）、step だけで畳むと
    # 聴けるはずの世代が一覧から消える。
    final = next((c for c in items if c["is_final"]), None)
    if final is not None:
        final_digest = _adapter_digest(Path(final["path"]))
        if final_digest is not None:
            items = [
                c for c in items
                if c["is_final"] or _adapter_digest(Path(c["path"])) != final_digest
            ]

    # Scored ones first, ascending loss; then the rest by step.
    items.sort(key=lambda c: (
        c["val_loss"] is None,
        c["val_loss"] if c["val_loss"] is not None else 0.0,
        c["step"] if c["step"] is not None else 0,
    ))
    return items


@router.get("/api/v1/lora/jobs/{job_id}/checkpoints")
def list_job_checkpoints(job_id: str) -> JSONResponse:
    s = _read_status(job_id)
    if s is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    return JSONResponse(content={
        "job_id": job_id,
        "name": s.get("name"),
        "base": s.get("base"),
        "registered_as": s.get("registered_as"),
        "registered_checkpoint": s.get("registered_checkpoint"),
        "checkpoints": _list_job_checkpoints(job_id),
        "checkpoints_bytes": _job_checkpoints_size(job_id),
        # 学習後の測定（SIM / CER）とおすすめ。測っていないジョブは None
        "evaluation": s.get("evaluation"),
    })


def _job_checkpoints_size(job_id: str) -> int:
    """Bytes held by this job's checkpoints, so the UI can show what freeing costs."""
    out_dir = _job_dir(job_id) / "train_output"
    if not out_dir.is_dir():
        return 0
    total = 0
    for child in out_dir.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint"):
            total += sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
    return total


def _purge_job_checkpoints(job_id: str, keep_adopted: bool) -> tuple[int, int]:
    """Delete a job's checkpoints. Returns (removed, freed_bytes).

    keep_adopted なら、採用した回だけ残す。registered_checkpoint はこの項目を
    足してから採用したものにしか入っていないが、それ以前は学習完了時の自動登録
    しかなく、その登録元は常に checkpoint_final だった。よって記録が無い
    採用済みジョブは checkpoint_final を採ったものとして扱う。
    """
    s = _read_status(job_id) or {}
    keep = None
    if keep_adopted:
        keep = s.get("registered_checkpoint")
        if not keep and s.get("registered_as"):
            keep = "checkpoint_final"

    out_dir = _job_dir(job_id) / "train_output"
    if not out_dir.is_dir():
        return 0, 0

    # 「採用した回だけ残す」つもりで全部消える経路を塞ぐ。採用した世代を先に
    # 個別削除していると keep が実体の無い名前を指し、どれとも一致しないまま
    # 残り全部が消える。残す当てが無いなら、何もしないほうが取り返しがつく。
    if keep_adopted and not (keep and (out_dir / keep).is_dir()):
        raise HTTPException(
            409,
            f"採用した回 {keep!r} が見つかりません。"
            "残すものが決まらないので削除しませんでした。",
        )

    removed = 0
    freed = 0
    for child in sorted(out_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("checkpoint"):
            continue
        if keep and child.name == keep:
            continue
        size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
        try:
            shutil.rmtree(child)
        except OSError as exc:
            raise HTTPException(500, f"could not delete {child.name}: {exc}")
        removed += 1
        freed += size
    return removed, freed


@router.delete("/api/v1/lora/jobs/{job_id}/checkpoints")
def delete_job_checkpoints(job_id: str, keep_adopted: bool = False) -> JSONResponse:
    """Drop a finished job's checkpoints, keeping the record and the log.

    採用が済めば、その世代は聴き直す用がなくなる。ただしいつ用済みになるかは
    人にしか決められないので、自動では消さずこの入口だけ用意する。
    消えるのは train_output/checkpoint* だけで、status.json と training.log、
    登録済みの LoRA には触れない。
    keep_adopted=true なら、採用した回だけ残して他を消す。
    """
    s = _read_status(job_id)
    if s is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    if s.get("state") in _ACTIVE_STATES:
        raise HTTPException(409, "job is still running; stop it first")

    removed, freed = _purge_job_checkpoints(job_id, keep_adopted)
    return JSONResponse(content={
        "status": "ok",
        "job_id": job_id,
        "removed": removed,
        "freed_bytes": freed,
    })


@router.delete("/api/v1/lora/jobs/{job_id}/checkpoints/{checkpoint}")
def delete_one_checkpoint(job_id: str, checkpoint: str) -> JSONResponse:
    """Delete a single generation from a job."""
    s = _read_status(job_id)
    if s is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    if s.get("state") in _ACTIVE_STATES:
        raise HTTPException(409, "job is still running; stop it first")

    # 名前は一覧に出したものだけ受け付ける。パスを組み立てさせない。
    entry = next(
        (c for c in _list_job_checkpoints(job_id) if c["name"] == checkpoint), None
    )
    if entry is None:
        raise HTTPException(404, f"checkpoint {checkpoint!r} not found for this job")

    target = Path(entry["path"])
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise HTTPException(500, f"could not delete {checkpoint}: {exc}")

    # 採用した回そのものを消したなら、その記録も落とす。残しておくと
    # 実体の無い名前を指したままになり、「採用以外を削除」が残す当てを
    # 見失う。登録済みの LoRA は別物なので registered_as は触らない。
    if s.get("registered_checkpoint") == checkpoint:
        _update_status(job_id, registered_checkpoint=None)

    return JSONResponse(content={
        "status": "ok",
        "job_id": job_id,
        "removed": 1,
        "freed_bytes": size,
    })


class RegisterCheckpointRequest(BaseModel):
    checkpoint: str = Field(..., description="Checkpoint directory name from the listing")
    lora_name: str = Field(..., description="Name to register it under")


@router.post("/api/v1/lora/jobs/{job_id}/register")
def register_job_checkpoint(job_id: str, req: RegisterCheckpointRequest) -> JSONResponse:
    """Register a chosen checkpoint under a name, replacing any existing one."""
    s = _read_status(job_id)
    if s is None:
        raise HTTPException(404, f"job {job_id!r} not found")

    available = {c["name"]: c for c in _list_job_checkpoints(job_id)}
    entry = available.get(req.checkpoint)
    if entry is None:
        raise HTTPException(
            400,
            f"checkpoint {req.checkpoint!r} not found for this job "
            f"(have: {sorted(available)})",
        )

    lora_name = (req.lora_name or "").strip()
    if not lora_name:
        raise HTTPException(400, "lora_name must not be empty")

    base = s.get("base") or "v4"
    step = entry.get("step")
    val = entry.get("val_loss")
    notes = f"picked {req.checkpoint}"
    if step is not None:
        notes += f" (step {step}"
        notes += f", val_loss {val}" if val is not None else ""
        notes += ")"
    _register_lora(lora_name, base, Path(entry["path"]), notes)
    # 試聴の一時登録は名前を隠しているぶん /loras に出てこない。呼び出し側が
    # 声質の有無を引けないと、完パケでも参照音声を足してしまい本番と違う音になる。
    # 判定はアダプタ自体から出るので、ここで返す。
    from server_lora import _provides_from_meta, is_preview_lora
    provides = _provides_from_meta({}, Path(entry["path"]))

    # 採用したことをジョブ側にも残す。学習しただけでは何も登録されないので、
    # どの回をどの名前で採ったかはここでしか記録されない。
    # 試聴の鳴らし比べは同じ API を一時名で叩くだけなので、採用とは扱わない。
    if not is_preview_lora(lora_name):
        _update_status(
            job_id,
            registered_as=lora_name,
            registered_checkpoint=req.checkpoint,
        )
    return JSONResponse(content={
        "status": "ok",
        "lora_name": lora_name,
        "checkpoint": req.checkpoint,
        "base": base,
        "provides": provides,
    })
