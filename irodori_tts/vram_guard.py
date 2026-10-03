"""Client 独自：VRAM があふれる前に止める。

Windows のドライバは VRAM が足りないと、黙って本体のメモリへ退避する。そうなると
処理は止まらずに何倍も遅くなり、PC 全体が重くなる。2026-10-03 には、VRAM 16GB の
GPU で v4-Large の生成が 17GiB を確保した状態で、PC がバグチェックで落ちた。

PyTorch が確保できる量に上限をかけ、あふれる処理はその場で失敗させる。上限は
その PC の「いま」に合わせる：

    上限 = min(VRAM の総量 x 0.95, 自分がもう持っている量 + いま空いている量 - 余裕)

ほかのアプリ（ブラウザ・ゲーム・別の生成アプリ）が VRAM を使っていれば、そのぶん
上限は下がる。空きは nvidia-smi で見る。torch.cuda.mem_get_info は Windows では
GPU 全体の使用量を数えず、6.7GiB ずれた（実測）。nvidia-smi が無い環境（Radeon など）
では総量 x 0.95 だけをかける。

IRODORI_VRAM_FRACTION=0 で外せる。値を入れると総量に対する割合を変えられる。
"""
from __future__ import annotations

import os
import subprocess
import time

DEFAULT_FRACTION = 0.95
# いま空いている量から引く余裕。ほかのアプリが少し増やしても、あふれないようにする。
FREE_MARGIN_GIB = 0.5
# 生成のたびに nvidia-smi を起動しない。空きはこの秒数だけ使い回す。
_FREE_TTL_SEC = 3.0
_free_cache: tuple[float, float | None] = (0.0, None)


def _fraction() -> float:
    raw = os.environ.get("IRODORI_VRAM_FRACTION", "").strip()
    try:
        return min(float(raw), 1.0) if raw else DEFAULT_FRACTION
    except ValueError:
        return DEFAULT_FRACTION


def free_gib(index: int = 0) -> float | None:
    """GPU 全体で、いま空いている VRAM（GiB）。分からなければ None。"""
    global _free_cache
    now = time.monotonic()
    if now - _free_cache[0] < _FREE_TTL_SEC:
        return _free_cache[1]
    value = None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.splitlines()
        total_mib, used_mib = (float(x) for x in out[0].split(","))
        value = (total_mib - used_mib) / 1024
    except Exception:  # noqa: BLE001 - NVIDIA 以外、または nvidia-smi が無い
        value = None
    _free_cache = (now, value)
    return value


def apply(device=None) -> dict | None:
    """いまの空きに合わせて、このプロセスの VRAM の上限をかけ直す。

    GPU を使う処理（モデルの読み込み・生成・書き起こし・学習）の直前に呼ぶ。
    かけた上限を返す（ログと、足りないときの案内に使う）。CUDA でなければ None。
    """
    import torch

    if not torch.cuda.is_available():
        return None
    if device is not None and getattr(device, "type", str(device)) not in ("cuda",) \
            and not str(device).startswith("cuda"):
        return None
    fraction = _fraction()
    if fraction <= 0:
        return None
    index = getattr(device, "index", None)
    if index is None:
        index = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    cap = total * fraction
    free = free_gib(index)
    own = torch.cuda.memory_reserved(index)
    if free is not None:
        cap = min(cap, own + max(0.0, free - FREE_MARGIN_GIB) * 2**30)
    # 0 にはできない（PyTorch が受け付けない）。持っている量を下回っても、
    # 新しい確保が失敗するだけで、読み込み済みのものは壊れない。
    cap = max(cap, 64 * 2**20)
    torch.cuda.set_per_process_memory_fraction(min(cap / total, 1.0), index)
    return {"cap_gib": round(cap / 2**30, 2), "total_gib": round(total / 2**30, 2),
            "free_gib": None if free is None else round(free, 2), "own_gib": round(own / 2**30, 2)}


def out_of_memory_message(limit: dict | None) -> str:
    """VRAM が足りずに止めたときの、利用者向けの文。"""
    if not limit:
        return "VRAM が足りません。"
    if limit.get("free_gib") is not None:
        return (f"VRAM が足りません（この処理に使える量: {limit['cap_gib']:.1f}GB）。"
                "ほかのアプリを閉じるか、軽いモデル・短い文でお試しください。")
    return f"VRAM が足りません（上限: {limit['cap_gib']:.1f}GB）。軽いモデル・短い文でお試しください。"
