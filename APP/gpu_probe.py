"""GPU の世代と VRAM を調べる。生成の精度の自動選択と、モデル一覧の「この GPU で動くか」に使う。

server.py と、初回セットアップのモデル一覧（model_catalog.py --json）の両方から使う。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from irodori_tts.inference_runtime import default_runtime_device  # noqa: E402

# bf16 のネイティブ命令があるのは Ampere（Compute Capability 8.0）以降。
# それ未満の GPU でも torch は bf16 を受け付けるが、変換を挟んで動くので
# 実行が遅くなるだけになる。ランタイム側の可否判定は device の種類しか
# 見ないので、世代の判断はここで持つ。
#
# 注意: 対応世代でも bf16 が「速い」わけではない。このパイプラインでは
# 実測で fp32 1.45s 対 bf16 1.75s（約 1.2 倍おそい）。行列積単体では
# bf16 が速い（4096^2 で fp32 6.77ms 対 bf16 2.57ms）ので、GPU の性能では
# なく前後の処理で相殺されている。bf16 の利点は VRAM で、常駐が
# 3460 -> 1785 MiB に減る。速度と VRAM の交換であって両立ではない。
# auto が bf16 を選ぶのは、8GB 級で v4 が載るかどうかを分ける差だから。
_BF16_MIN_CAPABILITY = (8, 0)


def gpu_info() -> dict:
    """UI が精度を選ぶための材料。CUDA 以外では bf16 を出さない。"""
    info = {"device": default_runtime_device(), "name": None,
            "capability": None, "bf16_fast": False, "rocm": False}
    try:
        import torch
        # ROCm 版 torch は torch.cuda として振る舞うので、cuda かどうかでは
        # 区別できない。torch.version.hip が入っているかで見る。
        info["rocm"] = bool(getattr(torch.version, "hip", None))
        if torch.cuda.is_available():
            info["name"] = torch.cuda.get_device_name(0)
            if not info["rocm"]:
                cap = torch.cuda.get_device_capability(0)
                info["capability"] = f"{cap[0]}.{cap[1]}"
                info["bf16_fast"] = cap >= _BF16_MIN_CAPABILITY
            else:
                # Radeon の bf16 は AMD の資料に明記が無く、実機で確かめて
                # いない。自動では選ばず、必要なら手で選んでもらう。
                info["capability"] = None
                info["bf16_fast"] = False
    except Exception:  # noqa: BLE001 - 情報が取れなくても生成は続けられる
        pass
    return info


def vram_gb() -> float | None:
    """GPU の VRAM の総量（GiB）。NVIDIA 以外・取れないときは None。"""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    return None
