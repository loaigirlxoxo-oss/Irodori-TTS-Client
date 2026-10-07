"""読み取り結果の使い回し（model.ConditionStateCache）を、ディスクに置いても同じ値が返るかのテスト。GPU は使わない。
  python scripts/test_condition_cache_disk.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irodori_tts.model import ConditionStateCache  # noqa: E402

PAD, LENGTH, DIM = 0, 12, 8


def build(dtype: torch.dtype, directory: Path | None) -> tuple[ConditionStateCache, torch.Tensor, torch.Tensor, list]:
    """3行ぶんの結果を入れた入れ物を作る。directory を渡すとディスクに置く。"""
    torch.manual_seed(7)
    ids = torch.tensor([[5, 6, 7, PAD, PAD], [8, 9, PAD, PAD, PAD], [3, 4, 5, 6, 7]])
    mask = ids != PAD
    cache = ConditionStateCache({"text": PAD, "caption": PAD})
    values = []
    if directory is not None:
        cache.open_disk("text", directory / "text.bin", ids.shape[0], LENGTH, DIM, dtype)
    for i in range(ids.shape[0]):
        norm, drop = torch.randn(LENGTH, DIM).to(dtype), torch.randn(LENGTH, DIM).to(dtype)
        values.append((norm, drop))
        cache.put("text", ConditionStateCache.make_key(ids[i], int(mask[i].sum()), PAD), norm, drop)
    return cache, ids, mask, values


def check(dtype: torch.dtype) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cond_cache_test_"))
    try:
        ram, ids, mask, values = build(dtype, None)
        disk, _, _, _ = build(dtype, tmp)
        dropped = mask.clone()
        dropped[1] = False                                   # 2行目は文を隠す（全部 False のマスク）
        for use_mask in (mask, dropped):
            a = ram.lookup("text", ids, mask, use_mask)
            b = disk.lookup("text", ids, mask, use_mask)
            assert a.dtype == b.dtype == dtype and a.shape == b.shape == (3, ids.shape[1], DIM), (a.dtype, b.shape)
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "ディスクとメモリで値が違う"
        got = disk.lookup("text", ids, mask, dropped)
        assert torch.equal(got[1].view(torch.uint8), values[1][1][: ids.shape[1]].view(torch.uint8))   # 隠した行は drop 側
        assert torch.equal(got[0].view(torch.uint8), values[0][0][: ids.shape[1]].view(torch.uint8))   # ほかは norm 側
        assert (tmp / "text.bin").stat().st_size == 3 * 2 * LENGTH * DIM * torch.empty((), dtype=dtype).element_size()
        disk.close()
        shutil.rmtree(tmp)                                   # 閉じたあとは Windows でも消せる
        assert not tmp.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bf16_round_trip() -> None:
    check(torch.bfloat16)


def test_fp32_round_trip() -> None:
    check(torch.float32)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
