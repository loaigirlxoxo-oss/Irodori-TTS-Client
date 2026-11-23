"""作り直した LoRA が、古い中身のまま鳴らないこと（1.2.5 の修正 1）。

再現（実機で確かめられた手順を、小さな作り物のモデルで行う）:
  zzmix = A を読ませて出力 → zzmix を消して同じ名前で C を置く → zzmix で出力
  直す前は、2 回目も A の出力のままだった。直したあとは、別の置き場所に置いた C の出力と一致する。
ランタイムの LoRA の読み込み（_prepare_lora_for_request_inner）をそのまま使う。GPU も本体のモデルも使わない。
LoRA は丸ごと保存する層（尺の予測と同じ modules_to_save）も持たせる。読み込んでいる LoRA が 1 つだけのときに
古いものを先に消すと、PEFT が読み込みを断る（実機で 500 になった）ので、その形も確かめる。
使い方: python scripts/test_lora_rebuild_reload.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from irodori_tts.inference_runtime import InferenceRuntime  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="lora_rebuild_"))
X = torch.randn(3, 8, generator=torch.Generator().manual_seed(1))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 8)   # saved whole, like the duration predictor

    def forward(self, x):
        return self.head(self.proj(x))


def make_adapter(path: Path, seed: int, r: int, alpha: int):
    model = get_peft_model(Tiny(), LoraConfig(r=r, lora_alpha=alpha, target_modules=["proj"], modules_to_save=["head"]))
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n or "modules_to_save" in n:
                p.copy_(torch.randn_like(p) * 0.3)
    model.save_pretrained(str(path))
    return path / "default" if (path / "default").is_dir() else path


def runtime():
    rt = InferenceRuntime.__new__(InferenceRuntime)
    rt.model = Tiny()
    rt.key = SimpleNamespace(compile_model=False)
    rt.model_device = torch.device("cpu")
    rt._model_dtype = torch.float32
    rt._lora_adapter_names = OrderedDict()
    rt._lora_adapter_stamps = {}
    return rt


def run(rt, path, scale=1.0):
    ctx = rt._prepare_lora_for_request_inner(str(path), lora_scale=scale, messages=[], log_fn=lambda m: None)
    with ctx, torch.no_grad():
        return rt.model(X).clone()


def main():
    a = make_adapter(TMP / "src_a", 1, 16, 32)
    c = make_adapter(TMP / "src_c", 2, 80, 80)    # different rank and scaling, like a merged LoRA
    zz = TMP / "loras" / "zzmix"
    shutil.copytree(a, zz)

    rt = runtime()
    out_a = run(rt, zz)
    time.sleep(0.05)                              # mtime resolution on some file systems
    shutil.rmtree(zz)
    shutil.copytree(c, zz)
    for f in zz.iterdir():                        # copytree keeps the source mtime; a rebuild writes new files
        f.touch()
    out_rebuilt = run(rt, zz)

    other = runtime()
    out_c = run(other, c)
    assert not torch.allclose(out_a, out_c), "test LoRAs must differ"
    assert torch.allclose(out_rebuilt, out_c, atol=1e-6), "rebuilt LoRA still sounds like the old one"
    print("1. 同じ名前で作り直した LoRA: 別の置き場所の C と一致 OK")

    # the scaling of the rebuilt adapter is the new one (r=80/alpha=80 -> 1.0), not the old 2.0
    out_rebuilt_15 = run(rt, zz, 1.5)
    out_c_15 = run(other, c, 1.5)
    assert torch.allclose(out_rebuilt_15, out_c_15, atol=1e-6), "old scaling remained after the rebuild"
    print("2. 作り直したあとの強さ 1.5: C と一致 OK")

    # unchanged files: served from the cache, same output
    assert torch.allclose(run(rt, zz), out_c, atol=1e-6)
    print("3. 中身が同じなら読み直さずに同じ音 OK")
    print("すべて通過")


if __name__ == "__main__":
    main()
