"""LoRA の強さ: 同じ名前で読み直したアダプタに、前の LoRA の倍率が残らないこと（1.2.5 の修正 2）。

再現: r=16 / alpha=32（倍率 2.0）を読む → delete_adapter → 同じ名前で r=80 / alpha=80（倍率 1.0）を読む
→ set_lora_scale(1.0)。直す前は、覚えてあった 2.0 が使われて 2 倍で効いた。
小さな作り物のモデルで確かめる。GPU も本体のモデルも使わない。
使い方: python scripts/test_lora_scale_reload.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

from irodori_tts.lora import set_lora_scale  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.proj(x)


def scalings(model, name):
    return {float(m.scaling[name]) for m in model.modules() if isinstance(getattr(m, "scaling", None), dict) and name in m.scaling}


def main():
    name = "runtime_same_place"
    model = get_peft_model(Tiny(), LoraConfig(r=16, lora_alpha=32, target_modules=["proj"]), adapter_name=name)
    assert scalings(model, name) == {2.0}
    set_lora_scale(model, name, 1.0, fresh=True)
    assert scalings(model, name) == {2.0}

    model.add_adapter("keep", LoraConfig(r=4, lora_alpha=4, target_modules=["proj"]))   # PEFT keeps one adapter at least
    model.delete_adapter(name)
    model.add_adapter(name, LoraConfig(r=80, lora_alpha=80, target_modules=["proj"]))
    model.set_adapter(name)
    assert scalings(model, name) == {1.0}, scalings(model, name)

    # without fresh: the remembered 2.0 comes back (the bug this guards against)
    stale = set_lora_scale(model, name, 1.0) and scalings(model, name)
    assert stale == {2.0}, stale
    # restore what PEFT set, then check the fixed path
    for m in model.modules():
        if isinstance(getattr(m, "scaling", None), dict) and name in m.scaling:
            m.scaling[name] = 1.0
    set_lora_scale(model, name, 1.0, fresh=True)
    assert scalings(model, name) == {1.0}, scalings(model, name)
    set_lora_scale(model, name, 1.5)
    assert scalings(model, name) == {1.5}, scalings(model, name)
    print("読み直したアダプタの倍率: 1.0（強さ 1.5 で 1.5）OK")


if __name__ == "__main__":
    main()
