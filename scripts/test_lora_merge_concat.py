"""マージの「そのまま足す」（concat）が、ランクの違う LoRA を劣化なしで混ぜること（1.2.5）。

小さな作り物の LoRA を一時フォルダに作って確かめる。GPU もモデルも使わない。
  1. ランク 4（尺の予測を丸ごと持つ）とランク 8（持たない＝同梱の Aegis と同じ形）を 1:1 で混ぜる
     → 各層の B@A が、2 つを重みどおりに足したものと一致する
     → 尺の予測は、持っている方のものがそのまま入る。adapter_config の modules_to_save も残る
  2. 両方が尺の予測を持つとき → 重みどおりの平均になる
  3. 尺の予測を持つ 2 つで、持っている層が違うとき → 400 で断る
使い方: python scripts/test_lora_merge_concat.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
TMP = Path(tempfile.mkdtemp(prefix="merge_concat_"))
os.environ["IRODORI_DATA_DIR"] = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import server_lora  # noqa: E402

LAYERS = ["base_model.model.blocks.0.attention.wq.", "base_model.model.blocks.0.attention.wk."]
FULL = "base_model.model.duration_predictor.proj.weight"
torch.manual_seed(0)


def make(name, r, alpha, layers, full=None):
    d = TMP / "loras" / name / "adapter"
    d.mkdir(parents=True)
    t = {}
    for lay in layers:
        t[lay + "lora_A.weight"] = torch.randn(r, 16)
        t[lay + "lora_B.weight"] = torch.randn(16, r)
    if full is not None:
        t[FULL] = full
    save_file(t, str(d / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": alpha, "target_modules": "^(blocks\\.0\\.attention\\.(wq|wk))$",
           "modules_to_save": ["duration_predictor"] if full is not None else None}
    (d / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d.parent / "meta.json").write_text(json.dumps({"name": name, "base": "v4_1"}), encoding="utf-8")
    return t


def merged(name):
    d = TMP / "loras" / name / "adapter"
    return load_file(str(d / "adapter_model.safetensors")), json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))


def delta(t, lay, scale=1.0):
    return scale * t[lay + "lora_B.weight"].double() @ t[lay + "lora_A.weight"].double()


def main():
    full_a = torch.randn(8, 8)
    a = make("char_a", 4, 8, LAYERS, full_a)                 # like a character LoRA
    b = make("acting", 8, 16, LAYERS[:1])                     # like Aegis: fewer layers, no duration predictor
    res = server_lora.merge_loras(server_lora.MergeLoraRequest(
        name="mix1", sources=[{"name": "char_a", "weight": 1}, {"name": "acting", "weight": 1}], method="concat"))
    assert res.status_code == 200, res.body
    m, cfg = merged("mix1")
    for lay in LAYERS:
        want = delta(a, lay, 0.5 * 8 / 4) + (delta(b, lay, 0.5 * 16 / 8) if lay in LAYERS[:1] else 0)
        got = delta(m, lay)
        assert (got - want).abs().max() < 1e-4, (lay, float((got - want).abs().max()))
    assert cfg["r"] == 12 and cfg["lora_alpha"] == 12, cfg
    assert torch.equal(m[FULL], full_a), "the duration predictor must be the one source that has it"
    assert cfg["modules_to_save"] == ["duration_predictor"], cfg
    print("1. ランク違い・尺の予測は片方だけ: OK")

    full_c = torch.randn(8, 8)
    make("char_c", 4, 8, LAYERS, full_c)
    res = server_lora.merge_loras(server_lora.MergeLoraRequest(
        name="mix2", sources=[{"name": "char_a", "weight": 3}, {"name": "char_c", "weight": 1}], method="concat"))
    assert res.status_code == 200, res.body
    m, _ = merged("mix2")
    assert (m[FULL].double() - (0.75 * full_a.double() + 0.25 * full_c.double())).abs().max() < 1e-6
    print("2. 両方が尺の予測を持つ: 重みどおりの平均 OK")

    d = TMP / "loras" / "char_c" / "adapter"
    t = load_file(str(d / "adapter_model.safetensors"))
    t["base_model.model.duration_predictor.extra.weight"] = torch.randn(2, 2)
    save_file(t, str(d / "adapter_model.safetensors"))
    try:
        server_lora.merge_loras(server_lora.MergeLoraRequest(
            name="mix3", sources=[{"name": "char_a", "weight": 1}, {"name": "char_c", "weight": 1}], method="concat"))
    except HTTPException as e:
        assert e.status_code == 400, e
        print("3. 尺の予測の中身が違う: 400 で断る OK")
    else:
        raise AssertionError("merging different whole-saved modules must be refused")
    print("すべて通過")


if __name__ == "__main__":
    main()
