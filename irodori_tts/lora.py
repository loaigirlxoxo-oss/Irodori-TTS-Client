from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .config import TrainConfig
from .model import TextToLatentRFDiT

LORA_TRAIN_CONFIG_FIELDS = (
    "lora_enabled",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_bias",
    "lora_target_modules",
    "lora_modules_to_save",
)

LORA_ADAPTER_CONFIG_NAME = "adapter_config.json"
LORA_ADAPTER_STATE_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
LORA_TRAINER_STATE_NAME = "trainer_state.pt"
LORA_METADATA_NAME = "irodori_lora_metadata.json"

# Explicit pretrained-backbone support for the published model families. Keep these
# architecture-specific rather than guessing attention/MLP layers from arbitrary HF models.
_MODERNBERT_BACKBONE_ATTN_TARGET = (
    r"pretrained_text_backbone\.backbone\.layers\.\d+\.attn\.(Wqkv|Wo)"
)
_MODERNBERT_BACKBONE_MLP_TARGET = r"pretrained_text_backbone\.backbone\.layers\.\d+\.mlp\.(Wi|Wo)"
_T5GEMMA2_BACKBONE_ATTN_TARGET = (
    r"pretrained_text_backbone\.backbone\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)"
)
_T5GEMMA2_BACKBONE_MLP_TARGET = (
    r"pretrained_text_backbone\.backbone\.layers\.\d+\.mlp\."
    r"(gate_proj|up_proj|down_proj)"
)
_PRETRAINED_BACKBONE_ATTN_TARGET = (
    rf"({_MODERNBERT_BACKBONE_ATTN_TARGET}|{_T5GEMMA2_BACKBONE_ATTN_TARGET})"
)
_PRETRAINED_BACKBONE_ATTN_MLP_TARGET = (
    rf"({_PRETRAINED_BACKBONE_ATTN_TARGET}"
    rf"|{_MODERNBERT_BACKBONE_MLP_TARGET}|{_T5GEMMA2_BACKBONE_MLP_TARGET})"
)

LORA_TARGET_PRESETS: dict[str, str] = {
    "text_attn_mlp": (
        r"^text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))$"
    ),
    "caption_attn_mlp": (
        r"^caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))$"
    ),
    "speaker_attn_mlp": (
        r"^(speaker_encoder\.in_proj"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3)))$"
    ),
    "diffusion_attn": (
        r"^blocks\.\d+\.attention\."
        r"(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)$"
    ),
    "diffusion_attn_mlp": (
        r"^blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3))$"
    ),
    "pretrained_backbone_attn": rf"^{_PRETRAINED_BACKBONE_ATTN_TARGET}$",
    "pretrained_backbone_attn_mlp": rf"^{_PRETRAINED_BACKBONE_ATTN_MLP_TARGET}$",
    "all_attn": (
        r"^(text_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|caption_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|speaker_encoder\.blocks\.\d+\.attention\.(wq|wk|wv|wo|gate)"
        r"|blocks\.\d+\.attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        rf"|{_PRETRAINED_BACKBONE_ATTN_TARGET})$"
    ),
    "diffusion_full": (
        r"^(cond_module\.(0|2|4)"
        r"|in_proj"
        r"|out_proj"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3)"
        r"|attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)))$"
    ),
    "adaln": (
        r"^blocks\.\d+\."
        r"(attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up))$"
    ),
    "conditioning": (
        r"^(cond_module\.(0|2|4)"
        r"|speaker_encoder\.in_proj"
        r"|blocks\.\d+\.attention\.(wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption))$"
    ),
    "all_attn_mlp": (
        r"^(text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|speaker_encoder\.in_proj"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate)"
        r"|mlp\.(w1|w2|w3))"
        rf"|{_PRETRAINED_BACKBONE_ATTN_MLP_TARGET})$"
    ),
    "all_linear": (
        r"^(speaker_encoder\.in_proj"
        r"|cond_module\.(0|2|4)"
        r"|in_proj"
        r"|out_proj"
        r"|text_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|caption_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|speaker_encoder\.blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wo|gate)|mlp\.(w1|w2|w3))"
        r"|blocks\.\d+\."
        r"(attention\.(wq|wk|wv|wk_text|wv_text|wk_speaker|wv_speaker|wk_caption|wv_caption|gate|wo)"
        r"|mlp\.(w1|w2|w3)"
        r"|attention_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up)"
        r"|mlp_adaln\.(shift_down|scale_down|gate_down|shift_up|scale_up|gate_up))"
        rf"|{_PRETRAINED_BACKBONE_ATTN_MLP_TARGET})$"
    ),
}


def _require_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA fine-tuning requires `peft`. Install with `pip install peft` or `uv sync`."
        ) from exc
    return LoraConfig, PeftModel, get_peft_model


def _lookup_config_value(raw: TrainConfig | Mapping[str, Any] | None, field: str) -> Any:
    if raw is None:
        return getattr(TrainConfig(), field)
    if isinstance(raw, TrainConfig):
        return getattr(raw, field)
    if isinstance(raw, Mapping):
        if field in raw:
            return raw[field]
        return getattr(TrainConfig(), field)
    raise TypeError(f"Unsupported LoRA config source: {type(raw)!r}")


def train_config_uses_lora(raw: TrainConfig | Mapping[str, Any] | None) -> bool:
    return bool(_lookup_config_value(raw, "lora_enabled"))


def checkpoint_state_uses_lora(model_state: Mapping[str, torch.Tensor]) -> bool:
    return any(key.startswith("base_model.model.") or ".lora_" in key for key in model_state)


def resolve_lora_target_modules(spec: str | Sequence[str] | None) -> str | list[str]:
    if spec is None:
        spec = TrainConfig().lora_target_modules

    if isinstance(spec, str):
        value = spec.strip()
        if not value:
            raise ValueError("lora_target_modules must not be empty.")
        preset = LORA_TARGET_PRESETS.get(value)
        if preset is not None:
            return preset
        if "," in value:
            modules = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
            if not modules:
                raise ValueError(f"Invalid LoRA target_modules list: {spec!r}")
            return modules
        return value

    modules = [str(item).strip() for item in spec if str(item).strip()]
    if not modules:
        raise ValueError("LoRA target_modules sequence must not be empty.")
    return modules


def resolve_lora_modules_to_save(
    spec: str | Sequence[str] | None,
    *,
    use_duration_predictor: bool,
) -> list[str] | None:
    if spec is None:
        return None

    if isinstance(spec, str):
        value = spec.strip()
        if not value or value.lower() == "none":
            return None
        if value.lower() == "auto":
            if use_duration_predictor:
                return ["duration_predictor"]
            return None
        modules = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    else:
        modules = [str(item).strip() for item in spec if str(item).strip()]

    if not modules:
        return None
    return modules


def build_lora_config_kwargs(
    raw: TrainConfig | Mapping[str, Any],
    *,
    use_duration_predictor: bool = False,
) -> dict[str, Any]:
    bias = str(_lookup_config_value(raw, "lora_bias")).strip().lower()
    if bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"Unsupported lora_bias={bias!r}. Expected one of: none, all, lora_only.")

    kwargs = {
        "r": int(_lookup_config_value(raw, "lora_r")),
        "lora_alpha": int(_lookup_config_value(raw, "lora_alpha")),
        "lora_dropout": float(_lookup_config_value(raw, "lora_dropout")),
        "bias": bias,
        "target_modules": resolve_lora_target_modules(
            _lookup_config_value(raw, "lora_target_modules")
        ),
    }
    modules_to_save = resolve_lora_modules_to_save(
        _lookup_config_value(raw, "lora_modules_to_save"),
        use_duration_predictor=use_duration_predictor,
    )
    if modules_to_save is not None:
        kwargs["modules_to_save"] = modules_to_save
    return kwargs


def apply_lora(
    model: TextToLatentRFDiT,
    raw: TrainConfig | Mapping[str, Any],
) -> torch.nn.Module:
    if not train_config_uses_lora(raw):
        return model

    lora_config_cls, _, get_peft_model = _require_peft()
    peft_model = get_peft_model(
        model,
        lora_config_cls(
            task_type=None,
            inference_mode=False,
            **build_lora_config_kwargs(
                raw,
                use_duration_predictor=bool(model.cfg.use_duration_predictor),
            ),
        ),
    )
    return peft_model


def is_lora_adapter_dir(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    if not (candidate / LORA_ADAPTER_CONFIG_NAME).is_file():
        return False
    return any((candidate / name).is_file() for name in LORA_ADAPTER_STATE_NAMES)


def load_lora_adapter(
    model: TextToLatentRFDiT,
    adapter_path: str | Path,
    *,
    is_trainable: bool,
    adapter_name: str = "default",
    torch_device: str | None = None,
) -> torch.nn.Module:
    _, peft_model_cls, _ = _require_peft()
    if isinstance(model, peft_model_cls):
        if adapter_name not in model.peft_config:
            model.load_adapter(
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                torch_device=torch_device,
            )
        model.set_adapter(adapter_name)
        return model
    return peft_model_cls.from_pretrained(
        model,
        str(adapter_path),
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        torch_device=torch_device,
    )


def set_lora_scale(model: torch.nn.Module, adapter_name: str, scale: float, *, fresh: bool = False) -> int:
    """Multiply the LoRA delta of one adapter by `scale`. Returns how many layers were touched.

    PEFT adds `scaling[adapter] * B(A(x))` per layer, where scaling is alpha / r (or alpha / sqrt(r)
    with rsLoRA). The value the adapter was loaded with is remembered on first use, so calling this
    again always scales from the original, never from the last request. Call it with 1.0 to restore.
    Modules saved whole (modules_to_save, e.g. duration_predictor) have no scaling and stay as loaded.

    fresh=True marks "this adapter was just loaded": the remembered original scaling is dropped and
    taken again from what PEFT set. Adapter names come from the adapter's folder, so a LoRA rebuilt
    in the same folder gets the same name; without dropping, the previous LoRA's scaling (e.g. r=16 /
    alpha=32 -> 2.0) would be applied to the new one (r=80 / alpha=80 -> 1.0).
    """
    touched = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict) or adapter_name not in scaling:
            continue
        base = module.__dict__.setdefault("_irodori_base_scaling", {})
        if fresh:
            base.pop(adapter_name, None)
        base.setdefault(adapter_name, float(scaling[adapter_name]))
        scaling[adapter_name] = base[adapter_name] * float(scale)
        touched += 1
    return touched


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    total = sum(int(param.numel()) for param in model.parameters())
    return trainable, total
