"""LoRA registry endpoints for the Electron desktop app.

Adapters trained externally (Emoji-TTS, CLI train.py) are imported here
and made available to the synthesize endpoint via `lora_name`.

Storage layout (under IRODORI_DATA_DIR, falls back to APP/ in dev):

    loras/
      <name>/
        adapter/                 # PEFT adapter dir
          adapter_config.json
          adapter_model.safetensors
        meta.json                # {name, base, imported_at, source, notes}
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Allow `from irodori_tts.lora import ...` when imported under APP/ cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from irodori_tts.lora import is_lora_adapter_dir  # noqa: E402

from data_paths import loras_dir, datasets_dir, voices_metadata_path  # noqa: E402

router = APIRouter()

ALLOWED_BASES = ("v4_1", "v4", "v3", "v3_voice_design", "v2", "voice_design")
META_NAME = "meta.json"
ADAPTER_SUBDIR = "adapter"
# Train タブの試聴が使い回す一時登録。生成 API は登録名しか受け取らないので
# 実体をレジストリに置くしかなく、一覧に出すと利用者の LoRA に紛れる。
# renderer.js の TR_PREVIEW_LORA と同じ値。
PREVIEW_LORA_NAME = "_ckpt_preview"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-\. 一-龥ぁ-んァ-ヶー]+$")

# Which of the two ingredients -- voice (who) and style (how) -- an adapter
# carries. Presets that never touch speaker_encoder cannot pin a speaker, so
# generation with one needs a reference wav to supply the voice.
_VOICE_PRESETS = (
    # UI の4プリセット（server_train.PRESETS のキー）のうち声を持つもの
    "speaker_style", "full", "broad",
    # 上流のプリセット名。外部で学習した LoRA を取り込むと meta.json に
    # こちらが入る。speaker_encoder に加えて diffusion 側にも触れるもの。
    "all_attn", "all_attn_mlp", "all_linear", "conditioning",
)

# speaker_encoder だけを対象とするプリセット。参照音声なしで生成すると
# runtime は ref_latent をゼロ・ref_mask を全 False にする
# （inference_runtime.py の no_ref 分岐）ため、speaker encoder に LoRA を
# 当てていても話者情報が伝わらない。diffusion 側にも焼き込まれていないので、
# 単体では声を出せない = サンプルボイスが要る。
_SPEAKER_ENCODER_ONLY_PRESETS = ("speaker_only", "speaker_attn_mlp")
_PRESET_RE = re.compile(r"preset=([A-Za-z0-9_]+)")


def _targets_speaker_encoder(adapter_dir: Path) -> Optional[bool]:
    """Whether the adapter actually adapts the speaker encoder.

    This is the ground truth for "does it carry a voice": an adapter that
    touches ``speaker_encoder`` reproduces the speaker on its own, one that
    does not needs a reference wav. Reading it from the adapter beats reading
    the preset out of ``notes`` — several adapters were trained before the
    preset was recorded there, and their notes say nothing about it.

    Returns None when the config cannot be read, so the caller can fall back.
    """
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.is_file():
        return None
    try:
        targets = json.loads(cfg.read_text(encoding="utf-8")).get("target_modules")
    except (json.JSONDecodeError, OSError):
        return None
    if targets is None:
        return None
    # PEFT stores either a regex string or a list of module names.
    # Preset aliases are expanded before the config is written, so what lands
    # here always names the modules — `all_attn_mlp` becomes a regex containing
    # `speaker_encoder.in_proj` and `speaker_encoder.blocks.\d+.`. Checked
    # against the 179 adapters stored on the dev machine: 85 name
    # speaker_encoder, none keep the alias (counts vary per machine; what
    # matters is the alias never survives). Matching the name is the whole test.
    text = targets if isinstance(targets, str) else " ".join(str(t) for t in targets)
    return "speaker_encoder" in text


# speaker_encoder 以外のモジュールを名指ししているか。生成側（diffusion の
# blocks / caption_encoder / テキスト backbone など）に触れていれば、話者は
# そちらにも焼き込まれている。直前が英数字や . のときは一致させない:
# `wk_text` の text や `speaker_encoder\.blocks\.` の blocks のような
# 部分文字列・別モジュール配下を誤って数えないため。後続は `.`（リスト形式）
# `\.`（正規表現形式）に加え `|` `)` も許す — `(speaker_encoder|blocks)\.`
# のようにグループ内でまとめられた形を取りこぼさないため。
_NON_SPEAKER_TARGET_RE = re.compile(
    r"(?<![\w.])"
    r"(blocks|text_encoder|caption_encoder|pretrained_text_backbone"
    # adaln の実モジュール名は attention_adaln / mlp_adaln で、素の adaln では
    # 直前の _ が lookbehind に弾かれて届かない。接頭辞ごと許す。
    # duration_predictor は現プリセットの target_modules に現れないが、
    # 外部アダプタが名指しする可能性に備えて残す。
    r"|(?:\w+_)?adaln|duration_predictor)"
    r"(?=[\\.|)])"
)


def _speaker_encoder_only(adapter_dir: Path) -> bool:
    """speaker encoder しか対象にしていないか。

    そういう adapter は参照音声がないと声を出せない。単体で声を持つものと
    区別する必要がある。
    """
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.is_file():
        return False
    try:
        targets = json.loads(cfg.read_text(encoding="utf-8")).get("target_modules")
    except (json.JSONDecodeError, OSError):
        return False
    if targets is None:
        return False
    text = targets if isinstance(targets, str) else " ".join(str(t) for t in targets)
    if "speaker_encoder" not in text:
        return False
    return not _NON_SPEAKER_TARGET_RE.search(text)


def _provides_from_meta(meta: dict, adapter_dir: Optional[Path] = None) -> list[str]:
    """Work out what an adapter supplies.

    Decided from the adapter itself where possible: whether it adapts the
    speaker encoder is what determines if it can voice a speaker unaided.
    The preset recorded in ``meta`` is only a fallback for adapters whose
    config cannot be read, and anything still unknown is treated as
    style-only — claiming a voice that is not there is what makes generation
    drift to a stranger.
    """
    explicit = meta.get("provides")
    if isinstance(explicit, list) and explicit:
        return [str(x) for x in explicit]

    if adapter_dir is not None:
        targets_speaker = _targets_speaker_encoder(adapter_dir)
        if targets_speaker is True:
            # speaker encoder しか触っていないものは、参照音声を通してしか
            # 声を出せない（_SPEAKER_ENCODER_ONLY_PRESETS の注記参照）。
            return ["style"] if _speaker_encoder_only(adapter_dir) else ["voice", "style"]
        if targets_speaker is False:
            return ["style"]

    preset = str(meta.get("preset") or "")
    if not preset:
        found = _PRESET_RE.search(str(meta.get("notes") or ""))
        preset = found.group(1) if found else ""

    if preset in _SPEAKER_ENCODER_ONLY_PRESETS:
        # 声は持つが、参照音声を通してしか出せない。単体では使えないので
        # voice は名乗らせない（名乗ると UI がサンプル不要と判断する）。
        return ["style"]
    if preset in _VOICE_PRESETS:
        return ["voice", "style"]
    return ["style"]


def _normalize_key(name: str) -> str:
    """Strip the -Irodori / -Irodori2 suffix used across the registry."""
    stem = re.sub(r"[-ー]Irodori\d*$", "", name, flags=re.IGNORECASE)
    return stem.strip().lower()


def _voice_sources() -> tuple[dict[str, str], dict[str, str]]:
    """Map normalized character name -> registered voice id / dataset name."""
    voices: dict[str, str] = {}
    meta_path = voices_metadata_path()
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            for entry in data.get("voices", []):
                voices[_normalize_key(entry.get("name", ""))] = entry.get("id", "")
        except (json.JSONDecodeError, OSError):
            pass

    datasets: dict[str, str] = {}
    root = datasets_dir()
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and not child.name.startswith("_"):
                datasets[_normalize_key(child.name)] = child.name
    return voices, datasets


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > 64:
        raise HTTPException(400, "name must be 1-64 characters")
    if not NAME_PATTERN.match(name):
        raise HTTPException(400, "name contains invalid characters")
    return name


def _load_meta(lora_dir: Path) -> dict:
    meta_path = lora_dir / META_NAME
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("name", lora_dir.name)
                return data
        except json.JSONDecodeError:
            pass
    return {
        "name": lora_dir.name,
        "base": "unknown",
        "imported_at": None,
        "source": "",
        "notes": "(no meta.json found; manually placed)",
    }


@router.get("/api/v1/loras")
def list_loras() -> JSONResponse:
    root = loras_dir()
    voices, datasets = _voice_sources()
    items = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        # 試聴の一時登録だけを隠す。消さないのは、試聴のあと Synthesize で
        # 確かめ直せるようにするため。
        # 先頭 _ をまとめて弾くと、利用者が _ab_official のように名付けた
        # 検証用まで一覧・削除・マージから消えて UI から復帰できなくなる。
        if child.name == PREVIEW_LORA_NAME:
            continue
        adapter_dir = child / ADAPTER_SUBDIR
        if not is_lora_adapter_dir(adapter_dir):
            continue
        meta = _load_meta(child)
        provides = _provides_from_meta(meta, adapter_dir)
        meta["provides"] = provides

        # A style-only adapter needs a voice from somewhere: a registered
        # sample, or a dataset we can cut one from. Without either it can
        # still be generated with, but the speaker is unconstrained and
        # drifts per seed -- so the UI offers it as unusable rather than
        # letting it produce a stranger's voice.
        key = _normalize_key(meta.get("name", child.name))
        voice_id = voices.get(key)
        dataset = datasets.get(key)
        if "voice" in provides:
            meta["voice_source"] = "lora"
        elif voice_id:
            meta["voice_source"] = f"voice:{voice_id}"
        elif dataset:
            meta["voice_source"] = f"dataset:{dataset}"
        else:
            meta["voice_source"] = None
        meta["usable"] = meta["voice_source"] is not None
        # 1 体あたり数百 MB になる。何体あるかだけでは見当が付かないので、
        # 占有量も返す。合計は UI の状態カードが出す。
        meta["bytes"] = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
        items.append(meta)
    return JSONResponse(
        content={"loras": items, "total_bytes": sum(m.get("bytes", 0) for m in items)}
    )


class ImportLoraRequest(BaseModel):
    name: str = Field(..., description="Display name; must be unique")
    base: str = Field(..., description="One of v4, v3, v2, voice_design")
    source_path: str = Field(..., description="Absolute path to a PEFT adapter dir")
    notes: str = Field("", description="Free text notes")


@router.post("/api/v1/loras/import")
def import_lora(req: ImportLoraRequest) -> JSONResponse:
    name = _validate_name(req.name)
    if req.base not in ALLOWED_BASES:
        raise HTTPException(400, f"base must be one of {list(ALLOWED_BASES)}")

    src = Path(req.source_path).expanduser()
    try:
        src = src.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise HTTPException(400, f"source_path does not exist: {req.source_path}")
    if not src.is_dir():
        raise HTTPException(400, f"source_path is not a directory: {src}")
    if not is_lora_adapter_dir(src):
        raise HTTPException(
            400,
            "source_path must contain adapter_config.json and adapter weights "
            "(adapter_model.safetensors or adapter_model.bin)",
        )

    dst = loras_dir() / name
    if dst.exists():
        raise HTTPException(409, f"LoRA '{name}' already exists; delete first")

    adapter_dst = dst / ADAPTER_SUBDIR
    adapter_dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(src, adapter_dst)
    except OSError as exc:
        # Best-effort cleanup so a half-imported dir doesn't pollute the registry.
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        raise HTTPException(500, f"failed to copy adapter: {exc}") from exc

    meta = {
        "name": name,
        "base": req.base,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "notes": req.notes,
    }
    (dst / META_NAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return JSONResponse(content={"status": "ok", "name": name, "base": req.base})


@router.delete("/api/v1/loras/{name}")
def delete_lora(name: str) -> JSONResponse:
    name = _validate_name(name)
    target = loras_dir() / name
    if not target.exists():
        raise HTTPException(404, f"LoRA '{name}' not found")
    shutil.rmtree(target)
    return JSONResponse(content={"status": "ok", "name": name})


class MergeLoraRequest(BaseModel):
    name: str = Field(..., description="New merged LoRA name")
    sources: list[dict] = Field(..., description="[{name, weight}, ...] 2〜3個")
    method: str = Field("linear", description="linear or slerp (2つのみ)")
    notes: str = Field("", description="Free text notes")


@router.post("/api/v1/loras/merge")
def merge_loras(req: MergeLoraRequest) -> JSONResponse:
    import torch
    from safetensors.torch import load_file, save_file

    name = _validate_name(req.name)
    if loras_dir() / name in [loras_dir() / s["name"] for s in req.sources]:
        raise HTTPException(400, "Output name must differ from source names")
    if (loras_dir() / name).exists():
        raise HTTPException(409, f"LoRA '{name}' already exists; delete first")
    if not (2 <= len(req.sources) <= 3):
        raise HTTPException(400, "sources must be 2 or 3")
    if req.method == "slerp" and len(req.sources) != 2:
        raise HTTPException(400, "slerp requires exactly 2 sources")

    # ソース LoRA を読み込む
    src_data = []
    bases = []
    total_w = sum(float(s.get("weight", 1)) for s in req.sources)
    if total_w <= 0:
        raise HTTPException(400, "Total weight must be > 0")

    for s in req.sources:
        src_name = _validate_name(s["name"])
        src_dir = loras_dir() / src_name
        if not src_dir.exists():
            raise HTTPException(404, f"LoRA '{src_name}' not found")
        adapter_path = src_dir / ADAPTER_SUBDIR / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise HTTPException(400, f"LoRA '{src_name}' has no safetensors")
        meta = _load_meta(src_dir)
        bases.append(meta.get("base", "unknown"))
        w = float(s.get("weight", 1)) / total_w  # 正規化
        tensors = load_file(str(adapter_path))
        cfg_path = src_dir / ADAPTER_SUBDIR / "adapter_config.json"
        try:
            adapter_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        except (json.JSONDecodeError, OSError):
            adapter_cfg = {}
        src_data.append({"name": src_name, "w": w, "tensors": tensors, "meta": meta,
                         "cfg": adapter_cfg})

    if len(set(bases)) > 1:
        raise HTTPException(400, f"All sources must share the same base model. Got: {set(bases)}")

    # --- ソース間の整合を先に検証する ---
    # キー集合: 積集合で黙って落とすと、adapter_config が要求する全量重みの
    # 欠けたアダプタができて、ロード時 KeyError が再発する。
    ref = src_data[0]
    ref_keys = set(ref["tensors"].keys())
    for sd in src_data[1:]:
        if set(sd["tensors"].keys()) != ref_keys:
            diff = sorted(set(sd["tensors"].keys()) ^ ref_keys)
            raise HTTPException(
                400,
                f"Source adapters have different tensor sets "
                f"('{sd['name']}' vs '{ref['name']}', e.g. {diff[:3]})",
            )
        # 形状: 検証せず加算に入ると内部500として返ってしまう。
        for key in ref_keys:
            if sd["tensors"][key].shape != ref["tensors"][key].shape:
                raise HTTPException(
                    400,
                    f"Tensor shape mismatch for '{key}': "
                    f"{tuple(ref['tensors'][key].shape)} vs {tuple(sd['tensors'][key].shape)}",
                )
    # LoRA スケール: PEFT の実効差分は (lora_alpha / r) × B@A なので、
    # r や alpha が違うソースを混ぜると指定した比率にならない。出力には
    # 先頭ソースの config をコピーするため、全ソース一致を要求する。
    _SCALE_FIELDS = ("r", "lora_alpha", "use_rslora", "rank_pattern", "alpha_pattern")
    ref_scale = {k: ref["cfg"].get(k) for k in _SCALE_FIELDS}
    for sd in src_data[1:]:
        scale = {k: sd["cfg"].get(k) for k in _SCALE_FIELDS}
        if scale != ref_scale:
            raise HTTPException(
                400,
                f"LoRA scaling config differs between '{ref['name']}' and '{sd['name']}' "
                f"({ref_scale} vs {scale}); merging would not honor the requested weights",
            )

    common_keys = ref_keys

    merged: dict[str, torch.Tensor] = {}

    # アダプタには LoRA 対のほかに modules_to_save の全量重みが入っている
    # （v4 は duration_predictor の31本）。これを落とすと adapter_config が
    # 宣言するキーが実体に無く、ロードが KeyError で死ぬ。全量重みは
    # 重み付き平均でブレンドする（slerp でも係数は w に一致させる）。
    for key in common_keys:
        if key.endswith(("lora_A.weight", "lora_B.weight")):
            continue
        first = src_data[0]["tensors"][key]
        if not torch.is_floating_point(first):
            # int / bool の平均は意味を持たない（切り捨て・True 潰れ）。
            # 最大重みのソースからそのまま持ってくる。
            dominant = max(src_data, key=lambda sd: sd["w"])
            merged[key] = dominant["tensors"][key]
            continue
        # float64 ソースを float32 で計算すると精度が落ちるので、
        # 計算は float64 で行い元の dtype へ戻す（31本程度なのでコストは無視できる）。
        acc = None
        for sd in src_data:
            v = sd["w"] * sd["tensors"][key].double()
            acc = v if acc is None else acc + v
        merged[key] = acc.to(first.dtype)

    if req.method == "slerp":
        # 2つのLoRAをSLERP（球面線形補間）でブレンド
        # 各レイヤーのΔW=B@Aを計算してSLERPし、SVDでA_new/B_newに分解
        t = src_data[1]["w"]  # src[0]からsrc[1]へのブレンド係数
        for key in common_keys:
            if key.endswith("lora_A.weight"):
                layer = key[:-len("lora_A.weight")]
                b_key = layer + "lora_B.weight"
                if b_key not in common_keys:
                    continue
                A0 = src_data[0]["tensors"][key].float()
                B0 = src_data[0]["tensors"][b_key].float()
                A1 = src_data[1]["tensors"][key].float()
                B1 = src_data[1]["tensors"][b_key].float()
                dW0 = B0 @ A0  # (out, in)
                dW1 = B1 @ A1
                # SLERP
                v0 = dW0.flatten()
                v1 = dW1.flatten()
                n0, n1 = v0.norm(), v1.norm()
                if n0 < 1e-8 or n1 < 1e-8:
                    dW = (1 - t) * dW0 + t * dW1
                else:
                    cos_theta = (v0 / n0).dot(v1 / n1).clamp(-1, 1)
                    theta = torch.acos(cos_theta)
                    if theta.abs() < 1e-6:
                        dW = (1 - t) * dW0 + t * dW1
                    else:
                        dW = (torch.sin((1 - t) * theta) / torch.sin(theta)) * dW0 + \
                             (torch.sin(t * theta) / torch.sin(theta)) * dW1
                # SVDで rank=A0.shape[0] に分解
                rank = A0.shape[0]
                U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
                U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
                B_new = U_r * S_r.sqrt().unsqueeze(0)
                A_new = Vh_r * S_r.sqrt().unsqueeze(1)
                merged[key] = A_new.to(src_data[0]["tensors"][key].dtype)
                merged[b_key] = B_new.to(src_data[0]["tensors"][b_key].dtype)
    else:
        # Linear: 重み付き平均 ΔW=Σ(w_i * B_i @ A_i) → SVD分解
        for key in common_keys:
            if key.endswith("lora_A.weight"):
                layer = key[:-len("lora_A.weight")]
                b_key = layer + "lora_B.weight"
                if b_key not in common_keys:
                    continue
                dW = None
                rank = src_data[0]["tensors"][key].shape[0]
                for sd in src_data:
                    A = sd["tensors"][key].float()
                    B = sd["tensors"][b_key].float()
                    d = sd["w"] * (B @ A)
                    dW = d if dW is None else dW + d
                U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
                U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
                B_new = U_r * S_r.sqrt().unsqueeze(0)
                A_new = Vh_r * S_r.sqrt().unsqueeze(1)
                merged[key] = A_new.to(src_data[0]["tensors"][key].dtype)
                merged[b_key] = B_new.to(src_data[0]["tensors"][b_key].dtype)

    # 保存
    dst = loras_dir() / name
    adapter_dst = dst / ADAPTER_SUBDIR
    try:
        adapter_dst.mkdir(parents=True)

        # adapter_config.json は最初のソースからコピーして name を更新
        src_cfg_path = loras_dir() / src_data[0]["name"] / ADAPTER_SUBDIR / "adapter_config.json"
        if src_cfg_path.exists():
            cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))
            (adapter_dst / "adapter_config.json").write_text(
                json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # torch.linalg.svd の U/Vh は非連続で、スライス・乗算を経ても
        # 解消されない。safetensors は非連続テンソルの保存を拒否するため
        # （実測で 466/466 テンソルが該当し、全マージが 500 になっていた）、
        # 保存直前に一括で連続化する。分岐ごとに付けるより漏れに強い。
        save_file(
            {k: v.contiguous() for k, v in merged.items()},
            str(adapter_dst / "adapter_model.safetensors"),
        )

        source_info = ", ".join(f"{s['name']}×{s['w']:.2f}" for s in src_data)
        meta = {
            "name": name,
            "base": bases[0],
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source": f"merged({req.method}): {source_info}",
            "notes": req.notes or f"Merged via {req.method}: {source_info}",
        }
        (dst / META_NAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # 途中で失敗すると adapter_config.json だけの残骸ができ、一覧には
        # 出ないのに同名の再実行が 409 で弾かれ続ける。import_lora と同じく
        # 出力先ごと消してから例外を伝える。
        shutil.rmtree(dst, ignore_errors=True)
        raise

    # UI は「Nレイヤーをマージ」と表示する。全量重み31本を足した今、
    # len(merged)//2 はどの実数にも一致しないので、LoRA 対の数を数えて返す。
    lora_pairs = sum(1 for k in merged if k.endswith("lora_A.weight"))
    return JSONResponse(content={"status": "ok", "name": name, "base": bases[0], "keys_merged": lora_pairs})


def lora_base_of(name: Optional[str]) -> Optional[str]:
    """Which base model a registered LoRA was trained on, or None if unknown.

    OpenAI 互換の口は voice に LoRA 名しか受け取らないので、どのベースで
    鳴らすかはここから決める。取り違えると v3 の adapter を v4 に載せて
    KeyError で落ちる。

    生成の口からも「学習したベースと違うモデルに当てていないか」の照合に
    使う。LoRA 未指定で呼ばれることがあるので None と空文字を許す。
    """
    if not name or not str(name).strip():
        return None
    meta = _load_meta(loras_dir() / str(name).strip())
    base = meta.get("base")
    return str(base) if base in ALLOWED_BASES else None


def lora_has_voice(name: str) -> bool:
    """True when the adapter can pin a speaker without a reference wav."""
    d = loras_dir() / str(name).strip()
    return "voice" in _provides_from_meta(_load_meta(d), d / ADAPTER_SUBDIR)


def resolve_lora_adapter_path(name: Optional[str]) -> Optional[str]:
    """Used by /api/v1/synthesize to translate a registry name into a path.

    Returns None when name is empty or null-ish. Raises ValueError when the
    name is set but missing from the registry.
    """
    if name is None:
        return None
    raw = str(name).strip()
    if not raw or raw.lower() in ("none", "null", "off", "base"):
        return None
    target = loras_dir() / raw / ADAPTER_SUBDIR
    if not target.is_dir():
        raise ValueError(f"LoRA '{raw}' not found in registry")
    return str(target.resolve())


