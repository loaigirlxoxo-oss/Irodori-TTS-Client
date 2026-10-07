"""取得するかを選べる音声モデルの一覧。初回セットアップと設定画面の表は、ここだけを見る。

容量は Hugging Face の公開情報。VRAM は RTX 5080 での実測で、一番重い使い方
（30秒の文・参照音声30秒・透かしまで）の山に CUDA の初期化分（約0.35GiB）を足し、
0.5 刻みで切り上げた値。bf16 は RTX 30 系以降で自動で選ばれる精度、fp32 はそれより
前の世代で選ばれる精度。量子化版と Large は bf16 でしか動かないので fp32 は None。

どのモデルを選んでも要る部品（コーデック・トークナイザ・書き起こし・透かし・
話者照合）は COMMON で、選ばせずに必ず取る。
"""
from __future__ import annotations

_Q = "Aratako/Irodori-TTS-v4-Large-Quantized"

# need: 動く GPU の条件。None は制限なし。
#   "bf16"  : bf16 が速い世代（RTX 30 系以降）
#   "ampere": RTX 30 系以降（compute capability 8.0 以上）
#   "ada"   : RTX 40 系以降（compute capability 8.9 以上）
MODELS: list[dict] = [
    {"id": "v4_1", "label": "v4.1-Small", "note": "推奨", "group": "おすすめ", "default": True,
     "files": [("Aratako/Irodori-TTS-v4.1-Small", "model.safetensors")],
     "size_gb": 3.1, "vram_bf16": 4.0, "vram_fp32": 7.5, "need": None},
    {"id": "v4_large_int8", "label": "v4-Large 軽量 int8", "note": "軽量・RTX 30 系以降必須", "group": "おすすめ", "default": True,
     "files": [(_Q, "int8-weight-only/model.safetensors"), (_Q, "tokenizer/tokenizer.json"),
               (_Q, "tokenizer/tokenizer_config.json")],
     "size_gb": 3.7, "vram_bf16": 6.0, "vram_fp32": None, "need": "bf16"},
    {"id": "v3_voice_design", "label": "v3 VoiceDesign", "note": "説明文から声を作る", "group": "おすすめ", "default": True,
     "files": [("Aratako/Irodori-TTS-600M-v3-VoiceDesign", "model.safetensors")],
     "size_gb": 2.5, "vram_bf16": 3.5, "vram_fp32": 7.0, "need": None},
    {"id": "v4_1_anime", "label": "v4.1-Anime", "note": "アニメ調・非公式", "group": "そのほか", "default": False,
     "files": [("phasefield-audio/Irodori-TTS-v4.1-Anime", "model.safetensors")],
     "size_gb": 3.1, "vram_bf16": 4.0, "vram_fp32": 7.5, "need": None},
    {"id": "v4_large", "label": "v4-Large", "note": "高品質・重い", "group": "そのほか", "default": False,
     "files": [("Aratako/Irodori-TTS-v4-Large", "model.safetensors"),
               ("Aratako/Irodori-TTS-v4-Large", "tokenizer/tokenizer.json"),
               ("Aratako/Irodori-TTS-v4-Large", "tokenizer/tokenizer_config.json")],
     "size_gb": 13.2, "vram_bf16": 8.5, "vram_fp32": None, "need": "bf16"},
    {"id": "v4_large_int4", "label": "v4-Large 軽量 int4", "note": "最軽量・RTX 30 系以降必須", "group": "そのほか", "default": False,
     "files": [(_Q, "int4-weight-only/model.safetensors"), (_Q, "tokenizer/tokenizer.json"),
               (_Q, "tokenizer/tokenizer_config.json")],
     "size_gb": 2.8, "vram_bf16": 5.5, "vram_fp32": None, "need": "ampere"},
    {"id": "v4_large_fp8", "label": "v4-Large 軽量 float8", "note": "軽量・RTX 40 系以降必須", "group": "そのほか", "default": False,
     "files": [(_Q, "float8-weight-only/model.safetensors"), (_Q, "tokenizer/tokenizer.json"),
               (_Q, "tokenizer/tokenizer_config.json")],
     "size_gb": 3.7, "vram_bf16": 6.0, "vram_fp32": None, "need": "ada"},
    {"id": "v4", "label": "v4-Small", "note": "v4 で作った LoRA 用", "group": "そのほか", "default": False,
     "files": [("Aratako/Irodori-TTS-v4-Small", "model.safetensors")],
     "size_gb": 3.1, "vram_bf16": 4.0, "vram_fp32": 7.5, "need": None},
    {"id": "v3", "label": "v3", "note": "旧版", "group": "旧版", "default": False,
     "files": [("Aratako/Irodori-TTS-500M-v3", "model.safetensors")],
     "size_gb": 2.1, "vram_bf16": 3.5, "vram_fp32": 6.5, "need": None},
    {"id": "v2", "label": "v2", "note": "旧版", "group": "旧版", "default": False,
     "files": [("Aratako/Irodori-TTS-500M-v2", "model.safetensors")],
     "size_gb": 2.0, "vram_bf16": 3.5, "vram_fp32": 6.5, "need": None},
    {"id": "voice_design", "label": "v2 VoiceDesign", "note": "旧版", "group": "旧版", "default": False,
     "files": [("Aratako/Irodori-TTS-500M-v2-VoiceDesign", "model.safetensors")],
     "size_gb": 2.1, "vram_bf16": 3.5, "vram_fp32": 6.5, "need": None},
]
BY_ID = {m["id"]: m for m in MODELS}

# 以前の版が全員に取らせていたもの。setup.bat（開発版）が引数なしで動くときはこれを取る。
LEGACY_IDS = ("v2", "v3", "v4", "v4_1", "v4_large", "voice_design", "v3_voice_design")

# どのモデルでも要る部品の容量（概算）。表の合計に足す。
COMMON_GB = 6.0


def default_ids() -> list[str]:
    return [m["id"] for m in MODELS if m["default"]]


def capability_ok(need: str | None, gpu: dict) -> bool:
    """gpu は server.gpu_info() と同じ形（device / rocm / capability / bf16_fast）。"""
    if need is None:
        return True
    if gpu.get("device") != "cuda" or gpu.get("rocm") or not gpu.get("capability"):
        return False
    cap = tuple(int(x) for x in str(gpu["capability"]).split("."))
    if need == "bf16":
        return bool(gpu.get("bf16_fast"))
    if need == "ampere":
        return cap >= (8, 0)
    if need == "ada":
        return cap >= (8, 9)
    return False


def is_downloaded(model_id: str) -> bool:
    """必要なファイルが全部キャッシュにあるか。ネットワークには行かない。"""
    from huggingface_hub import hf_hub_download

    from data_paths import manual_checkpoint

    for repo, filename in BY_ID[model_id]["files"]:
        # 手で置いた重み（models/manual/<リポジトリ名>/）も取得済みに数える。
        if filename == "model.safetensors" and manual_checkpoint(repo) is not None:
            continue
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)
        except Exception:  # noqa: BLE001 - 無いのと同じに扱う
            return False
    return True


def cached_path(repo: str, filename: str):
    """キャッシュにあるファイルの場所。無ければ None。"""
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=repo, filename=filename, local_files_only=True))
    except Exception:  # noqa: BLE001
        return None


def gpu_summary() -> dict:
    from gpu_probe import gpu_info, vram_gb

    info = dict(gpu_info())
    info["vram_gb"] = vram_gb()
    info["precision"] = "bf16" if info.get("bf16_fast") else "fp32"
    return info


def catalog_json() -> dict:
    """設定画面と初回セットアップの表に出す中身。"""
    gpu = gpu_summary()
    rows = []
    for m in MODELS:
        runnable = capability_ok(m["need"], gpu)
        vram = m["vram_bf16"] if gpu["precision"] == "bf16" else m["vram_fp32"]
        rows.append({"id": m["id"], "label": m["label"], "note": m["note"], "group": m["group"],
                     "size_gb": m["size_gb"], "vram_gb": vram if runnable else None,
                     "runnable": runnable, "default": m["default"],
                     "downloaded": is_downloaded(m["id"])})
    return {"gpu": gpu, "models": rows, "common_gb": COMMON_GB}


if __name__ == "__main__":
    # 初回セットアップ（runtime.js）が、取得前にこの一覧を読む。
    import json

    print(json.dumps(catalog_json(), ensure_ascii=False))
