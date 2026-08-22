import json
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import os

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import uvicorn
from typing import Optional
from datetime import datetime
import traceback

# Add parent directory to path so we can import irodori_tts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
    save_wav,
    default_runtime_device,
    list_available_runtime_devices,
)
from huggingface_hub import hf_hub_download

import server_lora  # LoRA registry endpoints (list/import/delete)
import server_audio  # Audio split + transcribe endpoints


def listen_port() -> int:
    """Port to bind to.

    main.js picks a free port and passes it via IRODORI_PORT so two copies of
    the app (or an unrelated service already on the default) never collide.
    Falls back to 8080 when the server is started by hand.
    """
    raw = os.environ.get("IRODORI_PORT", "").strip()
    if raw.isdigit():
        port = int(raw)
        if 1 <= port <= 65535:
            return port
    return 8080
import server_dataset  # Dataset CRUD endpoints
import server_train  # LoRA training job management
from data_paths import outputs_dir, voices_metadata_path, migrate_legacy_voices

# Migrate APP/references/ + APP/metadata.json into the new voices/ layout
# if needed. Idempotent; runs once per process at import time.
migrate_legacy_voices()

# Any LoRA job left mid-run by a previous process is an orphan — recover it
# so the UI does not treat a dead job as perpetually active.
server_train.recover_orphan_jobs()

app = FastAPI(title="Irodori-TTS API")
app.include_router(server_lora.router)
app.include_router(server_audio.router)
app.include_router(server_dataset.router)
app.include_router(server_train.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model registry. 上流 main のコードは v2/v3/VoiceDesign のチェックポイントと
# 後方互換なので、v4 も含めて同じランタイムで in-process に読める。
#
# v4.1 は v4-Small の duration predictor だけを差し替えたもので、本体
# （RF-DiT・text/caption encoder・speaker encoder）は同じ。上流は v4.1 を
# 推奨としている。尺の過大推定による生成崩れが減る（Kana-CER 7.43%→7.29%）。
# v4-Small も残すのは、既存の LoRA を作った条件で鳴らし直せるようにするため。
MODELS = {
    "v2": "Aratako/Irodori-TTS-500M-v2",
    "voice_design": "Aratako/Irodori-TTS-500M-v2-VoiceDesign",
    "v3": "Aratako/Irodori-TTS-500M-v3",
    "v3_voice_design": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
    "v4": "Aratako/Irodori-TTS-v4-Small",
    "v4_1": "Aratako/Irodori-TTS-v4.1-Small",
}

def _resolve_checkpoint(repo_id: str) -> str:
    """Path to a model.safetensors that setup has already downloaded.

    モデルの取得は setup.bat（fetch_models.py）の担当で、1つでも欠けたら
    セットアップ自体が失敗する。ここに来た時点で揃っているはずなので、
    取りに行かずローカルだけで解決する。

    ネットワークを避けるのは速度のためでもある。fetch_models.py は
    シンボリックリンクを張らない形でキャッシュへ入れる（Windows で
    権限が無いと WinError 1314 になるため）。その形だと hf_hub_download は
    ETag を照合できず、キャッシュがあっても毎回 HEAD を投げてしまう。
    """
    try:
        return hf_hub_download(
            repo_id=repo_id, filename="model.safetensors", local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{repo_id} の model.safetensors が見つかりません。"
            "setup.bat を再実行してモデルを取得してください。"
        ) from exc


# v4 は v2/v3 より強い guidance で学習されており、古い既定値のままだと
# 意味をなさない喃語になる。infer.py 自身の既定に合わせる。
# 数値を散らすと片方だけ直して食い違うので、決めるのはここだけにする。
_CFG_DEFAULTS = {
    # v4.1 は v4-Small の duration predictor だけを差し替えたもので、
    # 条件づけの作りは同じ。既定値も揃える。
    "v4": {"text": 3.0, "speaker": 5.0},
    "v4_1": {"text": 3.0, "speaker": 5.0},
    "_": {"text": 2.0, "speaker": 3.0},
}


def cfg_defaults_for(model_type: str) -> dict:
    """Guidance scales a model expects when the caller does not say."""
    return _CFG_DEFAULTS.get(model_type, _CFG_DEFAULTS["_"])


# キャプション（声の説明文）で条件づけるモデル。参照音声ではなく文章で声を
# 決めるので、生成の組み立てが他と違う。v2 版に加えて v3 の 600M 版がある。
_VOICE_DESIGN_MODELS = frozenset({"voice_design", "v3_voice_design"})


def is_voice_design_model(model_type: str) -> bool:
    return str(model_type) in _VOICE_DESIGN_MODELS


# OpenAI 互換の口は steps や尺を受け取らない（OpenAI の仕様に無い）。
# ネイティブ側の既定と同じものをここで一度だけ決める。
OPENAI_NUM_STEPS = 40
OPENAI_DURATION_SCALE = 1.0

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


def resolve_device(requested: str) -> str:
    """'auto' を今の環境で使える device に落とす。明示指定は使えるものだけ通す。

    GPU があっても CPU で回したいことがある（学習中に GPU を空けたい、
    GPU 側のドライバや ROCm が怪しくて切り分けたい、など）。auto 任せだと
    CUDA があれば必ず CUDA になるので、選べるようにしておく。
    """
    mode = str(requested or "auto").strip().lower()
    if mode and mode != "auto":
        avail = list_available_runtime_devices()
        if mode in avail:
            return mode
        # 使えない device を指定されたら黙って落とさず auto に倒す。
        print(f"[device] {mode!r} は使えません（利用可: {avail}）。auto に切り替えます。",
              flush=True)
    return default_runtime_device()


def resolve_precision(requested: str) -> str:
    """'auto' を今の GPU に合う精度へ落とす。明示指定はそのまま通す。

    auto の既定は、速い世代なら bf16（VRAM が半分で済む）、それ以外は fp32。
    GTX 1060 のような Pascal 世代で bf16 を選ぶと、エラーも警告も出ないまま
    遅くなるだけなので、自動では選ばせない。
    """
    mode = str(requested or "auto").strip().lower()
    if mode in ("fp32", "bf16"):
        return mode
    return "bf16" if gpu_info()["bf16_fast"] else "fp32"


def available_devices() -> list[str]:
    """UI に出す device の一覧。auto を先頭に置く。"""
    return ["auto"] + list_available_runtime_devices()


def resolve_and_load_model(model_type: str = "v2", precision: str = "auto",
                           device: str = "auto"):
    repo_id = MODELS.get(model_type)
    if repo_id is None:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    checkpoint_path = _resolve_checkpoint(repo_id)
    dev = resolve_device(device)
    # bf16 は CUDA/XPU でしか使えない。CPU に落としたときに bf16 のままだと
    # ランタイム側で例外になるので、ここで fp32 へ倒す。
    prec = resolve_precision(precision)
    if dev not in ("cuda", "xpu") and prec == "bf16":
        prec = "fp32"

    runtime_key = RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=dev,
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=prec,
        codec_device=dev,
        codec_precision=prec,
        compile_model=False,
        compile_dynamic=False,
    )

    runtime, reloaded = get_cached_runtime(runtime_key)
    return runtime, reloaded



# Ensure outputs dir exists
outputs_dir()  # ensures the directory exists at startup

@app.get("/")
async def root():
    return {
        "message": "Irodori-TTS API is running.",
        "endpoints": {
            "status": "GET /api/v1/status",
            "voices": "GET /api/v1/voices",
            "synthesize": "POST /api/v1/synthesize",
            "openai_models": "GET /v1/models",
            "openai_speech": "POST /v1/audio/speech"
        }
    }

@app.get("/api/v1/status")
async def get_status():
    voices = []
    meta_path = voices_metadata_path()
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            voices = [{"id": v["id"], "name": v["name"]} for v in meta.get("voices", [])]
        except:
            pass

    gpu = gpu_info()
    return {
        "status": "online",
        "device": default_runtime_device(),
        "available_models": list(MODELS.keys()),
        "registered_voices": voices,
        # 精度の選択肢。UI はこれを見て bf16 を出すか決める。旧世代の GPU で
        # bf16 を選べてしまうと、エラーも出ないまま遅くなるだけになる。
        "gpu": gpu,
        "available_precisions": ["auto", "fp32"] + (["bf16"] if gpu["bf16_fast"] else []),
        "auto_precision": resolve_precision("auto"),
        # 計算に使う device も選べる。GPU があっても CPU に落としたいことがある。
        "available_devices": available_devices(),
        "auto_device": default_runtime_device(),
    }

@app.get("/api/v1/voices")
async def get_voices():
    meta_path = voices_metadata_path()
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {"voices": meta.get("voices", [])}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    return {"voices": []}

import socket
@app.get("/api/v1/network_info")
async def get_network_info():
    ip_addr = "127.0.0.1"
    try:
        # Method 1: Try connecting to external DNS to find the active interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip_addr = s.getsockname()[0]
        s.close()
    except Exception:
        try:
            # Method 2: Fallback to hostname-based lookup
            ip_addr = socket.gethostbyname(socket.gethostname())
        except Exception:
            pass
    return {"ip": ip_addr, "port": listen_port()}

@app.get("/api/v1/outputs/{filename}")
async def get_output_file(filename: str):
    file_path = outputs_dir() / filename
    if file_path.exists():
        return FileResponse(path=file_path, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.delete("/api/v1/outputs/{filename}")
async def delete_output_file(filename: str):
    file_path = outputs_dir() / filename
    try:
        if file_path.exists():
            file_path.unlink()
        return JSONResponse(content={"status": "ok"})
    except OSError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 数値として読めない値は None にする。呼び出し側は「省略された」と同じに
# 扱えばよい。ここで例外を投げると、その先の try に入る前に落ちて 500 に
# なる。ブラウザからは JS の undefined が文字列 "undefined" として届くこと
# があり、それだけで生成が止まっていた。
_NULLISH = ("", "none", "null", "undefined", "nan")


def parse_float(val: Optional[str]) -> Optional[float]:
    if val is None or str(val).strip().lower() in _NULLISH:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def parse_int(val: Optional[str]) -> Optional[int]:
    if val is None or str(val).strip().lower() in _NULLISH:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

import json

@app.post("/api/v1/synthesize")
@app.post("/api/v1/synthesize/")
@app.post("/api/v1/synthesize/{voice_id}")
async def synthesize(request: Request, voice_id: Optional[str] = None):
    content_type = request.headers.get("content-type", "")
    data = {}
    ref_wav = None

    if "application/json" in content_type:
        try:
            data = await request.json()
        except:
            pass
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        for k, v in form.items():
            if getattr(v, "filename", None) and k == "ref_wav":
                ref_wav = v
            else:
                data[k] = v

    # Fallback to query params
    for k, v in request.query_params.items():
        if k not in data:
            data[k] = v

    if "text" not in data or not str(data["text"]).strip():
        return JSONResponse(status_code=400, content={"error": "'text' parameter is missing or empty."})

    text = str(data["text"])
    model_type = str(data.get("model_type", "v2"))
    caption = data.get("caption", None)

    num_steps = parse_int(str(data.get("num_steps", "40"))) or 40
    num_candidates = parse_int(str(data.get("num_candidates", "1"))) or 1
    seed = parse_int(str(data.get("seed", "None")))

    cfg_guidance_mode = str(data.get("cfg_guidance_mode", "independent"))
    # v4 is trained for stronger guidance than v2/v3 and degenerates into
    # babble at the older defaults; match infer.py's own defaults instead.
    _defaults = cfg_defaults_for(model_type)
    _cfg_text_default = str(_defaults["text"])
    _cfg_speaker_default = str(_defaults["speaker"])
    cfg_scale_text = parse_float(str(data.get("cfg_scale_text", _cfg_text_default))) or float(_cfg_text_default)
    cfg_scale_speaker = parse_float(str(data.get("cfg_scale_speaker", _cfg_speaker_default))) or float(_cfg_speaker_default)
    cfg_scale_caption = parse_float(str(data.get("cfg_scale_caption", "4.0"))) or 4.0
    cfg_scale = parse_float(str(data.get("cfg_scale", "None")))
    cfg_min_t = parse_float(str(data.get("cfg_min_t", "0.5"))) or 0.5
    cfg_max_t = parse_float(str(data.get("cfg_max_t", "1.0"))) or 1.0

    context_str = str(data.get("context_kv_cache", "true")).lower()
    context_kv_cache = context_str in ["true", "1", "yes"]
    
    max_text_len = parse_int(str(data.get("max_text_len", "None")))
    max_caption_len = parse_int(str(data.get("max_caption_len", "None")))

    truncation_factor = parse_float(str(data.get("truncation_factor", "None")))
    rescale_k = parse_float(str(data.get("rescale_k", "None")))
    rescale_sigma = parse_float(str(data.get("rescale_sigma", "None")))

    speaker_kv_scale = parse_float(str(data.get("speaker_kv_scale", "None")))
    speaker_kv_min_t = parse_float(str(data.get("speaker_kv_min_t", "0.9"))) or 0.9
    speaker_kv_max_layers = parse_int(str(data.get("speaker_kv_max_layers", "None")))

    t_schedule_mode = str(data.get("t_schedule_mode", "linear")).strip().lower()
    if t_schedule_mode not in ("linear", "sway"):
        t_schedule_mode = "linear"
    sway_coeff = parse_float(str(data.get("sway_coeff", "-1.0")))
    if sway_coeff is None:
        sway_coeff = -1.0
    # 尺の予測にかける倍率。上流（infer.py / SamplingRequest）と同じ 1.0 を
    # 既定にする。
    #
    # 以前は 0.75 にしていた。長すぎる窓を与えると LoRA によっては余った尺を
    # 埋めようとして語尾を捏造する（"...確認します。アイネームです"）ため。
    # ただし捏造するかは LoRA 固有で、ASR で測り直すと、ある LoRA は 0.55 超で
    # 常に捏造する一方、他の4体は 0.90 でも出ない。普遍的な安全値は無い。
    #
    # 一方で 0.75 は「言い切る前に切れる」という逆の実害を出していた。実測では
    # 71字の文が scale=0.75 で末尾 peak=14351（まだ喋っている）、1.3 で 76
    # （言い切り）。濁点カタカナや♡・絵文字が多い行はトークン密度が倍
    # （1.06 字/token 対 2.15）になり、尺の見積もりがさらに足りなくなる。
    # 捏造は聞けば分かるが、切れは取り返しがつかないので既定は 1.0 に戻す。
    duration_scale = parse_float(str(data.get("duration_scale", "1.0")))
    if duration_scale is None or duration_scale <= 0:
        duration_scale = 1.0
    # 末尾の切り落とし。潜在が平坦になった点を末尾と見なして切るが、
    # 判定を誤ると言い切る前に切れる。既定は上流と同じく有効。
    trim_str = str(data.get("trim_tail", "true")).lower()
    trim_tail = trim_str in ("true", "1", "yes")
    # 計算精度。'auto' は GPU の世代で決める（Ampere 以降なら bf16）。
    # 旧世代で bf16 を選ぶと黙って遅くなるだけなので自動では選ばない。
    precision = str(data.get("precision", "auto")).strip().lower()
    if precision not in ("auto", "fp32", "bf16"):
        precision = "auto"
    # 計算に使う device。GPU があっても CPU に落としたいことがある
    # （学習中に GPU を空ける、GPU 側が怪しいときの切り分け、など）。
    device = str(data.get("device", "auto")).strip().lower() or "auto"

    # 受け取った条件をそのまま残す。UI と手動リクエストで音が違うとき、
    # どのパラメータが効いたのかを後から突き合わせられるようにする。
    # 標準出力は cp932 なので、非対応文字で落ちないよう ASCII に退避する。
    _msg = (
        f"[synth] text={text!r} model={model_type} lora={data.get('lora_name')!r} "
        f"steps={num_steps} seed={seed} dur_scale={duration_scale} "
        f"cfg_text={cfg_scale_text} cfg_spk={cfg_scale_speaker} cfg_cap={cfg_scale_caption} "
        f"mode={cfg_guidance_mode} t_sched={t_schedule_mode} "
        f"ref_wav={'yes' if ref_wav else 'no'} voice_id={voice_id or data.get('voice_id')!r}"
    )
    print(_msg.encode("ascii", "backslashreplace").decode("ascii"), flush=True)

    # LoRA: convert registry name to adapter path.
    lora_name_raw = data.get("lora_name", None)
    try:
        lora_adapter_path = server_lora.resolve_lora_adapter_path(lora_name_raw)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # 学習したベースと違うモデルに当てさせない。UI は base の完全一致で
    # 絞っているが、API を直に叩けば通ってしまい、食い違いに気づけない。
    #
    # とくに v4 の LoRA を v4.1 に当てるのは無害に見えて意味が無い。公式
    # レシピの LoRA は modules_to_save に duration_predictor を含み、PEFT は
    # これを低ランク差分ではなくモジュールごと保存して読み込み時に置換する。
    # v4.1 は v4 と duration predictor だけが違うので、当てると唯一の改良点が
    # 丸ごと上書きされ、出力は v4 に当てたものとバイト単位で一致する
    # （実測で確認）。使えないのではなく「v4.1 にする意味が消える」ので、
    # 黙って通さず理由を返す。
    lora_base = server_lora.lora_base_of(lora_name_raw)
    if lora_base and lora_base != model_type:
        hint = ""
        if {lora_base, model_type} == {"v4", "v4_1"}:
            hint = ("。v4 の LoRA は duration predictor を丸ごと持つため、"
                    "v4.1 に当てても結果は v4 と同じになります")
        return JSONResponse(status_code=400, content={
            "error": f"LoRA '{lora_name_raw}' は base={lora_base} で学習されています。"
                     f"model_type={model_type} には当てられません{hint}",
        })

    try:
        temp_ref_path = None
        is_temp_file = False
        v_id = voice_id or data.get("voice_id")

        if ref_wav:
            temp_ref_path = Path("temp_ref.wav")
            with open(temp_ref_path, "wb") as f:
                f.write(await ref_wav.read())
            is_temp_file = True
        elif v_id:
            # Query metadata.json for the saved voice ID or name
            meta_path = voices_metadata_path()
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for v in meta.get("voices", []):
                    if v["id"] == v_id or v["name"] == v_id:
                        temp_ref_path = Path(v["path"])
                        break

        # 学習中は GPU を明け渡さない。ここで生成すると学習側と同じ GPU に
        # ベースモデルがもう一度載り、8GB 級では学習ジョブごと OOM で落ちる。
        # UI は学習中の生成を止めているが、API を直に叩かれると素通りする。
        job = server_train.active_job_id()
        if job:
            return JSONResponse(status_code=409, content={
                "error": f"学習中です（job {job}）。終わるまで生成できません。",
            })

        # 初回はチェックポイントの読み込みで十数秒かかる。ここも
        # イベントループを塞がないように別スレッドへ出す。
        runtime, _ = await run_in_threadpool(
            resolve_and_load_model, model_type, precision, device)
        is_voice_design = is_voice_design_model(model_type)

        req_kwargs = {
            "text": text,
            "ref_wav": str(temp_ref_path) if temp_ref_path else None,
            "ref_latent": None,
            "no_ref": temp_ref_path is None or is_voice_design,
            "ref_normalize_db": -16.0,
            "ref_ensure_max": True,
            "num_candidates": num_candidates,
            "decode_mode": "sequential",
            "seconds": None,
            "max_ref_seconds": 30.0,
            "max_text_len": max_text_len,
            "num_steps": num_steps,
            "seed": seed,
            "duration_scale": duration_scale,

            "cfg_guidance_mode": cfg_guidance_mode,
            "cfg_scale_text": cfg_scale_text,
            "cfg_scale_speaker": cfg_scale_speaker,
            "cfg_scale_caption": cfg_scale_caption,
            "cfg_scale": cfg_scale,
            "cfg_min_t": cfg_min_t,
            "cfg_max_t": cfg_max_t,

            "truncation_factor": truncation_factor,
            "rescale_k": rescale_k,
            "rescale_sigma": rescale_sigma,

            "context_kv_cache": context_kv_cache,
            "speaker_kv_scale": speaker_kv_scale,
            "speaker_kv_min_t": speaker_kv_min_t,
            "speaker_kv_max_layers": speaker_kv_max_layers,

            "t_schedule_mode": t_schedule_mode,
            "sway_coeff": sway_coeff,

            "lora_adapter": lora_adapter_path,

            "trim_tail": trim_tail,
        }

        if is_voice_design:
            req_kwargs["caption"] = caption
            req_kwargs["max_caption_len"] = max_caption_len

        request_obj = SamplingRequest(**req_kwargs)
        # 生成は数秒〜十数秒ブロックする。async ハンドラから直に呼ぶと
        # イベントループごと止まり、生成中は /status も含めて全エンドポイントが
        # 応答しなくなる（UI からは画面が固まったように見える）。
        result = await run_in_threadpool(runtime.synthesize, request_obj)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_urls = []

        for i, audio in enumerate(result.audios, start=1):
            filename = f"sample_{stamp}_{i:03d}.wav"
            out_path = outputs_dir() / filename
            save_wav(out_path, audio.float(), result.sample_rate)
            out_urls.append(f"/api/v1/outputs/{filename}")

        if is_temp_file and temp_ref_path and temp_ref_path.exists():
            os.remove(temp_ref_path)

        timing_details = []
        if hasattr(result, "stage_timings"):
            timing_details = [f"{name}: {sec*1000:.1f}ms" for name, sec in result.stage_timings]

        return JSONResponse(content={
            "status": "success",
            "seed_used": result.used_seed,
            "results": out_urls,
            "timings": timing_details
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    print(f"[Request] {client_host} -> {request.method} {request.url.path}", flush=True)
    response = await call_next(request)
    return response

@app.get("/v1/models")
@app.get("/v1/models/")
async def openai_models():
    models = []
    models.append({"id": "tts-1", "object": "model", "created": 1629853985, "owned_by": "irodori"})
    models.append({"id": "tts-1-hd", "object": "model", "created": 1629853985, "owned_by": "irodori"})

    # ベースモデルそのものも選べるようにする（LoRA 無しで鳴らす場合）。
    # voice_design は caption とセットで使うもので、この口には caption を
    # 渡す手段がないため出さない。
    for base in MODELS:
        if is_voice_design_model(base):
            continue
        models.append({
            "id": base, "object": "model", "created": 1629853985, "owned_by": "irodori",
        })

    # 登録済みの LoRA を voice として出す。/v1/audio/speech の voice に
    # そのまま渡せる名前で、どのベースで鳴るかも添える。
    try:
        payload = json.loads(server_lora.list_loras().body.decode("utf-8"))
        for l in payload.get("loras", []):
            # 話し方のみのものは話者を持たず、参照音声なしでは誰の声か
            # 決まらない。この口は参照音声を受け取らないので出さない。
            if "voice" not in (l.get("provides") or []):
                continue
            # base が読めないものは載せるモデルを決められない。一覧に出して
            # 選ばせておいて 400 を返すより、最初から出さない。
            if l.get("base") not in MODELS:
                continue
            models.append({
                "id": l.get("name"),
                "object": "model",
                "created": 1629853985,
                "owned_by": "user",
                "base": l.get("base"),
            })
    except Exception as exc:  # noqa: BLE001 - 一覧が出せなくても既定は返す
        print(f"[OpenAI Models] could not list LoRAs: {exc}", flush=True)

    return {"object": "list", "data": models}

@app.post("/v1/audio/speech")
@app.post("/v1/audio/speech/")
@app.post("/audio/speech")
@app.post("/")
async def openai_audio_speech(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        print(f"[OpenAI Error] Failed to parse JSON: {e}")
        return JSONResponse(status_code=400, content={"error": "Invalid JSON mapping"})
        
    text = data.get("input", "")
    voice_param = str(data.get("voice", "") or "").strip()
    model_param = str(data.get("model", "") or "").strip()

    print("\n" + "="*50, flush=True)
    print("[API Request] /v1/audio/speech", flush=True)
    print(f"   -> model: {model_param}", flush=True)
    print(f"   -> voice: {voice_param}", flush=True)
    print("="*50 + "\n", flush=True)

    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "Missing 'input' parameter."})

    # voice に LoRA 名、model にベースを取る。呼ぶ側は LoRA 名だけ知っていれば
    # よく、ベースは登録情報から決まる。model を明示されたときだけ突き合わせ、
    # 食い違えば止める（v3 の adapter を v4 で読むと KeyError で落ちる）。
    lora_name = None
    lora_adapter_path = None
    model_type = "v2"
    if voice_param:
        try:
            lora_adapter_path = server_lora.resolve_lora_adapter_path(voice_param)
        except ValueError:
            return JSONResponse(status_code=400, content={
                "error": f"unknown voice {voice_param!r}. "
                         "Pass a registered LoRA name (see GET /v1/models).",
            })
        if not lora_adapter_path:
            # resolve_lora_adapter_path は "none"/"null"/"off"/"base" を
            # 「LoRA なし」と解釈して None を返す。ここで黙って通すと、
            # 指定したつもりの利用者に素のモデルの声が返る。
            return JSONResponse(status_code=400, content={
                "error": f"{voice_param!r} is not a LoRA name. "
                         "Omit 'voice' to synthesize without a LoRA, "
                         "or pass a name from GET /v1/models.",
            })
        # 話し方のみの LoRA は話者を持たない。参照音声を受け取らない
        # この口では誰の声か決まらず、seed 次第で別人（実測では男声）が
        # 出る。黙って違う声を返すより、使えないと言って止める。
        if not server_lora.lora_has_voice(voice_param):
            return JSONResponse(status_code=400, content={
                "error": f"LoRA {voice_param!r} carries speaking style only, "
                         "so it cannot pin a speaker on this endpoint. "
                         "Use a LoRA whose provides includes 'voice' "
                         "(see GET /v1/models), or call /api/v1/synthesize "
                         "with a reference wav.",
            })
        # base が読めないものを既定の v2 に載せると、合わない adapter を
        # 読んで 500 になる。分からないなら鳴らさず理由を返す。
        resolved_base = server_lora.lora_base_of(voice_param)
        if not resolved_base:
            return JSONResponse(status_code=400, content={
                "error": f"LoRA {voice_param!r} has no usable base model recorded. "
                         "Re-register it, or call /api/v1/synthesize with an explicit model_type.",
            })
        lora_name = voice_param
        model_type = resolved_base

    # tts-1 / tts-1-hd は OpenAI 側の既定値なので、ベース指定とは見なさない。
    if model_param and model_param not in ("tts-1", "tts-1-hd"):
        if model_param not in MODELS:
            return JSONResponse(status_code=400, content={
                "error": f"unknown model {model_param!r}. Expected one of "
                         f"{[m for m in sorted(MODELS) if not is_voice_design_model(m)]} or tts-1.",
            })
        if lora_name and model_param != model_type:
            return JSONResponse(status_code=400, content={
                "error": f"LoRA {lora_name!r} is for {model_type}, not {model_param}.",
            })
        model_type = model_param

    # voice_design は caption（どんな声かの説明）とセットで使うモデルだが、
    # この口に caption を渡す手段がない。空のまま鳴らすと何を指定したのか
    # 分からない音が返るので、受け付けない。
    if is_voice_design_model(model_type):
        return JSONResponse(status_code=400, content={
            "error": "voice_design needs a caption, which this endpoint cannot take. "
                     "Use /api/v1/synthesize with model_type=voice_design and caption=...",
        })

    print(f"[OpenAI Speech] resolved -> model_type={model_type} lora={lora_name}", flush=True)

    try:
        # 学習中は GPU を明け渡さない。ここで生成すると学習側と同じ GPU に
        # ベースモデルがもう一度載り、8GB 級では学習ジョブごと OOM で落ちる。
        # UI は学習中の生成を止めているが、API を直に叩かれると素通りする。
        job = server_train.active_job_id()
        if job:
            return JSONResponse(status_code=409, content={
                "error": f"学習中です（job {job}）。終わるまで生成できません。",
            })

        # OpenAI の仕様に精度を渡す口は無いので auto に任せる。ここだけ fp32
        # 固定にすると、同じ GPU なのにネイティブ API と結果が食い違う。
        runtime, _ = await run_in_threadpool(
            resolve_and_load_model, model_type, "auto")
        defaults = cfg_defaults_for(model_type)

        req_kwargs = {
            "text": text,
            "ref_wav": None,
            "ref_latent": None,
            "no_ref": True,
            "ref_normalize_db": -16.0,
            "ref_ensure_max": True,
            "num_candidates": 1,
            "decode_mode": "sequential",
            # 尺は予測に任せる。30 秒を渡すと上限まで引き延ばされ、参照音声を
            # 持たない話し方のみの LoRA でちょうど 30 秒の音が返っていた。
            "seconds": None,
            "duration_scale": OPENAI_DURATION_SCALE,
            "max_ref_seconds": 30.0,
            "max_text_len": None,
            "num_steps": OPENAI_NUM_STEPS,
            "seed": None,
            "cfg_guidance_mode": "independent",
            "lora_adapter": lora_adapter_path,
            "cfg_scale_text": defaults["text"],
            "cfg_scale_speaker": defaults["speaker"],
            "cfg_scale_caption": 4.0,
            "cfg_scale": None,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "truncation_factor": None,
            "rescale_k": None,
            "rescale_sigma": None,
            "context_kv_cache": True,
            "speaker_kv_scale": None,
            "speaker_kv_min_t": 0.9,
            "speaker_kv_max_layers": None,
            "trim_tail": True,
        }

        request_obj = SamplingRequest(**req_kwargs)
        # 生成は数秒〜十数秒ブロックする。async ハンドラから直に呼ぶと
        # イベントループごと止まり、生成中は /status も含めて全エンドポイントが
        # 応答しなくなる（UI からは画面が固まったように見える）。
        result = await run_in_threadpool(runtime.synthesize, request_obj)
        
        if not result.audios:
            return JSONResponse(status_code=500, content={"error": "Generation failed"})
            
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"openai_{stamp}.wav"
        out_path = outputs_dir() / filename
        save_wav(out_path, result.audios[0].float(), result.sample_rate)
        
        return FileResponse(path=out_path, media_type="audio/wav")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = listen_port()
    print("[API] Starting Irodori-TTS Server...")
    print(f"[API] Listening on 0.0.0.0:{port}", flush=True)
    # 明示的にIPv4の0.0.0.0へバインド（ポートは IRODORI_PORT で指定可能）
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
