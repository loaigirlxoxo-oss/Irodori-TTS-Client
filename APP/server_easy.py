"""かんたん学習タブのバックエンド。

声のデザインから LoRA 登録までを一続きにする。工程そのものは既存の API
（synthesize / datasets / lora.jobs）を順に呼ぶだけで、このモジュールが持つのは
「順番」と「採用の判断」だけにする。

設計: docs/plans/2026-09-19-かんたん学習タブ-design.md
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()

# 同梱のひな形。配布版ではアプリの中（読み取り専用）に入る。
PRESET_DIR = Path(__file__).resolve().parent / "presets"
LINES_FILE = PRESET_DIR / "easy_train_lines.txt"
EVAL_FILE = PRESET_DIR / "easy_eval_lines.txt"


def user_lines_file() -> Path:
    """編集して使うセリフファイル。

    同梱のひな形をそのまま読むと、配布版では書き込めず「編集」が成立しない。
    data の下に写しを持ち、以降はそちらを読む。Electron 側も同じ場所を開く
    （main.js の getEasyLinesPath）ので、開いた先と読む先が必ず一致する。

    手で直した写しは、ひな形を更新しても上書きしない（上書きすると、直したセリフがアプリの
    更新で消える）。手を入れていない写し（＝以前の版のひな形と同じ中身）だけ、新しいひな形に
    置き換える。これをしないと、同梱の台本を直しても、入れてあるアプリには届かない。
    """
    from data_paths import data_root

    target = data_root() / "presets" / "easy_train_lines.txt"
    if LINES_FILE.is_file() and (not target.is_file() or _is_untouched_old_copy(target)):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(LINES_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return target


# 以前の版に同梱していた台本の sha256（改行を LF にそろえた中身）。main.js も同じファイルを読む。
PAST_LINES_FILE = PRESET_DIR / "easy_train_lines.past.json"


def _is_untouched_old_copy(target: Path) -> bool:
    """写しが、以前の版のひな形と一字一句同じか（＝利用者が手を入れていないか）。"""
    import hashlib

    try:
        past = set(json.loads(PAST_LINES_FILE.read_text(encoding="utf-8"))["sha256"])
        text = target.read_bytes().replace(b"\r\n", b"\n")
    except (OSError, ValueError, KeyError):
        return False        # 判断できないときは触らない（手で直したものを消さない側に倒す）
    return hashlib.sha256(text).hexdigest() in past


# 学習素材にする文の長さ。短い文を入れない理由は2つある。数文字の音声だと
# Whisper が前後を埋めようとして幻聴を起こし、CER が実態と無関係に悪化する。
# SIM も1秒に満たない音声では speaker embedding が安定せず、似ているかどうかの
# 判定にならない。ランタイム側にも ref_min_seconds=1.0 の下限がある。
#
# 2026-10-10: 下限を 20 字から 5 字に下げた。性格の型ごとの台本（easy_scripts）は、相槌や驚きのような
# 短い行が 4 割あり、20 字では読み込めない。似ている度は短い行でも使えた（実測。相槌の枠で、ほかの枠と
# 同じ水準）。書き起こしの幻聴は短い行で実際に起きるので、CER_MIN_LEN より短い行は、読み間違いの判定だけ外す。
MIN_LEN = 5
MAX_LEN = 200
CER_MIN_LEN = 20

# 性格の型ごとの台本。同梱のものをそのまま読む（data の下への写しは作らない。「編集」で開くのは、
# 型を選ばないときに使う easy_train_lines.txt だけ）。
# 暫定: 利用者が型ごとの台本を手で直す道は、まだ無い。直せるようにするには、easy_train_lines.txt と同じ
# 「写しを作り、手を入れていない写しだけ更新する」仕組みを、型ごとに持たせる必要がある。
SCRIPT_DIR = PRESET_DIR / "easy_scripts"
SCRIPT_TYPES: tuple[tuple[str, str, str], ...] = (
    ("kachiki", "勝ち気", "強気で言い返す。照れると強がる"),
    ("ottori", "おっとり", "ていねいで、ゆっくり。包みこむように話す"),
    ("genki", "元気", "明るく、感情がそのまま大きく出る"),
    ("cool", "クール", "短く言い切る。感情をあまり表に出さない"),
    ("amaenbo", "甘えんぼ", "くっついて、わがままを言って甘える"),
    ("oneesan", "お姉さん", "余裕があり、からかいながら導く"),
    ("mesugaki", "メスガキ", "相手を下に見て、あおる"),
)

# 日本語テキストに紛れ込みやすい他言語のブロック（ハングル・ハングル字母・キリル）。
_FOREIGN_RANGES = (
    (0xAC00, 0xD7AF),
    (0x1100, 0x11FF),
    (0x0400, 0x04FF),
)


class LineError(ValueError):
    """プリセットが壊れているときに投げる。

    生成を始めてから気づくと402本を作り直すことになるので、読み込みの時点で
    止める。
    """


def _foreign_chars(text: str) -> list[str]:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _FOREIGN_RANGES):
            out.append(ch)
    return out


def script_file(script: str | None) -> Path | None:
    """型の名前から、同梱の台本の場所を返す。型を選ばないとき（空）は None。"""
    if not script:
        return None
    if script not in {key for key, _, _ in SCRIPT_TYPES}:
        raise LineError(f"台本の型がありません: {script}")
    return SCRIPT_DIR / f"{script}.txt"


# 枠ごとの「>> 名前=値」で書ける設定。素材を作るときだけ使い、学習に渡す文には影響しない。
_FRAME_SETTINGS = ("説明文", "LoRA強さ")


def _split_tag(line: str) -> tuple[str, str, str]:
    """1 行を (タグ, 本文, TTS に渡す文) に分ける。分けられなければ LineError。

    行頭に並んだ絵文字をタグとする。「🙏 本文」と「🙏本文」のどちらの書き方も読む。TTS に渡す文は
    書かれたとおりにする（空白の有無を変えない）。行頭が絵文字でない行は、最初の空白までをタグとする。
    """
    head = re.match(rf"^((?:{_STYLE_TAG_RE.pattern})+)(\s*)(.+)$", line)
    if head and head.group(3).strip():
        tag, gap, body = head.group(1), head.group(2), head.group(3).strip()
        return tag, body, f"{tag}{' ' if gap else ''}{body}"
    spaced = re.match(r"^(\S+)\s+(.+)$", line)
    if not spaced:
        raise LineError(f"タグと本文に分けられません: {line[:40]}")
    tag, body = spaced.group(1), spaced.group(2).strip()
    return tag, body, f"{tag} {body}"


def load_script(path: str | Path | None = None, min_len: int = MIN_LEN) -> list[dict]:
    """台本を読んで、行ごとの dict の列を返す。壊れていれば LineError。

    書式:
      `## 見出し`      … 枠の名前。次の見出しまでが 1 つの枠
      `> 文`           … その枠の演技の説明文（素材を作るときに、声の説明文のあとに付けて渡す）
      `>> 説明文=文`   … 素材を作るときだけ、`> ` の文の代わりに使う説明文
      `>> LoRA強さ=数` … 素材を作るときの、演技 LoRA の強さ（書かなければ 1.0）
      `#` で始まる行と空行は無視する。それ以外はセリフ（行頭の絵文字がタグ）。
    見出しと `> ` の行が無い台本（easy_train_lines.txt の書式）も、そのまま読める。

    返す dict: tag（行頭の絵文字）, body（本文）, text（TTS と学習に渡す文）, frame（枠の名前）,
    caption（素材を作るときの説明文。無ければ空）, lora_scale（演技 LoRA の強さ）。
    """
    target = Path(path) if path else user_lines_file()
    if not target.is_file():
        raise LineError(f"セリフファイルがありません: {target}")

    text = target.read_text(encoding="utf-8")

    foreign = _foreign_chars(text)
    if foreign:
        found = "".join(sorted(set(foreign)))[:20]
        raise LineError(f"他言語の文字が混ざっています: {found}")

    out: list[dict] = []
    frame, caption, settings = "", "", {}
    for raw in text.split("\n"):
        line = raw.strip()
        if line.startswith("## "):
            frame, caption, settings = line[3:].strip(), "", {}
            continue
        if line.startswith(">>"):
            name, sep, value = line[2:].partition("=")
            name, value = name.strip(), value.strip()
            if not sep or name not in _FRAME_SETTINGS or not value:
                raise LineError(f"読めない設定です（使えるのは {'・'.join(_FRAME_SETTINGS)}）: {line[:40]}")
            if name == "LoRA強さ":
                try:
                    settings[name] = float(value)
                except ValueError as exc:
                    raise LineError(f"LoRA強さ が数ではありません: {line[:40]}") from exc
                if not 0.0 < settings[name] <= 3.0:
                    raise LineError(f"LoRA強さ は 0 より大きく 3 以下にしてください: {line[:40]}")
            else:
                settings[name] = value
            continue
        if line.startswith(">"):
            caption = line[1:].strip()
            continue
        if not line or line.startswith("#"):
            continue

        tag, body, spoken = _split_tag(line)

        if not min_len <= len(body) <= MAX_LEN:
            raise LineError(
                f"{len(body)}字で範囲外（{min_len}〜{MAX_LEN}字）: {body[:30]}"
            )

        # 「だ い じ ょ う ぶ」のような表記は変に鳴る（実機確認済み）。
        # 速度の指示はタグ側に任せ、間は読点と三点リーダで作る。
        if re.search(r"\S　\S　\S", body):
            raise LineError(f"全角スペース区切りは変に鳴ります: {body[:30]}")

        out.append({"tag": tag, "body": body, "text": spoken, "frame": frame,
                    "caption": settings.get("説明文", caption),
                    "lora_scale": settings.get("LoRA強さ", 1.0)})

    return out


def load_lines(path: str | Path | None = None, min_len: int = MIN_LEN) -> list[tuple[str, str]]:
    """(タグ, 本文) の列を返す。壊れていれば LineError。書式は load_script と同じ。"""
    return [(row["tag"], row["body"]) for row in load_script(path, min_len)]


# 評価文は短文も測る（驚き・怒りなどの一言）。学習文の下限 MIN_LEN は
# 生成素材の品質のためのもので、評価には当てはめない。
EVAL_MIN_LEN = 8


def load_eval_lines(path: str | Path | None = None) -> list[tuple[str, str]]:
    """採用の判断に使う評価文（短文から長文、表現を変えて14本）。

    学習に使った402本とは別に持つ。学習した文で評価すると、覚えた文を
    読ませることになって有利に出るため。
    """
    return load_lines(path or EVAL_FILE, min_len=EVAL_MIN_LEN)


# === 生成ジョブ ===
#
# 合成そのものは既存の /api/v1/synthesize を使う。あれは Request を受け取る
# 作りなので in-process で呼ぶと組み立てが面倒だが、それ以上に「生成処理を
# 二重に持たない」ことを優先する。自分の口を叩くので、パラメータの既定値や
# 将来の変更が片方だけ古くなることがない。1本あたりの往復は数ミリ秒で、
# 合成本体（約1秒）に対して無視できる。

def _jobs_root() -> Path:
    """工程の記録を置く場所。

    Python のソース隣に作ると、配布版（インストール先が読み取り専用の
    ことがある）で最初の書き込みに失敗して、タブごと使えなくなる。
    学習ジョブと同じく data の下に置く。
    """
    from data_paths import data_root

    return data_root() / "easy_jobs"


class GenerateRequest(BaseModel):
    ref_wav: str = Field(..., description="①で確定した参照音声の絶対パス")
    model_type: str = Field("v4_1")
    # 素材に透かしを入れるか。入れない素材（既定）は、学習のあとで消す約束で作る（TrainRequest.delete_materials）。
    watermark: bool = Field(False, description="素材に電子透かしを入れる（入れた素材は消さずに残す）")
    limit: int | None = Field(None, ge=1, description="試験用。既定は全件")
    # 台本の型（SCRIPT_TYPES の名前）。空なら、編集できる easy_train_lines.txt を読む（これまでどおり）。
    script: str = Field("", description="台本の型。空なら easy_train_lines.txt")
    # 素材を作るときに当てる演技 LoRA の名前。空なら当てない（これまでどおり）。
    acting_lora: str = Field("", description="素材づくりに当てる演技 LoRA。空なら当てない")
    # 声を作るときに書いた声の説明文。台本の枠の説明文の前に付けて渡す。
    voice_caption: str = Field("", description="声の説明文。枠の説明文の前に付ける")
    dataset: str | None = Field(
        None, description="指定すると生成後にこの名前でデータセットを作る"
    )


# いま生成がランタイムを掴んでいるか。かんたん学習は工程の区切りでキャッシュを
# 空にして VRAM を返すが、別のタブや外部クライアントが生成している最中に空に
# すると、次の生成が2つ目のランタイムをGPUへ載せて二重常駐になる。
# 掴んでいる人が居なくなってから空にする。
_runtime_users = 0
_runtime_users_lock = threading.Lock()


def runtime_busy() -> bool:
    with _runtime_users_lock:
        return _runtime_users > 0


def release_runtime_if_idle() -> bool:
    """誰も掴んでいなければ、ランタイムのキャッシュを空にする。

    確認と破棄を同じロックの中でやる。分けると、確認の直後に始まった生成が
    既存のランタイムを取り出したあとでキャッシュが空になり、その次の生成が
    2つ目をGPUへ載せる。生成の側は、このロックを取って数を増やしてから
    ランタイムを取りに行く（admit_runtime が acquisition の前に数える）。
    """
    from irodori_tts import inference_runtime as ir

    with _runtime_users_lock:
        if _runtime_users > 0:
            return False
        with ir._RUNTIME_CACHE_LOCK:
            ir._RUNTIME_CACHE_KEY = None
            ir._RUNTIME_CACHE_VALUE = None
        return True


def _job_dir(job_id: str) -> Path:
    return _jobs_root() / job_id


def _write_status(job_id: str, data: dict) -> None:
    """途中まで書けたファイルを読ませない。

    402本の生成中はここが毎本走り、UI は2秒ごとに読む。直接上書きすると、
    切り詰めてから書き終わるまでの間に読まれて JSON が壊れ、_read_status が
    None を返す。画面には「ジョブが無い」と出て、動いている工程が失敗扱いに
    なる。別名で書いてから置き換える（os.replace は同じフォルダなら一発）。
    """
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "status.json"
    tmp = d / f"status.json.{os.getpid()}.{threading.get_ident()}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        # Windows では、読み手が開いている最中の置き換えが WinError 5 で落ちる
        # （実測：毎秒4千回読ませ続けると起きる）。読みは一瞬なので待てば通る。
        for wait in (0, 0.01, 0.02, 0.05, 0.1, 0.2):
            if wait:
                time.sleep(wait)
            try:
                os.replace(tmp, dst)
                return
            except PermissionError:
                continue
        # 置き換えが通らないまま諦めると、工程が終わったことも伝わらなくなる。
        # 直接書く。ここだけは途中を読まれうるが、_read_status が読み直す。
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        # 置き換えが成功していれば tmp はもう無い。失敗したぶんを残さない。
        try:
            tmp.unlink()
        except OSError:
            pass


def _read_status(job_id: str) -> dict | None:
    """読めなければ少し待って読み直す。

    書き手は置き換えで更新するので、そこに重なると開けないことがある
    （Windows の共有違反）。ここで None を返すと get_job が 404 になり、
    動いているジョブが画面では「無い」ことになってしまう。一瞬の衝突と、
    本当に無いことを取り違えない。
    """
    p = _job_dir(job_id) / "status.json"
    for wait in (0, 0.02, 0.05, 0.1, 0.2):
        if wait:
            time.sleep(wait)
        if not p.is_file():
            continue
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _server_module():
    """読み込み済みの server モジュール。

    `python server.py` で起動すると server は __main__ なので、`import server`
    は同じファイルを二重に読んでしまう。すでに読まれているものを使う。
    """
    import sys

    mod = sys.modules.get("server")
    if mod is None or not hasattr(mod, "listen_port"):
        mod = sys.modules.get("__main__")
    if mod is None or not hasattr(mod, "listen_port"):
        import server as mod  # テストから直接使うとき
    return mod


def _api_base() -> str:
    """自分の口の URL。ポートは server が決めたものに合わせる。"""
    return f"http://127.0.0.1:{_server_module().listen_port()}/api/v1"


def _allowed_audio_roots() -> list[Path]:
    """音声を読んでよい場所。

    CORS を全オリジンに開けているので、任意の絶対パスを受けると、悪意ある
    ページから「既知のパスの音声を outputs へコピーさせて読む」ことができて
    しまう。アプリが自分で扱う場所だけに限る。
    """
    from data_paths import data_root, datasets_dir, outputs_dir, voices_dir

    roots = [outputs_dir(), voices_dir(), datasets_dir(),
             data_root() / "inputs"]
    return [r.resolve() for r in roots if r.exists()]


def _ensure_allowed(path: Path) -> Path:
    """許可された場所の下にあることを確かめる。"""
    resolved = path.resolve()
    for root in _allowed_audio_roots():
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise HTTPException(
        403,
        f"この場所の音声は扱えません: {path}。"
        "outputs / voices / datasets / inputs の下に置いてください。",
    )


def _resolve_wav(path_or_url: str) -> Path:
    """UI から来る音声の指定を実体のパスにする。

    合成の口が返すのは /api/v1/outputs/xxx.wav という URL で、これは
    ファイルシステム上には無い。UI はその URL しか持っていないので、
    受け取る側で読み替える。絶対パスはそのまま通す。
    """
    from data_paths import outputs_dir

    if path_or_url.startswith("/api/v1/outputs/"):
        return outputs_dir() / Path(path_or_url).name
    return Path(path_or_url)


def _synthesize_one(text: str, ref_wav: str | None, model_type: str,
                    lora: str | None = None, seed: int | None = None, watermark: bool = False,
                    extra: dict | None = None) -> str:
    """1本合成して、出力 wav の絶対パスを返す。ref_wav が None なら参照なしで読む。

    watermark が False なら透かしを入れない（学習用の素材と、測ったら消す評価用の音声）。
    extra は /synthesize にそのまま足す項目（説明文、LoRA の強さなど）。
    """
    payload: dict = {"text": text, "model_type": model_type, "easy_token": INTERNAL_TOKEN, **(extra or {})}
    if watermark:
        payload["easy_watermark"] = True
    if ref_wav:
        payload["ref_wav"] = ref_wav
    if lora:
        payload["lora_name"] = lora
    if seed is not None:
        payload["seed"] = seed
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _api_base() + "/synthesize/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        result = json.loads(r.read().decode("utf-8"))
    return str(_resolve_wav(result["results"][0]))


# 読み方の指示に使う絵文字（行頭のタグと、文中に差し込んだタグ）。生成のときも学習データの文にも
# 入れる。読み間違いの判定（書き起こしとの比較）のときだけ外す。
_STYLE_TAG_RE = re.compile(r"[🀀-🫿☀-♤♦-➿⏩-⏿️‍]")


def strip_style_tags(text: str) -> str:
    """読み方の指示の絵文字を外す。♥（U+2665）は台詞の表記なので残す。"""
    return _STYLE_TAG_RE.sub("", text).strip()


# 演技 LoRA を当てて素材を作るときの、参照音声への寄せ方。演技 LoRA を当てると声が参照から離れやすいので、
# 寄せる力を強める（2026-10-04 の作り直しから使ってきた値。kv 1.1、話者 CFG 8）。
ACTING_SPEAKER_KV = 1.1
ACTING_CFG_SPEAKER = 8.0


def generation_extra(row: dict, req: "GenerateRequest") -> dict:
    """台本の 1 行を読ませるときに、/synthesize に足す項目。

    説明文は、枠に説明文があるときだけ渡す（声の説明文＋枠の説明文）。演技 LoRA を選んでいれば、
    その名前と、枠ごとの強さ、参照音声への寄せ方を渡す。どちらも無ければ空で、これまでと同じ鳴り方になる。
    """
    extra: dict = {}
    if row["caption"]:
        extra["caption"] = (req.voice_caption.strip() + row["caption"]).strip()
    if req.acting_lora:
        extra.update(lora_name=req.acting_lora, lora_scale=row["lora_scale"],
                     speaker_kv_scale=ACTING_SPEAKER_KV, cfg_scale_speaker=ACTING_CFG_SPEAKER)
    return extra


# 選別で落ちた本を作り直す回数。1 回で通る本が多く、何度やっても通らない行は台本のほうに原因があるので、
# 2 回で打ち切る（残りは、これまでどおり上位 7 割までの補充に任せる）。
REDO_ROUNDS = 2


def _redo_dropped(scored: list[dict], kept: list[dict], req: "GenerateRequest") -> int:
    """選別で落ちた本を、同じ設定・別の乱数で作り直して、同じ場所に置きかえる。作り直した本数を返す。

    作り直した本からは評価値（sim・cer）を外す。次の選別で、その本だけが測り直される。
    """
    keep = {c["path"] for c in kept}
    count = 0
    for clip in scored:
        if clip["path"] in keep:
            continue
        fresh = _synthesize_one(clip["text"], req.ref_wav, req.model_type, watermark=req.watermark,
                                extra=clip.get("extra"))
        shutil.move(fresh, clip["path"])
        for key in ("sim", "cer", "eval_failed"):
            clip.pop(key, None)
        count += 1
    return count


def _run_generate(job_id: str, req: GenerateRequest) -> None:
    status = _read_status(job_id) or {}
    try:
        rows = load_script(script_file(req.script))
        if req.limit:
            rows = rows[: req.limit]
        status.update(done=0, clips=[], state="running")
        _write_status(job_id, status)

        for row in rows:
            tag, body = row["tag"], row["body"]
            extra = generation_extra(row, req)
            path = _synthesize_one(row["text"], req.ref_wav, req.model_type, watermark=req.watermark, extra=extra)
            # 学習データの文にも、生成のときと同じ絵文字を残す（2026-10-04 に変更）。作者の回答では、
            # 学習テキストに絵文字が無いと絵文字での制御が弱まり、感情を強くかけたとき声が変わる
            # （Aratako/Irodori-TTS Issue #8）。plain は絵文字を外した文で、読み間違いを測るときにだけ使う
            # （書き起こしには絵文字が出ないので、絵文字つきの文と比べると全部が誤りに数えられる）。
            # frame は選別が枠ごとに足切りを決めるのに使う。short は、読み間違いの判定を外す印
            # （短い行は書き起こしが幻聴を起こしやすい。CER_MIN_LEN の説明を参照）。
            plain = strip_style_tags(body)
            # extra は、選別で落ちたときに同じ設定で作り直すために持っておく。
            status["clips"].append({"path": path, "text": row["text"], "plain": plain, "tag": tag,
                                    "frame": row["frame"], "short": len(plain) < CER_MIN_LEN, "extra": extra})
            status["done"] = len(status["clips"])
            _write_status(job_id, status)

        if req.dataset:
            # 作った本を1つのフォルダにまとめてから選別する。outputs に
            # 置いたままだと他の生成物に紛れて、あとから追えない。
            _stash_clips(req.dataset, status["clips"])
            status["samples_dir"] = str(samples_dir(req.dataset))
            _write_status(job_id, status)

            # 生成しただけの本には失敗作が混ざる。そのまま学習させると
            # LoRA が崩れごと覚えるので、ここで選別する。
            from easy_eval import release_eval_models, release_runtime, screen_clips

            busy = RuntimeError(
                "他の生成が続いていて、いまは選別に移れません。"
                "生成タブの処理が終わってからやり直してください。"
            )
            # 選別は声の近さのモデルと書き起こしを載せる。合成のランタイムを抱えたまま
            # だと 8GB に収まらないので、ここで返す。次に鳴らすときに読み直す。
            if not release_runtime():
                raise busy

            status["state"] = "screening"
            status["screened"] = 0
            _write_status(job_id, status)

            def _tick(done: int, total: int) -> None:
                status["screened"] = done
                status["screen_total"] = total
                _write_status(job_id, status)

            scored: list[dict] = []
            kept, report = screen_clips(status["clips"], req.ref_wav, _tick, scored_out=scored)
            # 落ちた本は、乱数を変えて作り直し、もう一度選別する。落とすだけだと素材が減り、
            # 喘ぎのように落ちやすい枠が学習データから抜ける。
            redone = 0
            for _ in range(REDO_ROUNDS):
                if len(kept) == len(scored):
                    break
                # 作り直しは合成のランタイムを使う。選別の道具と同時には 8GB に載らないので、入れ替える。
                if not release_eval_models():
                    break       # 返せないなら作り直しはあきらめ、いまの選別の結果で進む
                try:
                    made = _redo_dropped(scored, kept, req)
                except Exception as exc:  # noqa: BLE001 - 作り直しは足し算。失敗しても、通った本は使える
                    # 作り直しの途中で合成ができなくなることがある（ほかの処理が GPU を使い始めた、など）。
                    # ここで工程を失敗にすると、選別を通った素材ごと捨てることになる。作り直しをやめて先へ進む。
                    # 途中まで置きかえた本は、どれも選別で落ちた本なので、通った本（kept）には含まれない。
                    status["warn"] = (f"落ちた本の作り直しができませんでした（{type(exc).__name__}: {exc}）。"
                                      "作り直さずに、通った本だけで進めます。")
                    break
                status["redo"] = redone = redone + made
                _write_status(job_id, status)
                if not release_runtime():
                    raise busy
                again: list[dict] = []
                kept, report = screen_clips(scored, req.ref_wav, _tick, scored_out=again)
                scored = again
            report["redone"] = redone
            status["screen"] = report
            _write_sample_list(req.dataset, status["clips"], kept)

            status["state"] = "saving"
            _write_status(job_id, status)
            _create_dataset(req.dataset, kept)
            status["dataset"] = req.dataset

        status["state"] = "done"
    except Exception as exc:  # noqa: BLE001 - 失敗理由を UI に出したい
        import traceback

        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()[-1500:]
    finally:
        # 402本の生成が終わったら VRAM を返す。学習が続くので、ここで
        # 抱えたままだと学習側と二重に載る。
        try:
            from easy_eval import release_eval_models, release_runtime

            release_runtime()
            # 書き起こしと ECAPA も返す。学習はこのあとすぐ始まるので、
            # 抱えたままだと学習側と二重に載る。返せなかったことは記録する
            # （ここで例外にすると、素材はできているのに工程が失敗扱いになる）。
            if not release_eval_models():
                status["warn"] = ("評価用のモデルを返せませんでした。"
                                  "他の処理が使っている可能性があります。")
        except Exception:  # noqa: BLE001
            pass
    _write_status(job_id, status)


# 走っている工程。生成タブはこの間の生成を断る。
#
# 同じ GPU で別のモデルを鳴らされると、ランタイムが読み直しになって
# 1本 2.83秒 が 13.85秒 になる（実測。402本で19分が93分）。同じモデルでも
# LoRA を切り替えられると、共有しているアダプタが差し替わって、素材の
# 何本かに別の声が混ざる。どちらも音を聞くまで気付けない。
_ACTIVE_STATES = ("queued", "running", "tuning", "screening", "saving",
                  "measuring", "training", "evaluating")


# 自分の工程が自分の足を止めないための合言葉。
#
# かんたん学習は生成の口（/api/v1/synthesize/）を叩いて402本を作る。上の
# 判定をそのまま当てると、自分の生成が「かんたん学習の最中です」で弾かれる。
# 起動ごとに作り直すので、外から当てられない。CORS を全開にしているぶん、
# 固定文字列にはしない。
INTERNAL_TOKEN = uuid.uuid4().hex


# GPU を使い始めてよいかを決める場所。生成・学習・かんたん学習が、どれも
# ここを通る。確認と登録を別のロックでやると、その隙間で両方が通ってしまう
# （「相手は動いていない」と互いに見てから、互いに登録する）。
GPU_LOCK = threading.RLock()

_start_lock = threading.Lock()


class _Admission:
    """生成がランタイムを掴んでいる間を示す。終わったら数を戻す。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        global _runtime_users

        with _runtime_users_lock:
            _runtime_users -= 1
        return False


def admit_runtime(token: str | None = None) -> _Admission:
    """生成してよければ受け付け、掴んでいる数を1つ増やして返す。

    駄目なら HTTPException。学習中・かんたん学習中の確認と、掴んだ数の加算を
    すべて同じロックの中でやるのが肝。1つでも外に出すと、その隙間で相手が
    始まり、同じ GPU で2つ動く。

    合言葉つきの呼び出し（かんたん学習自身の402本と採用の判定）は素通しする。
    その工程を始める時点で、学習も他の生成も走っていないことを _claim が
    確かめている。
    """
    global _runtime_users

    import server_train

    with GPU_LOCK:
        if token != INTERNAL_TOKEN:
            job = active_job_id()
            if job:
                raise HTTPException(
                    409, "かんたん学習の最中です。終わるまで生成できません。")
            train_job = server_train.active_job_id()
            if train_job:
                raise HTTPException(
                    409, f"学習中です（job {train_job}）。終わるまで生成できません。")
        with _runtime_users_lock:
            _runtime_users += 1
    return _Admission()


def _claim(dataset: str | None) -> tuple[str, str | None]:
    """名前を確かめ、走っている工程が無ければ新しい id を取る。

    整えた名前も返す。_validate_name は前後の空白を落とすので、元の文字列を
    そのまま使い続けると、データセットは整えた名前で作られ、あとの学習は
    元の名前で探して 404 になる。

    二重クリックや API の同時呼び出しで2本走ると、同じ GPU のランタイムを
    奪い合って素材に別の声が混ざる。確認と登録をロックの中でやる。
    """
    # データセットを作る側の規則で見る。登録名の規則（server_lora）より
    # 厳しく、先頭の "_" を予約している。そちらで通してしまうと、402本を
    # 作り終えてデータセットを作る段になって初めて弾かれる。
    from server_dataset import _validate_name

    name = None
    if dataset is not None:
        name = _validate_name(dataset)
        samples_dir(name)   # 置き場が外へ出ないことをここで確かめる
        # データセットの作成は overwrite で既存を消す。402本かけたあとに
        # 気付いても戻せないので、始める前に断る。
        from server_dataset import _dataset_dir

        if _dataset_dir(name).is_dir():
            raise HTTPException(
                409, f"「{name}」はもうあります。別の名前にしてください。")

    with GPU_LOCK, _start_lock:
        busy = active_job_id()
        if busy:
            raise HTTPException(
                409, f"かんたん学習がすでに動いています（{busy}）。"
                     "終わるまで待ってください。")

        # GPU を使う工程は1つずつ。学習中に始めると、書き起こしや合成が
        # 学習の隣に載って 8GB 級では落ちる。生成中に始めると、共有している
        # ランタイムの LoRA を切り替え合って、どちらの音にも別の声が混ざる。
        import server_train

        train_job = server_train.active_job_id()
        if train_job:
            raise HTTPException(
                409, f"学習中です（job {train_job}）。終わるまで始められません。")
        if runtime_busy():
            raise HTTPException(
                409, "いま音声を生成しています。終わってから始めてください。")
        job_id = uuid.uuid4().hex[:12]
        _write_status(job_id, {"id": job_id, "state": "queued"})
        return job_id, name


def active_job_id() -> str | None:
    """動いている工程があればその id。無ければ None。"""
    root = _jobs_root()
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if not d.is_dir():
            continue
        s = _read_status(d.name)
        if s and s.get("state") in _ACTIVE_STATES:
            return d.name
    return None


def samples_dir(name: str) -> Path:
    """作った402本を残す場所。

    生成そのものは outputs に吐かれるが、そこは他の生成物と混ざる（実測で
    1700本以上）。あとから聞き直したり、気に入らなかった本を差し替えたり
    できるよう、その回のぶんだけを1つのフォルダにまとめる。
    データセットには採用されたぶんしか入らないので、除外された本を見たい
    ときもここを見る。
    """
    from data_paths import data_root

    root = (data_root() / "easy_samples").resolve()
    dst = (root / name).resolve()
    # 名前は API から来る。".." や絶対パスで外へ出されると、生成した wav を
    # 任意の場所へ置かれ、そこの 0001.wav を上書きしうる（CORS が全開なので
    # 外のページからも叩ける）。入口でも弾くが、組み立てる側でも確かめる。
    if dst == root or root not in dst.parents:
        raise HTTPException(400, f"使えない名前です: {name}")
    return dst


def _stash_clips(name: str, clips: list[dict]) -> None:
    """生成した本を samples_dir へ移し、clips のパスを移動先に書き換える。

    コピーではなく移動。402本はそれなりの大きさ（実測 1本 5〜6秒）なので、
    outputs に同じものを残さない。
    """
    import shutil

    # samples_dir が置き場の外を指さないことを確かめている。作り直す前に
    # 空にする。残っていると、前回のほうが本数が多かった場合に古い
    # 0400.wav などが生き残り、一覧.txt と中身が食い違う。
    dst = samples_dir(name)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(clips, start=1):
        src = Path(clip["path"])
        if not src.is_file():
            continue
        out = dst / f"{i:04d}{src.suffix.lower() or '.wav'}"
        try:
            shutil.move(str(src), str(out))
        except OSError:
            # 移せなければコピーで済ませる。ここで工程を止める理由が無い。
            shutil.copy2(str(src), str(out))
        clip["path"] = str(out)


def _write_sample_list(name: str, clips: list[dict], kept: list[dict]) -> None:
    """何を作って、どれを使ったかを一覧にする。"""
    keep = {c["path"] for c in kept}
    sep = chr(9)
    lines = [sep.join(["ファイル", "採否", "タグ", "本文"])]
    for clip in clips:
        p = Path(clip["path"])
        mark = "採用" if clip["path"] in keep else "除外"
        # text は絵文字つきなので、本文の列には絵文字を外した plain を出す（タグの列と二重にしない）
        lines.append(sep.join([p.name, mark, clip.get("tag", ""), clip.get("plain", clip.get("text", ""))]))
    text = chr(10).join(lines) + chr(10)
    (samples_dir(name) / "一覧.txt").write_text(text, encoding="utf-8")


def _create_dataset(name: str, clips: list[dict]) -> None:
    """生成した音声をデータセットにする。

    テキストは生成に使ったものが分かっているので、分割も書き起こしも通さない。
    保存するのは、生成に使ったのと同じ文（読み方の絵文字つき）。学習の文に絵文字が無いと、
    絵文字での制御が弱まり、感情を強くかけたとき声が変わる（Aratako/Irodori-TTS Issue #8）。
    """
    body = json.dumps({
        "name": name,
        "clips": [{"path": c["path"], "text": c["text"]} for c in clips],
        "notes": "かんたん学習タブが生成",
        "overwrite": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        _api_base() + "/datasets",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()


class TuneRequest(BaseModel):
    """参照ボイスの微調整。全部 0 なら加工しない。"""

    wav: str = Field(..., description="加工する wav の絶対パス")
    pitch_semitones: float = Field(0.0, ge=-12, le=12)
    formant: float = Field(0.0, ge=-10, le=10)
    breathiness: float = Field(0.0, ge=0, le=10)
    brightness: float = Field(0.0, ge=-10, le=10)


@router.post("/api/v1/easy/tune")
def tune_voice(req: TuneRequest) -> JSONResponse:
    """参照ボイスに微調整をかけ、別ファイルとして返す。

    元のファイルは残す。やり直せないと、気に入らなかったときに声から
    作り直すことになる。
    """
    from data_paths import outputs_dir
    import easy_tune

    src = _ensure_allowed(_resolve_wav(req.wav))
    if not src.is_file():
        raise HTTPException(400, f"音声がありません: {req.wav}")

    if easy_tune.is_identity(req.pitch_semitones, req.formant,
                             req.breathiness, req.brightness):
        return JSONResponse(content={"wav": str(src), "tuned": False})

    dst = outputs_dir() / f"tuned_{uuid.uuid4().hex[:8]}.wav"

    global _last_tuned, _tuned_seq, _tuned_done

    with _tuned_lock:
        _tuned_seq += 1
        mine = _tuned_seq

    try:
        easy_tune.tune(
            str(src), str(dst),
            pitch_semitones=req.pitch_semitones,
            formant=req.formant,
            breathiness=req.breathiness,
            brightness=req.brightness,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"加工に失敗しました: {exc}") from exc

    # 出来上がってから、前回の試聴用と入れ替える。つまみを動かすたびに1つ
    # できるので（350ms ごと）、残すと outputs が延々と増える。
    #
    # 作る前に入れ替えると、加工に失敗したときに前回の正常なものだけが消える。
    # また、自分より新しい要求が先に終わっていたら入れ替えない。入れ替えると
    # 新しいほうのファイルを消してしまい、画面がそれを指したまま再生できなく
    # なる（画面は新しい応答を採用して古い応答を捨てるため）。
    with _tuned_lock:
        if mine > _tuned_done:
            _tuned_done = mine
            doomed, _last_tuned = _last_tuned, dst
        else:
            doomed = dst   # 自分が古い。自分のぶんを片付ける
    # dst は毎回 uuid で作るので、doomed が dst と同じになるのは
    # 「自分を消す」場合だけ。そこを除外すると片付かない。
    if doomed is not None:
        try:
            doomed.unlink(missing_ok=True)
        except OSError:
            # 再生中などで掴まれていれば消せない。次の入れ替えでまた試す
            # ことはしないが、1つ残るだけなので追いかけない。
            pass

    return JSONResponse(content={"wav": str(dst), "tuned": True,
                                 "url": f"/api/v1/outputs/{dst.name}"})


# 直前に作った試聴用の加工音声。次を作るときに消す。
# 追い越しが起きるので、何番目の要求かも持つ（古い要求があとから終わると、
# 新しいほうのファイルを消してしまう）。
_last_tuned: Path | None = None
_tuned_seq = 0
_tuned_done = 0
_tuned_lock = threading.Lock()


class FolderRequest(BaseModel):
    """フォルダの音声を学習素材にする。

    微調整が全部 0 なら加工せず、元のファイルをそのまま使う。通すだけでも
    音は劣化するので、触られていないときは触らない。
    """

    folder: str = Field(..., description="wav の入ったフォルダ")
    dataset: str = Field(..., description="作るデータセット名")
    limit: int | None = Field(None, ge=1, description="試験用")
    pitch_semitones: float = Field(0.0, ge=-12, le=12)
    formant: float = Field(0.0, ge=-10, le=10)
    breathiness: float = Field(0.0, ge=0, le=10)
    brightness: float = Field(0.0, ge=-10, le=10)


def _tune_folder_files(
    job_id: str, status: dict, files: list[Path], req: FolderRequest
) -> list[Path]:
    """フォルダの音声すべてに微調整をかけ、加工後のパスを返す。

    参照ボイスを1本だけ加工する経路と違い、ここは素材そのものを作り替える。
    元のファイルは書き換えない。気に入らなかったときに選び直せなくなる。
    """
    from data_paths import outputs_dir
    import easy_tune

    if easy_tune.is_identity(req.pitch_semitones, req.formant,
                             req.breathiness, req.brightness):
        return files

    status.update(state="tuning", tuned=0, tune_total=len(files))
    _write_status(job_id, status)

    out = outputs_dir()
    done: list[Path] = []
    for i, src in enumerate(files):
        # 名前は job_id で分ける。同じフォルダを別の設定で2回使っても、
        # 前回の加工後が混ざらない。
        dst = out / f"easytune_{job_id}_{i:04d}.wav"
        try:
            easy_tune.tune(
                str(src), str(dst),
                pitch_semitones=req.pitch_semitones,
                formant=req.formant,
                breathiness=req.breathiness,
                brightness=req.brightness,
            )
        except Exception as exc:  # noqa: BLE001
            # 1本だけ元のまま混ぜると、その本だけ別の声で学習することになる。
            # 揃わないなら止める。
            raise RuntimeError(f"微調整できませんでした: {src.name}（{exc}）") from exc
        done.append(dst)
        status["tuned"] = i + 1
        _write_status(job_id, status)
    return done


def _split_like_dataset_tab(job_id: str, status: dict, files: list[Path],
                            staging_dirs: list[str]) -> list[Path]:
    """データセットタブの「おすすめ」と同じ切り方で分け、頭と尻の無音を切る。

    1ファイル＝1本のまま流すと、長い録音が丸ごと1本の素材になる
    （14分の音源で実際に起きた）。切り方は音源を測って決める。
    """
    from server_audio import ProbeRequest, SplitRequest, probe_sources, split_audio

    status.update(state="splitting", split_done=0, split_total=len(files))
    _write_status(job_id, status)
    rec = json.loads(probe_sources(ProbeRequest(paths=[str(f) for f in files])).body)["recommended"]
    extra = {"min_sec": rec["min_sec"]} if rec.get("min_sec") is not None else {}
    pieces: list[Path] = []
    for i, f in enumerate(files, start=1):
        res = json.loads(split_audio(SplitRequest(path=str(f), method=rec["method"], trim=True, **extra)).body)
        if res.get("staging_dir"):
            staging_dirs.append(res["staging_dir"])
        pieces.extend(Path(c["path"]) for c in res["chunks"])
        status["split_done"] = i
        _write_status(job_id, status)
    return pieces


def _run_folder(job_id: str, req: FolderRequest) -> None:
    """フォルダの音声を分けて書き起こし、データセットにする。

    生成はしない。すでにある音声を素材にするので、データセットタブの
    「おすすめ」と同じく分割と前後の空白カットをしてから、読み上げ内容を
    付ける。書き起こしは anime-whisper に任せる。
    """
    from easy_eval import transcribe

    status = _read_status(job_id) or {}
    tuned: list[str] = []
    staging_dirs: list[str] = []
    try:
        folder = Path(req.folder)
        # フォルダが許可された場所にあっても、中身が同じとは限らない。
        # リンクが外を指していれば、そこの音声を書き起こして clips[].text に
        # 載せられてしまう（CORS を全開にしているので外のページから読める）。
        # _folder_audio が1本ずつ解決したパスで確かめる。
        files = _folder_audio(folder)
        if req.limit:
            files = files[: req.limit]
        if not files:
            raise RuntimeError(f"音声が見つかりません: {folder}")

        before = list(files)
        files = _tune_folder_files(job_id, status, files, req)
        # 加工したぶんだけ控える。加工していなければ元のファイルなので触らない。
        tuned.extend(str(f) for f in files if f not in before)
        files = _split_like_dataset_tab(job_id, status, files, staging_dirs)
        if not files:
            raise RuntimeError(f"声の入った部分が見つかりません: {folder}")

        status.update(total=len(files), done=0, clips=[], state="running")
        _write_status(job_id, status)

        for path in files:
            text = transcribe(str(path)).strip()
            # 書き起こせなかった本は学習に使えない。無音や雑音の可能性が高い。
            if len(text) >= 2:
                status["clips"].append({"path": str(path), "text": text})
            status["done"] = len(status["clips"])
            _write_status(job_id, status)

        if not status["clips"]:
            raise RuntimeError("書き起こせた音声がありません")

        status["state"] = "saving"
        _write_status(job_id, status)
        _create_dataset(req.dataset, status["clips"])

        # データセットへ写したので、記録の参照先もそちらへ向ける。
        # 画面はこのあと clips[0] を「採用の判定に使う基準の声」として
        # /easy/train に渡す。outputs の加工後を指したままだと、下で
        # 片付けた直後に「参照音声がありません」で必ず失敗する。
        from server_dataset import _dataset_dir

        cdir = _dataset_dir(req.dataset) / "clips"
        for i, clip in enumerate(status["clips"], start=1):
            moved = cdir / f"{i:04d}.wav"
            if moved.is_file():
                clip["path"] = str(moved)

        status["dataset"] = req.dataset
        status["state"] = "done"
    except Exception as exc:  # noqa: BLE001
        import traceback

        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()[-1500:]
    finally:
        # 微調整で作った音声は outputs に残る。データセットへ写し終えたら
        # 用済みなので消す。失敗しても消す（残すと毎回たまる）。
        for w in tuned:
            try:
                Path(w).unlink(missing_ok=True)
            except OSError:
                pass
        # 分けた音声はデータセットへ写し終えたら用済み。失敗しても消す。
        for d in staging_dirs:
            shutil.rmtree(d, ignore_errors=True)
        # 書き起こしのモデルを抱えたままにしない。画面はこの直後に学習を
        # 始めるので、残っていると学習側と二重に載る（生成の経路では
        # _run_generate が同じことをしている）。
        try:
            from easy_eval import release_eval_models

            release_eval_models()
        except Exception:  # noqa: BLE001 - 返せなくても結果は書く
            pass
    _write_status(job_id, status)


AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")


def _folder_audio(folder: Path) -> list[Path]:
    """フォルダの中の音声。中身も1本ずつ許可された場所か確かめる。"""
    return sorted(
        _ensure_allowed(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )


class PeekRequest(BaseModel):
    folder: str = Field(..., description="中を見るフォルダ")


@router.post("/api/v1/easy/folder/peek")
def peek_folder(req: PeekRequest) -> JSONResponse:
    """フォルダの本数と、1本目を返す。

    微調整はフォルダの全本にかかる。何が起きるか聞かずに決めることになるので、
    1本目を試聴の対象にする。
    """
    folder = _ensure_allowed(Path(req.folder))
    if not folder.is_dir():
        raise HTTPException(400, f"フォルダがありません: {req.folder}")
    files = _folder_audio(folder)
    if not files:
        raise HTTPException(400, f"音声が見つかりません: {folder}")
    # 30秒を超える本の数。画面が入れた時点で知らせる（学習には、30秒以下に分けて使う）
    from server_audio import CLIP_MAX_SEC
    import soundfile as sf

    long_count = 0
    for f in files:
        try:
            long_count += sf.info(str(f)).duration > CLIP_MAX_SEC
        except Exception:  # noqa: BLE001 - 読めない本は書き起こしの段で落ちる
            pass
    return JSONResponse(content={"count": len(files), "first": str(files[0]), "long_count": long_count})


@router.post("/api/v1/easy/folder")
def start_folder(req: FolderRequest) -> JSONResponse:
    folder = _ensure_allowed(Path(req.folder))
    if not folder.is_dir():
        raise HTTPException(400, f"フォルダがありません: {req.folder}")
    # 検証した絶対パスに差し替えてから渡す。元の文字列のままだと、確かめた
    # 後にリンクを張り替えられて別の場所を読まされる余地が残る
    # （generate / train は既にこうしている）。
    req.folder = str(folder)

    job_id, req.dataset = _claim(req.dataset)
    _write_status(job_id, {"id": job_id, "state": "queued", "total": 0,
                           "done": 0, "clips": [], "folder": str(folder)})
    threading.Thread(target=_run_folder, args=(job_id, req), daemon=True).start()
    return JSONResponse(content={"job_id": job_id})


@router.get("/api/v1/easy/lines")
def get_lines(script: str = "") -> JSONResponse:
    """セリフの本数とタグ数。UI が「そのまま進めるか」を判断する材料。

    script に型の名前を渡すと、その型の台本を数える。空なら編集できる easy_train_lines.txt。
    壊れていたら 400 で理由を返す。生成を始めてから気づくと402本を作り直す
    ことになるので、タブを開いた時点で分かるようにする。
    """
    try:
        rows = load_script(script_file(script))
    except LineError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(content={
        "count": len(rows),
        "tags": len({row["tag"] for row in rows}),
        "path": str(script_file(script) or user_lines_file()),
        "editable": not script,
        "scripts": [{"key": key, "label": label, "note": note} for key, label, note in SCRIPT_TYPES],
    })


def _check_acting_lora(name: str, model_type: str) -> None:
    """演技 LoRA が、素材を作るモデルに当てられるか。当てられなければ 400。

    /synthesize も同じ判定で断るが、そちらに任せると、1 本目を作る段になってから工程が失敗する。
    """
    if not name:
        return
    import server_lora

    server = _server_module()
    try:
        server_lora.resolve_lora_adapter_path(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    base = server_lora.lora_base_of(name)
    if base and not server.is_lora_compatible(base, model_type):
        raise HTTPException(400, f"演技 LoRA「{name}」は {server.model_label(base)} で学習されています。"
                                 f"{server.model_label(model_type)} の素材づくりには当てられません")


@router.post("/api/v1/easy/generate")
def start_generate(req: GenerateRequest) -> JSONResponse:
    ref = _ensure_allowed(_resolve_wav(req.ref_wav))
    if not ref.is_file():
        raise HTTPException(400, f"参照音声がありません: {req.ref_wav}")
    req.ref_wav = str(ref)
    try:
        rows = load_script(script_file(req.script))
    except LineError as exc:
        raise HTTPException(400, str(exc)) from exc
    if req.limit:
        rows = rows[: req.limit]
    _check_acting_lora(req.acting_lora, req.model_type)

    # total はここで確定させる。スレッドが走り出す前に UI が進捗を取りに来ると
    # 0 が返り、進捗バーが一瞬空になる。
    job_id, req.dataset = _claim(req.dataset)
    _write_status(job_id, {"id": job_id, "state": "queued", "total": len(rows),
                           "done": 0, "clips": [], "ref_wav": req.ref_wav})
    threading.Thread(target=_run_generate, args=(job_id, req), daemon=True).start()
    return JSONResponse(content={"job_id": job_id})


@router.get("/api/v1/easy/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    status = _read_status(job_id)
    if status is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    return JSONResponse(content=status)


# === 学習と採用 ===
#
# 学習は既存の /api/v1/lora/jobs に任せる。終わったらチェックポイントを
# 一時名で登録して評価文を鳴らし、SIM/CER を測って採用を決める。
# 一時名は UI の試聴が使っているものと同じ前置きで、register 側が特別扱いして
# 履歴に残さない。
#
# 評価のたびに名前を変えるのは、ランタイムが「読み込み済みの LoRA」を
# 登録先のパスで覚えているため。同じ名前に別のチェックポイントを入れ直しても
# 読み直してくれず、2本目以降が1本目の重みのまま測られる（実測: 別々の LoRA
# を同じ名前に入れ替えて鳴らしたら、出力の sha256 が完全に一致した）。
# アダプタを消して忘れさせる手もあるが、最後の1つを消すと PEFT の
# active_adapter が空になり、次の読み込みが落ちる。パスを変えるのが素直。
PREVIEW_LORA = "_ckpt_preview"


def _preview_name(easy_id: str, index: int) -> str:
    """評価用の一時登録名。

    ジョブごと・チェックポイントごとに変える。同じ名前を使い回すと、
    ランタイムが読み込み済みの重みを返してしまう（この関数がある理由）。
    ステップ数ではなく順番で振るのは、checkpoint_0000040 と checkpoint_final
    のようにステップ数が同じものが並びうるため。
    ジョブIDを入れるのは、2つの工程が同時に評価へ入ったときに、片方が相手の
    重みを踏み、後片付けが相手の評価中のフォルダを消すのを避けるため。

    server_lora.is_preview_lora が見る形に揃えること。
    """
    return f"{PREVIEW_LORA}~{easy_id}~{index}"


def _drop_previews(names: list[str]) -> None:
    """評価用に作った一時登録を片付ける。

    消すのは、この評価が自分で作った名前だけ。前置きで拾うと、利用者が
    _ckpt_preview_ で始まる名前を付けていた場合にその LoRA まで消える
    （登録名の規則はこの前置きを禁じていない）。
    """
    import shutil

    from data_paths import loras_dir

    root = loras_dir().resolve()
    for name in names:
        target = (root / name).resolve()
        if target.parent == root and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)


class TrainRequest(BaseModel):
    dataset: str = Field(..., description="学習に使うデータセット名")
    lora_name: str = Field(..., description="登録する名前")
    ref_wav: str = Field(..., description="①で確定した参照音声。SIM の基準になる")
    base: str = Field("v4_1")
    # 透かしなしで作った素材を、学習が終わったら消す（素材のフォルダと、そのデータセット）。
    # 透かしの無い音声を手元に残さないため。フォルダの音声を素材にしたときは付けない。
    delete_materials: bool = Field(False, description="学習のあと、素材とデータセットを消す")
    max_steps: int | None = Field(None, ge=10, description="省略時は auto_config の推奨")
    # 学習側の下限が 10。ここを緩くすると 422 になるだけなので揃える。
    save_every: int | None = Field(None, ge=10)


def _train_batch_size() -> tuple[int, int]:
    """VRAM に合わせて学習のバッチを決める。(batch_size, 勾配の積み上げ回数)

    実効バッチは 32 で揃える。小さいカードでは1回に載せる数を減らし、
    そのぶん積み上げ回数を増やす。学習の中身は変えずに山だけ低くする。

    段の切り方は、かんたん学習の実測（ピーク約11GB、うち学習の段が約7.7GB）
    から引いた。足りないと Windows では共有メモリへ退避して極端に遅くなる。
    """
    total_gb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001 - 取れなければ一番小さい設定で回す
        total_gb = 0.0

    if total_gb >= 15.0:
        return 4, 8
    if total_gb >= 11.0:
        return 2, 16
    return 1, 32


def _q(name: str) -> str:
    """URL の1区画として安全な形にする。

    データセット名は「LoRA名_日時」で、LoRA 名にはひらがな・漢字・空白が
    入りうる（登録の規則が許している）。そのまま URL に差し込むと
    urllib が落ちる（日本語で UnicodeEncodeError、空白で InvalidURL）。
    """
    from urllib.parse import quote

    return quote(str(name), safe="")


def _http(method: str, path: str, body: dict | None = None, timeout: int = 600) -> dict:
    """既存 API を叩く。失敗したら本文もエラーに載せる。

    HTTPError をそのまま投げると「422」しか残らず、どのフィールドが悪いのか
    分からない。工程の途中で落ちたときに追えるよう、応答本文を添える。
    """
    import urllib.error

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        _api_base() + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{method} {path} が {exc.code}: {detail}") from exc
    return json.loads(raw) if raw else {}


def _wait_lora_job(job_id: str, status: dict, easy_id: str) -> dict:
    """学習の終わりを待つ。進捗はそのまま easy 側の status に映す。"""
    import time

    while True:
        time.sleep(5)
        s = _http("GET", f"/lora/jobs/{job_id}")
        status["train_step"] = s.get("current_step")
        status["train_total"] = s.get("max_steps")
        _write_status(easy_id, status)
        if s.get("state") in ("done", "failed", "error", "stopped"):
            return s


def _evaluate_checkpoints(job_id: str, ref_wav: str, base: str,
                          max_steps: int, status: dict, easy_id: str,
                          save=None, should_stop=None) -> list[dict]:
    """全チェックポイントを評価する。

    1つずつ一時名で登録して評価文を鳴らし、SIM と CER を測る。選ぶのは
    pick_by_score（全部から、合わせた点数で）。学習タブも同じ道具で測る
    （save で自分の記録へ書く、should_stop で停止を受ける）。
    """
    if save is None:
        save = lambda st: _write_status(easy_id, st)  # noqa: E731

    listing = _http("GET", f"/lora/jobs/{job_id}/checkpoints")
    items = listing.get("checkpoints", []) if isinstance(listing, dict) else listing

    targets = [(_step_of(e["name"], max_steps) or 0, e["name"]) for e in items]

    eval_lines = load_eval_lines()
    rows: list[dict] = []
    made: list[str] = []
    scratch: list[str] = []
    try:
        rows = _measure(targets, eval_lines, job_id, ref_wav, base,
                        status, easy_id, made, scratch, save, should_stop)
    finally:
        # 判定に使った音声を片付ける。途中で落ちても消す。
        for w in scratch:
            try:
                Path(w).unlink(missing_ok=True)
            except OSError:
                pass
        # 途中で落ちても片付ける。1つ数百MBあるので、失敗のたびに
        # 残していくと登録先が膨らむ。
        _drop_previews(made)
        # 評価の道具は、全部測り終えてから返す。チェックポイントごとに
        # 返すと書き起こしモデルを毎回読み直すことになり、そのたびに
        # HuggingFace へ問い合わせて時々失敗した（実測で2回）。
        try:
            from easy_eval import release_eval_models

            release_eval_models()
        except Exception:  # noqa: BLE001 - 返せなくても結果は返す
            pass
    return rows


def _measure(targets, eval_lines, job_id: str, ref_wav: str, base: str,
             status: dict, easy_id: str, made: list[str],
             made_wavs: list[str], save, should_stop=None) -> list[dict]:
    """チェックポイントを1つずつ鳴らして SIM と CER を測る。"""
    from easy_eval import (
        eval_seed, release_eval_models, release_runtime, score_lines,
        speaker_similarity, transcribe,
    )

    rows: list[dict] = []
    status["evaluate_total"] = len(targets)
    for index, (step, name) in enumerate(sorted(targets)):
        if should_stop is not None and should_stop():
            break
        status["evaluating"] = name
        save(status)

        preview = _preview_name(easy_id, index)
        made.append(preview)
        _http("POST", f"/lora/jobs/{job_id}/register",
              {"checkpoint": name, "lora_name": preview})

        # 鳴らすのと測るのを混ぜない。1本ごとに行き来すると、合成のランタイムと
        # ECAPA・書き起こしが同時に載って 13.5GB まで伸びた（実測）。
        # まとめて鳴らしてから合成を返し、それから測る。
        # 乱数は本ごとに固定し、どのチェックポイントにも同じ値を渡す。
        # 参照の声は渡さない（LoRA だけで読ませる）。参照を渡すと浅いチェックポイントでも
        # 声が寄り、コハルの学習で耳の評価（1500）と逆にステップ100〜300が選ばれた。
        # 似ている度は、これまでどおり ref_wav と比べる。
        wavs = [(_synthesize_one(f"{tag} {body}", None, base, lora=preview, seed=eval_seed(i)), tag, body)
                for i, (tag, body) in enumerate(eval_lines)]
        made_wavs.extend(w for w, _, _ in wavs)
        if not release_runtime():
            raise RuntimeError(
                "他の生成が続いていて、いまは採用の判定に移れません。"
                "生成タブの処理が終わってからやり直してください。"
            )

        results = []
        try:
            for wav, tag, body in wavs:
                sim = speaker_similarity(ref_wav, wav)
                results.append({"tag": tag, "body": body, "heard": transcribe(wav), "sim": sim})
        finally:
            # 次のチェックポイントを鳴らす前に返す。抱えたままだと合成と
            # 同時に載って 13.5GB まで伸びる（実測）。読み直しはキャッシュ
            # からなので、問い合わせは発生しない（server_audio 参照）。
            release_eval_models()

        rows.append({
            "name": name,
            "step": step,
            **dict(zip(("sim", "cer"), score_lines(results))),
        })
        status["evaluated"] = rows
        save(status)

    return rows


def _step_of(checkpoint_name: str, max_steps: int) -> int | None:
    """チェックポイント名から学習ステップ数を読む。

    最終チェックポイントは checkpoint_final という名前で数字を持たない。
    0 と見なすと、一番学習が進んだものが一番手前として扱われ、点数が近いときに
    「学習が進んだほうを選ぶ」の対象から外れる。総ステップ数として扱う。
    """
    if checkpoint_name.endswith("final"):
        return max_steps
    m = re.search(r"(\d+)$", checkpoint_name)
    return int(m.group(1)) if m else None


def _delete_materials(dataset: str) -> bool:
    """透かしなしで作った素材を消す。素材のフォルダ（easy_samples）と、それを写したデータセットの両方。

    登録した LoRA は残る。消すのは学習が最後まで終わったあとだけ（途中で失敗したときは、やり直せるよう残す）。
    """
    shutil.rmtree(samples_dir(dataset), ignore_errors=True)
    try:
        _http("DELETE", f"/datasets/{_q(dataset)}")
    except Exception:  # noqa: BLE001 - 消せなかったら、消せなかったと画面に出す
        return False
    return not samples_dir(dataset).exists()


def _run_train(easy_id: str, req: TrainRequest) -> None:
    from easy_eval import pick_by_score

    status = _read_status(easy_id) or {}
    try:
        # 学習タブの「おまかせ設定」と同じ決め方にする。ステップ数と学習範囲は素材から、
        # バッチはこの PC・この素材での試し走りで測る。「学習を速くする」は使える環境なら入れる。
        # （以前は VRAM の総量の表で決め、v4-Large・15GB 以上のときしか速い設定にならなかった）
        q = urllib.parse.urlencode({"base": req.base})
        cfg = _http("GET", f"/datasets/{_q(req.dataset)}/auto_config?{q}")
        rec = cfg.get("recommended", {})
        max_steps = req.max_steps or int(rec.get("max_steps") or 600)
        save_every = req.save_every or int(rec.get("save_every") or 100)
        preset = rec.get("preset") or "speaker_style"
        status.update(state="measuring")
        _write_status(easy_id, status)
        defaults = _http("GET", f"/lora/train_defaults?{q}")
        measured = _http("POST", "/lora/measure_batch", {
            "easy_token": INTERNAL_TOKEN, "dataset": req.dataset, "base": req.base, "preset": preset,
            "compile_blocks": bool(defaults.get("compile_available")),
        }, timeout=3600)
        batch_size = int(measured["batch_size"])
        accum = int(measured["gradient_accumulation_steps"])
        status["batch_size"] = batch_size

        status.update(state="training", train_total=max_steps)
        _write_status(easy_id, status)

        job_body = {
            "easy_token": INTERNAL_TOKEN,
            "lora_name": req.lora_name,
            "dataset": req.dataset,
            "base": req.base,
            "preset": preset,
            "max_steps": max_steps,
            "save_every": save_every,
            "batch_size": batch_size,
            "gradient_accumulation_steps": accum,
            # 測ったときと同じ条件で学習する
            "compile_blocks": bool(measured["compile_blocks"]),
            "gradient_checkpointing": bool(measured["gradient_checkpointing"]),
        }
        if measured.get("ref_max_seconds") is not None:
            job_body["ref_max_seconds"] = float(measured["ref_max_seconds"])
        job = _http("POST", "/lora/jobs", job_body)
        train_id = job["job_id"]
        status["train_job"] = train_id
        _write_status(easy_id, status)

        result = _wait_lora_job(train_id, status, easy_id)
        if result.get("state") != "done":
            raise RuntimeError(f"学習が {result.get('state')}: {result.get('error')}")

        status["state"] = "evaluating"
        _write_status(easy_id, status)
        rows = _evaluate_checkpoints(
            train_id, req.ref_wav, req.base, max_steps, status, easy_id
        )

        picked = pick_by_score(rows)
        # 始めるときに見ているが、20〜30分のあいだに別のタブや API から
        # 同じ名前で登録されていることがある。登録は同名フォルダを消して
        # 置き換えるので、ここでもう一度見る。
        from data_paths import loras_dir

        if (loras_dir() / req.lora_name.strip()).exists():
            raise RuntimeError(
                f"「{req.lora_name.strip()}」が、作っている間に登録されました。"
                "上書きしないので、別の名前でやり直してください。"
            )
        _http("POST", f"/lora/jobs/{train_id}/register",
              {"checkpoint": picked["name"], "lora_name": req.lora_name})
        status["picked"] = picked
        if req.delete_materials:
            status["materials_deleted"] = _delete_materials(req.dataset)
        status["state"] = "done"
    except Exception as exc:  # noqa: BLE001
        import traceback

        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        # どの行で落ちたかが分からないと、工程が長いぶん追うのに時間がかかる。
        status["traceback"] = traceback.format_exc()[-1500:]
    finally:
        # 成否にかかわらず VRAM を返す。評価でチェックポイントを何度も
        # 読み込むので、抱えたままにすると次の作業を圧迫する。
        try:
            from easy_eval import release_runtime

            release_runtime()
        except Exception:  # noqa: BLE001 - 解放に失敗しても工程の結果は変えない
            pass
    _write_status(easy_id, status)


@router.post("/api/v1/easy/train")
def start_train(req: TrainRequest) -> JSONResponse:
    ref = _ensure_allowed(_resolve_wav(req.ref_wav))
    if not ref.is_file():
        raise HTTPException(400, f"参照音声がありません: {req.ref_wav}")
    req.ref_wav = str(ref)

    # 名前は登録先のフォルダ名になる。Windows で作れない文字（? * " < > |）を
    # 含んでいても、弾かれるのは20〜30分かけた最後の登録なので、ここで見る。
    from data_paths import loras_dir
    from server_dataset import _validate_name as _validate_dataset
    from server_lora import _validate_name

    req.lora_name = _validate_name(req.lora_name)
    # 生成のときと同じ形に整える。前後の空白が残っていると、学習側が
    # データセットを見つけられず 404 になる（実測）。
    req.dataset = _validate_dataset(req.dataset)

    # 登録は同名のフォルダを消して入れ替える。かんたん学習には人が確かめる
    # 場面が無いので、既にある名前はここで断る。上書きはしない。
    if (loras_dir() / req.lora_name.strip()).exists():
        raise HTTPException(
            409, f"「{req.lora_name.strip()}」はもうあります。別の名前にしてください。"
        )

    easy_id, _ = _claim(None)   # データセットは既にあるもの。検査は済んでいる
    _write_status(easy_id, {"id": easy_id, "state": "queued",
                            "dataset": req.dataset, "lora_name": req.lora_name})
    threading.Thread(target=_run_train, args=(easy_id, req), daemon=True).start()
    return JSONResponse(content={"job_id": easy_id})


def _recover_orphans() -> None:
    """起動時に、途中で終わったままの工程を片付ける。

    工程はこのプロセスのスレッドで動く。プロセスが入れ替われば必ず死んで
    いるので、学習（別プロセス）と違って生き残りを疑う必要がない。
    走行中のまま残すと active_job_id() が拾い続け、生成が永久に 409 になる。
    利用者は easy_jobs を手で消すまで復帰できない。
    """
    root = _jobs_root()
    if not root.is_dir():
        return
    n = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        s = _read_status(d.name)
        if s and s.get("state") in _ACTIVE_STATES:
            s["state"] = "failed"
            s["error"] = "アプリが終了したため中断されました。"
            _write_status(d.name, s)
            n += 1
    if n:
        print(f"[easy] recovered {n} orphan job(s) -> failed", flush=True)


_recover_orphans()
