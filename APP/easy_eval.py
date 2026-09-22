"""かんたん学習の採用判断に使う評価。

val_loss は当てにならないので使わない。声が似ているか（SIM）と、読みが
崩れていないか（CER）の2つで決める。

設計: docs/plans/2026-09-19-かんたん学習タブ-design.md
"""
from __future__ import annotations

import contextlib
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def _server_module():
    """実行中の server モジュール。

    `python server.py` で起動すると server は __main__ になる。そこで
    `import server` すると同じファイルがもう一度読まれ、FastAPI app も
    マイグレーションも二重に走ったうえ、別インスタンスを掴むことになる。
    すでに読まれているものを探して使う。
    """
    mod = sys.modules.get("server")
    if mod is not None and hasattr(mod, "MODELS"):
        return mod
    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "MODELS"):
        return main
    import server as mod  # テストから直接使うとき

    return mod


# === SIM（話者類似度） ===
#
# 合成モデル内蔵の speaker encoder ではなく ECAPA-TDNN を使う。あちらは
# 「合成の条件づけ」用で、話者が同じかを判定するために訓練されていない。
# ECAPA は話者照合そのもののモデルで、ここでの判断に直接効く。
# setup が取得済みのものを使い、取りに行かない。

ECAPA_DIR = PROJECT_ROOT / "models" / "ecapa"
ECAPA_SAMPLE_RATE = 16000

_ecapa = None
# ECAPA を使っている本数。工程の区切りで手放すが、別の工程が測っている最中に
# 手放すと、次の1本が2つ目をGPUへ載せる。誰も使っていないときだけ手放す。
_ecapa_users = 0
_ecapa_lock = threading.Lock()


@contextlib.contextmanager
def _ecapa_in_use():
    global _ecapa_users

    with _ecapa_lock:
        _ecapa_users += 1
    try:
        yield
    finally:
        with _ecapa_lock:
            _ecapa_users -= 1


def _release_ecapa_if_idle() -> bool:
    """誰も使っていなければ手放す。確認と破棄を同じロックの中でやる。"""
    global _ecapa

    with _ecapa_lock:
        if _ecapa_users > 0:
            return False
        _ecapa = None
        with _emb_lock:
            _emb_cache.clear()
        return True


def _ecapa_model():
    """ECAPA-TDNN。初回だけ読み、以降は使い回す。

    読み込みはロックの中でやる。2つの工程が同時に選別へ入ると、どちらも
    「まだ無い」と見て2つGPUへ載せてしまう（返すつもりが倍になる）。
    """
    global _ecapa

    if _ecapa is not None:
        return _ecapa

    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    if not (ECAPA_DIR / "embedding_model.ckpt").is_file():
        raise RuntimeError(
            f"ECAPA のモデルがありません: {ECAPA_DIR}。"
            "setup.bat を再実行して取得してください。"
        )

    with _ecapa_lock:
        # ロックを取る間に、別のスレッドが読み終えていることがある。
        if _ecapa is None:
            # 既定の SYMLINK は Windows で WinError 1314（権限が無い）になる。
            # fetch_models.py が HF キャッシュで同じ問題を避けているのと同じ理由。
            _ecapa = EncoderClassifier.from_hparams(
                source=str(ECAPA_DIR),
                savedir=str(ECAPA_DIR),
                local_strategy=LocalStrategy.COPY,
            )
    return _ecapa


# 直前に計算したぶんを覚えておく。
#
# 選別は同じ参照音声に対して402回 speaker_similarity を呼ぶ。素直に書くと
# 参照側の読み込み・リサンプル・ECAPA 推論を毎回やり直すことになる。
#
# 覚えるのは2本。speaker_similarity は (参照, 候補) の順で呼ぶので、1本だと
# 候補が参照を押し出し、次の回で参照が必ず作り直しになる（意味が無い）。
# 中身が変わったものを使い回さないよう、更新時刻と大きさも鍵に入れる。
_EMB_KEEP = 2
_emb_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
# 2つの工程が同時に選別へ入ると、探した直後に相手が追い出して move_to_end が
# KeyError になる。読み書きはこのロックの中だけで行う。
# 計算そのものはロックの外でやること。中で _ecapa_model() を呼ぶと
# _ecapa_lock と掴む順が逆になり、噛み合う（デッドロック）。
_emb_lock = threading.Lock()


def _ecapa_embedding(wav_path: str) -> torch.Tensor:
    """音声1本の話者ベクトル（192次元）。

    読み込みは torchaudio.load を使わない。Windows では torchcodec の DLL
    読み込みに失敗する（既知）。合成側と同じ _load_audio を通す。
    リサンプルは純粋な torch 演算なのでそちらは使える。
    """
    import os

    try:
        st = os.stat(wav_path)
        key = (os.path.abspath(wav_path), st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None:
        with _emb_lock:
            hit = _emb_cache.get(key)
            if hit is not None:
                _emb_cache.move_to_end(key)
        if hit is not None:
            return hit

    import torchaudio

    from irodori_tts.inference_runtime import _load_audio

    wav, sr = _load_audio(wav_path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != ECAPA_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, int(sr), ECAPA_SAMPLE_RATE)

    with torch.no_grad():
        emb = _ecapa_model().encode_batch(wav.float())
    out = emb.squeeze().float().cpu()
    if key is not None:
        with _emb_lock:
            _emb_cache[key] = out
            _emb_cache.move_to_end(key)
            while len(_emb_cache) > _EMB_KEEP:
                _emb_cache.popitem(last=False)
    return out


def speaker_similarity(a_wav: str, b_wav: str) -> float:
    """2本の音声の話者類似度。コサイン類似度なので -1〜1。"""
    with _ecapa_in_use():
        a = _ecapa_embedding(a_wav)
        b = _ecapa_embedding(b_wav)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


# === CER ===

# 比較から外す文字。読点や空白の有無で誤差が出ると、読みが崩れているのか
# 書き起こしの癖なのか区別できなくなる。
# 比較から外すのは句読点と括弧だけ。長音「ー」は外さない。読点の有無は
# 書き起こしの癖で揺れるが、長音は発音そのもので、「コール」と「コル」を
# 同じ扱いにすると読み間違いが CER に出ない。CER は採用の足切りに使うので、
# 音を作る文字は必ず残す。
_IGNORED = str.maketrans("", "", " 　、。，．,.!?！？「」『』…・~〜")


def _normalize(text: str) -> str:
    return text.translate(_IGNORED)


def cer(reference: str, hypothesis: str) -> float:
    """文字誤り率。編集距離 / 元テキストの文字数。

    1.0 を超えることがある。幻聴で元より長く返したときに、そうなる。
    """
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    # Levenshtein 距離。行を使い回して O(len(hyp)) の領域で済ませる。
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(
                prev[j] + 1,          # 削除
                cur[j - 1] + 1,       # 挿入
                prev[j - 1] + (r != h),  # 置換
            ))
        prev = cur
    return prev[-1] / len(ref)


def transcribe(wav_path: str) -> str:
    """音声を書き起こす。既存の /api/v1/audio/transcribe を使う。

    no_repeat_ngram_size は 4 に固定する。モデルカードの推奨は 0 だが、
    既定のままだと幻聴で同じ語を繰り返し、CER が 57% まで悪化した（実測。
    4 にすると 33% まで下がる）。評価に使う以上、ここがぶれると判断が狂う。
    """
    import json
    import urllib.request

    from server_easy import _api_base

    body = json.dumps({
        "path": wav_path,
        "language": "Japanese",
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        _api_base() + "/audio/transcribe",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8")).get("text", "")


# === 採用 ===

# 読みが崩れているとみなす閾値。これを超えたら SIM が高くても使わない。
CER_LIMIT = 0.10

# 評価対象にする学習の進み具合。SIM/CER だけで選ぶと、時々まだ浅い
# チェックポイントが当たってしまう。総ステップのここから先だけを見ることで
# 構造的に防ぐ。
MIN_PROGRESS = 0.60


def pick_checkpoint(rows: list[dict], max_steps: int) -> dict:
    """採用するチェックポイントを決める。

    val_loss は当てにならないので使わない。SIM を主、CER を足切りにする。
    LoRA の目的は声を似せることなので SIM が本命で、CER は読みが崩れて
    いないかの確認に使う。

    rows は {"name", "step", "sim", "cer"} の列。返すのはその1件に
    "fallback"（足切りで全滅して最終に落ちたか）を足したもの。
    """
    if not rows:
        raise ValueError("評価できるチェックポイントがありません")

    late = [r for r in rows if r["step"] >= max_steps * MIN_PROGRESS]
    if not late:
        # save_every が粗いと60%以降が1つも無いことがある。何も選べない
        # よりは、あるものから選ぶ。
        late = rows

    passed = [r for r in late if r["cer"] <= CER_LIMIT]
    if passed:
        best = max(passed, key=lambda r: r["sim"])
        return dict(best, fallback=False)

    final = max(late, key=lambda r: r["step"])
    return dict(final, fallback=True)


def release_runtime() -> bool:
    """ランタイムを捨てて VRAM を返す。

    かんたん学習は生成402本と評価で何度もモデルを触るので、工程が終わった
    時点で抱えたままにしない。次に生成が来たら get_cached_runtime が読み直す。

    unload() は呼ばない。別のタブや外部クライアントが同じランタイムを掴んで
    いる最中に中身を消すと、その生成がモデルを失って落ちる。キャッシュから
    外して参照を手放すだけにすれば、掴んでいる人はそのまま鳴らし終えられ、
    最後の参照が消えた時点で VRAM が戻る。実測（RTX 5080、v4.1 を1本鳴らした
    直後）: unload 無しで +3887MiB -> +353MiB。unload と同じだけ戻る。

    inference_runtime のキャッシュは「キーが変わったら古いものを unload」
    という作りで、明示的に空にする口が無い。工程の区切りで確実に返したいので
    モジュール変数を直接触る。公開 API が生えたらそちらに移す。
    """
    import gc
    import time

    from server_easy import release_runtime_if_idle

    # 掴んでいる生成が終わるのを待つ。待たずに空にすると、その生成が終わる前に
    # 次の生成が2つ目のランタイムを載せ、二重常駐になる。待ち切れなければ
    # 空にしない（VRAM が戻らないだけで、増やすよりは軽い）。
    for _ in range(120):
        if release_runtime_if_idle():
            break
        time.sleep(0.5)
    else:
        # 返せなかった。呼び出し元は、このあと ECAPA と書き起こしを載せるか
        # 学習を始める。載せると二重常駐になるので、黙って進ませない。
        return False

    # 参照の輪が残ると解放が遅れる。ここで断ち切る。
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def release_eval_models() -> bool:
    """評価に使う道具（ECAPA と書き起こし）を捨てて VRAM を返す。

    合成のランタイムと同時に載せると 8GB では足りない。実測（RTX 5080）:

        学習の段                 7.7GB
        評価の段（同時に載せる） 13.5GB
        評価の段（段ごとに返す） 下の release と組み合わせて 7.7GB 以下

    README は VRAM 8GB 以上を推奨としているので、同時に載せたままにはできない。
    書き起こしのモデルは server_audio が抱えているので、そちらも落とす。
    """
    import time

    # 使っている工程が終わるのを少し待つ。返せないまま次へ進むと、学習や
    # 402本の生成と同時に載って VRAM が足りなくなる。
    ok_ecapa = False
    for _ in range(60):
        if _release_ecapa_if_idle():
            ok_ecapa = True
            break
        time.sleep(0.5)

    ok_whisper = False
    try:
        server_audio = sys.modules.get("server_audio")
        if server_audio is None:
            import server_audio  # noqa: PLC0415
        # 走っている書き起こしがあるときは手放さない。手放すと、次の書き起こしが
        # 2つ目をGPUへ載せてしまい、返すつもりが増やすことになる。
        # 確認と破棄は server_audio 側で同じロックの中でやる。
        for _ in range(60):
            if server_audio.release_whisper_if_idle():
                ok_whisper = True
                break
            time.sleep(0.5)
    except Exception:  # noqa: BLE001 - 返せなくても工程は続ける
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ok_ecapa and ok_whisper


# === 素材の選別 ===

# 生成した402本から、失敗作を落とすための基準。
#
# TTS は一定の割合で崩れる。声がぶれた本や読み間違えた本をそのまま学習
# させると、LoRA がその崩れごと覚える。採用の判断に使っているのと同じ
# 道具（ECAPA と anime-whisper）を、素材の選別にも使う。

# 読み間違いの上限。採用時（0.10）より緩くする。1本ずつの短い音声では
# 書き起こしが揺れやすく、厳しくすると正常な本まで落ちるため。
CLIP_CER_LIMIT = 0.25

# 声のぶれの許容。全体の中央値からこれだけ下がったら外れとみなす。
# 絶対値で切らないのは、声によって SIM の出方が違うから。
CLIP_SIM_MARGIN = 0.12

# 落としすぎると素材が足りなくなる。ここを下回ったら選別を諦めて全部使う。
CLIP_KEEP_MIN = 0.70

# 評価そのものが失敗した本の許容割合。ECAPA のモデルが無い、書き起こしが
# 落ちている、といった場合は全本が失敗値になり、全部を足切りに落としたうえで
# 「諦めて全部使う」に倒れる。選別が一度も動いていないのに素通りするので、
# 広範囲に失敗したら工程ごと止める。1本2本の揺らぎでは止めない。
CLIP_FAIL_MAX = 0.20


def screen_clips(
    clips: list[dict],
    ref_wav: str,
    on_progress=None,
) -> tuple[list[dict], dict]:
    """生成した本を評価して、使えるものだけ返す。

    clips は {"path", "text", ...} の列。返すのは (残した本, 内訳)。
    評価値は各本に sim / cer として書き足す。
    """
    scored: list[dict] = []
    failed = 0
    first_error = ""
    for i, clip in enumerate(clips):
        path = clip["path"]
        try:
            sim = speaker_similarity(ref_wav, path)
            err = cer(clip["text"], transcribe(path))
            broke = False
        except Exception as exc:  # noqa: BLE001 - 1本の失敗で全体を止めない
            sim, err = 0.0, 1.0
            broke = True
            failed += 1
            first_error = first_error or f"{type(exc).__name__}: {exc}"
        # 評価できなかった本は、値を持っていない。0.0 / 1.0 は「悪い」ではなく
        # 「測れなかった」印。数として混ぜると中央値も足切りも狂う。
        scored.append(dict(clip, sim=sim, cer=err, eval_failed=broke))
        if on_progress:
            on_progress(i + 1, len(clips))

    if scored and failed > len(scored) * CLIP_FAIL_MAX:
        raise RuntimeError(
            f"できの確認ができませんでした（{failed} / {len(scored)} 本で失敗）。"
            f"最初の失敗: {first_error}"
        )

    rated = [c for c in scored if not c["eval_failed"]]
    sims = sorted(c["sim"] for c in rated)
    median = sims[len(sims) // 2] if sims else 0.0
    sim_floor = median - CLIP_SIM_MARGIN

    kept = [c for c in rated if c["sim"] >= sim_floor and c["cer"] <= CLIP_CER_LIMIT]

    report = {
        "total": len(scored),
        "kept": len(kept),
        "sim_median": median,
        "sim_floor": sim_floor,
        "dropped_sim": sum(1 for c in rated if c["sim"] < sim_floor),
        "dropped_cer": sum(1 for c in rated if c["cer"] > CLIP_CER_LIMIT),
        "failed": failed,
        "gave_up": False,
    }

    # 選別が効きすぎたら、基準の方を疑う。素材が足りない方が損。
    # ただし戻すのは評価できた本だけ。測れなかった本は、無音や壊れている
    # 可能性がそのまま残っているので、基準を緩めても入れてよい根拠が無い。
    if len(kept) < len(rated) * CLIP_KEEP_MIN:
        report["gave_up"] = True
        report["kept"] = len(rated)
        return rated, report

    return kept, report
