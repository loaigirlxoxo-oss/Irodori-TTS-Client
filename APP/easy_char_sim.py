"""かんたん学習の素材の選別に使う、声の近さ。

アニメ声・演技の声で学習した話者ベクトル（litagin 氏の anime_speaker_embedding、キャラクター単位で
学習した版）を使う。それまで使っていた一般の話者照合モデル（ECAPA、easy_eval.speaker_similarity）は、
喘ぎや叫びのような演技の声で、同じ声かどうかを見分けられなかった。

モデルの作り（speechbrain の ECAPA-TDNN の BatchNorm を GroupNorm に替えたもの）と前処理は、
anime_speaker_embedding（https://github.com/litagin02/anime_speaker_embedding）に合わせている。
重みは同梱せず、初めて使うときに Hugging Face から取得する。表記は THIRD-PARTY-NOTICES.md。
"""
from __future__ import annotations

import contextlib
import os
import threading
from collections import OrderedDict
from pathlib import Path

import torch

CHAR_REPO = "litagin/anime_speaker_embedding_ecapa_tdnn_groupnorm"
CHAR_FILE = "embedding_model.pth"
SAMPLE_RATE = 16000

_model = None
_users = 0
_lock = threading.Lock()

# 直前に計算したぶんを覚えておく（参照と候補の 2 本。理由は easy_eval._emb_cache と同じ）。
_EMB_KEEP = 2
_emb_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_emb_lock = threading.Lock()


def weights_path() -> Path:
    """重みの場所。手元に無ければ取得する。取得できなければ例外。"""
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=CHAR_REPO, filename=CHAR_FILE, local_files_only=True))
    except Exception:  # noqa: BLE001 - 手元に無い。理由を問わず取得へ進む
        pass
    # setup と同じ取得の道（1 ファイルずつ、リンクを作らない、再試行つき）
    from fetch_models import fetch_with_retry

    fetch_with_retry(CHAR_REPO, CHAR_FILE)
    return Path(hf_hub_download(repo_id=CHAR_REPO, filename=CHAR_FILE, local_files_only=True))


def _swap_batchnorm_to_groupnorm(module: torch.nn.Module, num_groups: int = 32) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm1d):
            setattr(module, name, torch.nn.GroupNorm(num_groups, child.num_features, affine=True))
        else:
            _swap_batchnorm_to_groupnorm(child, num_groups)


class _CharEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

        self.backbone = ECAPA_TDNN(input_size=80, lin_neurons=192,
                                   channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1])
        _swap_batchnorm_to_groupnorm(self.backbone)
        self.fbank = Fbank(sample_rate=SAMPLE_RATE, n_mels=80)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        peak = torch.max(torch.abs(wav))
        if peak > 1.0:
            wav = wav / peak
        # 重みは、16bit の整数の大きさ（×32768）の波形で学習されている。合わせないと別の値になる。
        return self.backbone(self.fbank(wav * 32768.0))


def _load_model():
    global _model

    if _model is not None:
        return _model
    path = weights_path()
    with _lock:
        if _model is None:
            model = _CharEmbedding()
            state = torch.load(path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            # 重みの名前がずれていると、読み込みは通るのに中身が初期値のままになる。黙って進めない。
            if [k for k in missing if k.startswith("backbone.")] or unexpected:
                raise RuntimeError(f"声の近さのモデルの重みが合いません（足りない {len(missing)}、余り {len(unexpected)}）")
            model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            _model = model
    return _model


@contextlib.contextmanager
def _in_use():
    global _users

    with _lock:
        _users += 1
    try:
        yield
    finally:
        with _lock:
            _users -= 1


def release_if_idle() -> bool:
    """使っている工程が無ければモデルを捨てる。捨てられた（または元から無い）なら True。"""
    global _model

    with _lock:
        if _users > 0:
            return False
        _model = None
    with _emb_lock:
        _emb_cache.clear()
    return True


def embedding(wav_path: str) -> torch.Tensor:
    """音声 1 本の話者ベクトル（192 次元、長さ 1）。"""
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

    from irodori_tts.inference_runtime import _load_audio   # torchaudio.load は Windows で失敗する（easy_eval と同じ理由）

    wav, sr = _load_audio(wav_path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, int(sr), SAMPLE_RATE)

    model = _load_model()
    with torch.inference_mode():
        emb = model(wav.float().to(next(model.parameters()).device))
    out = torch.nn.functional.normalize(emb.reshape(-1).float(), dim=0).cpu()
    if key is not None:
        with _emb_lock:
            _emb_cache[key] = out
            _emb_cache.move_to_end(key)
            while len(_emb_cache) > _EMB_KEEP:
                _emb_cache.popitem(last=False)
    return out


def char_similarity(a_wav: str, b_wav: str) -> float:
    """2 本の音声の声の近さ。コサイン類似度なので -1〜1。"""
    with _in_use():
        a = embedding(a_wav)
        b = embedding(b_wav)
    return float(torch.dot(a, b))
