"""生成した wav に、作り直すための情報を埋める。

WAV の RIFF は「知らないチャンクは読み飛ばす」決まりなので、末尾に
チャンクを足しても音は壊れない（実測: soundfile / librosa / ブラウザとも
そのまま読める）。soundfile からは文字列を書けないので、自分で足す。

入れるのは seed・モデル・説明文・本文など、同じ音をもう一度出すのに要るもの。
ファイルだけ残っていても再現できるようにする。

書き込み先は2つに分ける。

- LIST/INFO の ICMT: 半角だけの要約。エクスプローラーの「コメント」に出る。
  Windows のシェルはこの欄を CP932 として読むので、UTF-8 の日本語を入れると
  化ける（実測: 本文が `縺ｹ縲∝挨` になった）。だから半角に限る。
- IRDR: 全項目の JSON を UTF-8 で。アプリはこちらを読む。エクスプローラーは
  知らないチャンクとして読み飛ばす。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

_SOFTWARE = "Irodori-TTS Client App"

# 全項目を置く独自チャンク。RIFF の規約どおり4バイト。
_FULL = b"IRDR"


# エクスプローラー用の ID3 チャンク。RIFF の規約どおり4バイト（末尾は空白）。
_ID3 = b"id3 "


def _syncsafe(n: int) -> bytes:
    """ID3v2 のサイズ表記。各バイトの最上位ビットを使わない。"""
    return bytes(((n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F))


def _id3_comment(text: str) -> bytes:
    """COMM フレームだけを持つ ID3v2.3 タグ。

    エクスプローラーは wav の LIST/INFO を読まない（実測：詳細タブにも
    コメント列にも出なかった）。読むのは id3 チャンクのほう。
    中身は _summary と同じ半角の一行なので ISO-8859-1 で足りる。
    """
    body = b"\x00eng\x00" + text.encode("latin-1", "replace") + b"\x00"
    frame = b"COMM" + struct.pack(">I", len(body)) + b"\x00\x00" + body
    return b"ID3\x03\x00\x00" + _syncsafe(len(frame)) + frame


def _chunk(cid: bytes, payload: bytes) -> bytes:
    if len(payload) % 2:
        payload += b"\x00"          # RIFF は偶数長
    return cid + struct.pack("<I", len(payload)) + payload


def _summary(info: dict) -> str:
    """エクスプローラーに出す一行。半角のみ。seed を先頭に置く。

    列幅が狭くても seed が読めるように、順番を固定する。日本語が混じりうる
    本文・説明文は入れない（CP932 で化けるため）。
    """
    parts = []
    if info.get("seed") is not None:
        parts.append(f"seed={info['seed']}")
    if info.get("model_type"):
        parts.append(f"model={info['model_type']}")
    lora = info.get("lora_name")
    if lora and lora.isascii():
        parts.append(f"lora={lora}")
    if info.get("num_steps") is not None:
        parts.append(f"steps={info['num_steps']}")
    if info.get("duration_scale") is not None:
        parts.append(f"speed={float(info['duration_scale']):.2f}")
    cfg = [info.get("cfg_scale_text"), info.get("cfg_scale_speaker"),
           info.get("cfg_scale_caption")]
    if any(v is not None for v in cfg):
        parts.append("cfg=" + "/".join("-" if v is None else f"{float(v):g}" for v in cfg))
    if info.get("created_at"):
        parts.append(str(info["created_at"]))
    line = " ".join(parts)
    # 半角に収まらないものが紛れ込んだら落とす。化けるより消えたほうがいい。
    return line if line.isascii() else ""


def tag_wav(path: str | Path, info: dict) -> None:
    """wav の末尾に情報チャンクを足す。失敗しても音声は残す。"""
    p = Path(path)
    try:
        raw = p.read_bytes()
        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            return
        inner = b"INFO" + _chunk(b"ISFT", _SOFTWARE.encode("ascii") + b"\x00") \
                        + _chunk(b"ICMT", _summary(info).encode("ascii") + b"\x00")
        full = json.dumps(info, ensure_ascii=False, separators=(",", ":"))
        summary = _summary(info)
        out = raw + _chunk(b"LIST", inner) + _chunk(_FULL, full.encode("utf-8"))
        if summary:
            # エクスプローラーの「コメント」に出るのはこちら
            out += _chunk(_ID3, _id3_comment(summary))
        # 先頭の RIFF サイズを直す（ファイル全体 - 8）
        out = out[:4] + struct.pack("<I", len(out) - 8) + out[8:]
        p.write_bytes(out)
    except Exception:
        # 記録が付かないだけ。生成そのものは成功させる。
        pass


def read_tag(path: str | Path) -> dict | None:
    """埋めた情報を読み出す。無ければ None。

    IRDR を先に見る。無ければ ICMT を JSON として読む（要約に分ける前の
    ファイルがこの形。以前の出力も読めるようにしておく）。
    """
    p = Path(path)
    try:
        raw = p.read_bytes()
        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            return None
        icmt: bytes | None = None
        i, n = 12, len(raw)
        while i + 8 <= n:
            cid = raw[i:i + 4]
            size = struct.unpack("<I", raw[i + 4:i + 8])[0]
            payload = raw[i + 8:i + 8 + size]
            if cid == _FULL:
                return json.loads(payload.rstrip(b"\x00").decode("utf-8"))
            if cid == b"LIST" and payload[:4] == b"INFO":
                j = 4
                while j + 8 <= len(payload):
                    sid = payload[j:j + 4]
                    ssize = struct.unpack("<I", payload[j + 4:j + 8])[0]
                    if sid == b"ICMT":
                        icmt = payload[j + 8:j + 8 + ssize]
                    j += 8 + ssize + (ssize & 1)
            i += 8 + size + (size & 1)
        if icmt is not None:
            return json.loads(icmt.rstrip(b"\x00").decode("utf-8"))
    except Exception:
        return None
    return None
