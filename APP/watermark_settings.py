"""透かしの入れ方の設定。生成タブ・朗読・かんたん学習など、全部の生成に同じものを当てる。

透かしは外せない。選べるのは、透かしを入れるときに声をどう扱うかだけ。
透かしのモデル（SilentCipher）は 44.1kHz 専用で、音声は 48kHz で出てくる。

  resample : 44.1kHz に落として透かしを入れ、48kHz に戻す（以前からの入れ方）。声を2回変換する
  native44 : 44.1kHz に落として透かしを入れ、44.1kHz のまま出す。声の変換は1回
  delta55  : 透かしの成分だけを 48kHz にして元の声に足す。声は変換しない。
             透かしは強さ55（声より約57dB 下）。4つの声で読み取れることを確かめた一番弱い値
"""
from __future__ import annotations

import json

from data_paths import data_root

CHOICES: dict[str, dict] = {
    "resample": {"mode": "resample", "strength_db": None, "label": "48kHz に戻す（従来どおり）"},
    "native44": {"mode": "native44", "strength_db": None, "label": "44.1kHz のまま出す"},
    "delta55": {"mode": "delta", "strength_db": 55.0, "label": "声を変換せずに足す（弱め）"},
}
DEFAULT_CHOICE = "resample"


def _path():
    return data_root() / "watermark.json"


def load_choice() -> str:
    try:
        choice = json.loads(_path().read_text(encoding="utf-8")).get("choice")
    except (OSError, ValueError):
        return DEFAULT_CHOICE
    return choice if choice in CHOICES else DEFAULT_CHOICE


def save_choice(choice: str) -> None:
    if choice not in CHOICES:
        raise ValueError(f"unknown watermark choice: {choice!r}")
    _path().write_text(json.dumps({"choice": choice}), encoding="utf-8")


def request_kwargs(choice: str) -> dict:
    """SamplingRequest に渡す透かしの指定。"""
    c = CHOICES[choice]
    return {"watermark_mode": c["mode"], "watermark_strength_db": c["strength_db"]}
