"""声の近さ（easy_char_sim）が、元の anime_speaker_embedding と同じ値を出すこと。

元のパッケージは同梱していないので、置き場所を引数か環境変数で渡す（無ければ、この比較は飛ばさず失敗させる）。
  python scripts/test_easy_char_sim.py <anime_speaker_embedding の src フォルダ> <wav> <wav> [wav ...]
CPU で走る。重みは手元の Hugging Face の置き場（HF_HOME）から読む。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # GPU を使わない
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "APP"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import easy_char_sim  # noqa: E402

# 読み込みとリサンプルの道具が違う（元は librosa、こちらは合成側と同じ読み込み＋torchaudio）ので、
# 完全には一致しない。実測の差は最大 0.013（2026-10-10、5 本の組み合わせ 10 通り）。選別の足切りの幅
# （中央値から 0.12）に対して十分小さい。元に合わせるために librosa を依存に足すことはしない。
TOLERANCE = 0.02


def main(reference_src: str, wavs: list[str]) -> None:
    sys.path.insert(0, reference_src)
    import numpy as np
    from anime_speaker_embedding import AnimeSpeakerEmbedding

    reference = AnimeSpeakerEmbedding(device="cpu", variant="char")
    ours = [easy_char_sim.embedding(w).numpy() for w in wavs]
    theirs = [np.asarray(reference.get_embedding(w)).reshape(-1) for w in wavs]
    worst = 0.0
    for i in range(len(wavs)):
        for j in range(i + 1, len(wavs)):
            a, b = float(ours[i] @ ours[j]), float(theirs[i] @ theirs[j])
            worst = max(worst, abs(a - b))
            print(f"{Path(wavs[i]).name} × {Path(wavs[j]).name}: こちら {a:.4f}  元 {b:.4f}  差 {abs(a - b):.4f}")
        same = float(ours[i] @ theirs[i])
        assert same > 0.99, f"同じ音声のベクトルが元と違う向き: {same:.4f}（{wavs[i]}）"
    assert worst <= TOLERANCE, f"声の近さが元と {worst:.4f} ずれている（許す差 {TOLERANCE}）"
    assert easy_char_sim.char_similarity(wavs[0], wavs[0]) > 0.999, "同じ音声どうしが 1 にならない"
    assert easy_char_sim.release_if_idle(), "使い終わったのに手放せない"
    print("OK  いちばん大きい差", f"{worst:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
