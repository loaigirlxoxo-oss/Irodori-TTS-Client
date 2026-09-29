"""インストーラの絵を作る。アプリと同じ「純黒の地・金の細線・夜永オールド明朝」。

    python build/make_installer_art.py

作るもの（NSIS は BMP しか受け取らないので 24bit BMP で焼く）
    build/installerSidebar.bmp     164x314  完了ページの横帯
    build/uninstallerSidebar.bmp   164x314  同上（アンインストーラ）
    build/installerHeader.bmp      150x57   各ページ上部の帯

意匠を変えたときだけ流せばよい。出来た BMP はリポジトリに入れる。
"""
from __future__ import annotations

import io
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ASSETS = os.path.join(APP, "assets")

BG = (0, 0, 0)
GOLD = (201, 168, 106)        # --accent
GOLD_DIM = (90, 76, 48)       # 内側の細線。--accent を落としたもの
TEXT = (233, 236, 242)        # --text


def yonaga(size: int) -> ImageFont.FreeTypeFont:
    """夜永オールド明朝を woff2 から読む。

    Pillow は woff2 を直接開けないので、fontTools で ttf に展開して渡す。
    入っていない環境では Yu Gothic UI に落とす（アプリの代替指定と同じ）。
    """
    try:
        from fontTools.ttLib import TTFont

        f = TTFont(os.path.join(ASSETS, "YonagaOldMincho-Bold.woff2"))
        buf = io.BytesIO()
        f.flavor = None
        f.save(buf)
        buf.seek(0)
        return ImageFont.truetype(buf, size)
    except Exception:
        return ImageFont.truetype("YuGothM.ttc", size)


def paste_mark(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    """assets/mark.png を box に収めて中央に置く。縦横比は保つ。"""
    mark = Image.open(os.path.join(ASSETS, "mark.png")).convert("RGBA")
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    scale = min(w / mark.width, h / mark.height)
    size = (max(1, int(mark.width * scale)), max(1, int(mark.height * scale)))
    mark = mark.resize(size, Image.LANCZOS)
    canvas.paste(mark, (x0 + (w - size[0]) // 2, y0 + (h - size[1]) // 2), mark)


def product_name() -> tuple[str, str]:
    """package.json の productName を2行に割る。

    「Irodori-TTS Client App」→（Irodori-TTS, Client App）。空白が無ければ1行だけ。
    リポジトリごとに名前が違うので、ここで読んで焼き込む。
    """
    with io.open(os.path.join(APP, "package.json"), encoding="utf-8") as f:
        name = json.load(f).get("build", {}).get("productName", "Irodori-TTS")
    if " " in name:
        # 最初の空白で割る。最後で割ると3語の名前が「Irodori-TTS Client / App」になり、
        # 1行目が 164px の幅からはみ出す。
        head, _, tail = name.partition(" ")
        return head, tail
    return name, ""


def sidebar() -> Image.Image:
    W, H = 164, 314
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    # 二重の金枠。アプリのカードと同じ見せ方（面ではなく線で見せる）
    d.rectangle([9, 9, W - 10, H - 10], outline=GOLD, width=1)
    d.rectangle([13, 13, W - 14, H - 14], outline=GOLD_DIM, width=1)

    paste_mark(im, (38, 70, W - 38, 184))

    # 区切りの細線
    d.line([(44, 214), (W - 44, 214)], fill=GOLD_DIM, width=1)

    head, tail = product_name()
    d.text((W // 2, 240), head, font=yonaga(20), fill=TEXT, anchor="mm")
    if tail:
        d.text((W // 2, 262), tail, font=yonaga(13), fill=GOLD, anchor="mm")
    return im


def header() -> Image.Image:
    """各ページ上部の帯に載せる絵。

    帯の地は Windows 標準の白。ここだけ黒くすると切り貼りに見えるので、
    白地に金の細枠を置いた小さな札として作り、その中に印を入れる。
    地を黒にする案は捨てた。MUI_BGCOLOR で帯を黒くしたところ、そこに載る
    文字が黒のまま残って読めなくなった。テーマ描画のボタン類には
    SetCtlColors が効かず、ページごとに塗り直す手も届かない。
    """
    W, H = 150, 57
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)

    # 黒い小札。印は白地では沈むので、この中に置く
    d.rectangle([W - 60, 6, W - 6, H - 7], fill=BG)
    d.rectangle([W - 60, 6, W - 6, H - 7], outline=GOLD, width=1)
    paste_mark(im, (W - 54, 11, W - 12, H - 12))

    # 帯の下端の金線。本文との境をアプリと同じ線で見せる
    d.line([(0, H - 1), (W, H - 1)], fill=GOLD, width=1)
    return im


def main() -> None:
    side = sidebar()
    for name in ("installerSidebar.bmp", "uninstallerSidebar.bmp"):
        side.save(os.path.join(HERE, name), "BMP")
        print("作った:", os.path.join(HERE, name), side.size)
    head = header()
    head.save(os.path.join(HERE, "installerHeader.bmp"), "BMP")
    print("作った:", os.path.join(HERE, "installerHeader.bmp"), head.size)


if __name__ == "__main__":
    main()
