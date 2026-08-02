"""TikTok-style bubble text: wrapped lines in white rounded bubbles.

Requires DejaVu Sans and Apple Color Emoji — see ``pixel_alchemy.text_stickers``.
Writes ``output.png`` next to this file. Tune the style with the constants below
(corner radius, padding, line overlap, wrap width, font size).
"""

from pathlib import Path

from PIL import Image, ImageDraw

from pixel_alchemy.text_stickers.render import layout_text, measure_text

POINTSIZE = 60
RADIUS = 5
VPAD = 4
OVERLAP = 10
MAXWIDTH = 500
TEXT = "Hello jorld another text here very long text text 2 1234 🎉🔥😎"

OUT = Path(__file__).with_name("output.png")


def _pad() -> int:
    spacew = measure_text("X ", POINTSIZE)[0] - measure_text("X", POINTSIZE)[0]
    return RADIUS + spacew


def _wrap(text: str, pad: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = (line + " " + word).strip()
        width = measure_text(candidate, POINTSIZE)[0] + 2 * pad
        if line and width > MAXWIDTH:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    return lines


def _line_images(line: str, pad: int) -> tuple[Image.Image, Image.Image]:
    """Return (white rounded rect, text) bubble images for a line."""
    text = layout_text(line, POINTSIZE)
    text = text.crop(text.getbbox())
    w, h = text.width + 2 * pad, text.height + 2 * VPAD

    rect = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(rect).rounded_rectangle((0, 0, w - 1, h - 1), radius=RADIUS, fill="white")

    text_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    text_img.alpha_composite(text, (pad, VPAD))
    return rect, text_img


def _extent(img: Image.Image, width: int) -> Image.Image:
    if img.width == width:
        return img
    out = Image.new("RGBA", (width, img.height), (0, 0, 0, 0))
    out.alpha_composite(img, ((width - img.width) // 2, 0))
    return out


def main() -> None:
    pad = _pad()
    lines = _wrap(TEXT, pad)
    rects, texts = [], []
    for line in lines:
        rect, text = _line_images(line, pad)
        rects.append(rect)
        texts.append(text)

    maxw = max(rect.width for rect in rects)
    total = sum(rect.height for rect in rects) - (len(rects) - 1) * OVERLAP
    canvas = Image.new("RGBA", (maxw, total), (0, 0, 0, 0))

    y = 0
    for rect in rects:
        canvas.alpha_composite(_extent(rect, maxw), (0, y))
        y += rect.height - OVERLAP
    y = 0
    for text in texts:
        canvas.alpha_composite(_extent(text, maxw), (0, y))
        y += text.height - OVERLAP

    canvas.save(OUT)
    print(f"wrote {OUT} ({maxw}x{total})")


if __name__ == "__main__":
    main()
