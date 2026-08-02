"""Render text with mixed latin + color emoji runs to transparent RGBA images.

DejaVu Sans covers latin glyphs; Apple Color Emoji is used for emoji and both
are composed on a shared baseline. Backs the TikTok bubble-text example.

Example::

    from pixel_alchemy.text_stickers.render import measure_text, render_outlined_text, render_text

    print(measure_text("Hello 🎉", 60))          # -> (width, height)
    render_text("Hello 🎉", 60, "out.png")
    render_outlined_text("Your text here", 52, "tiktok_style.png")

CLI::

    python -m pixel_alchemy.text_stickers.render measure  "Hello 🎉" 60
    python -m pixel_alchemy.text_stickers.render render   "Hello 🎉" 60 out.png
    python -m pixel_alchemy.text_stickers.render outlined "Your text here" 52 out.png

Styling controls::

    Bubble shape and line spacing are tuned with the module-level constants in
    example/tiktok_bubble_text_example.py:

    pointsize   font size in pixels
    radius      rounded-corner radius of each bubble
    pad         horizontal padding inside each bubble (radius + space width)
    vpad        vertical padding inside each bubble
    overlap     vertical overlap between consecutive bubbles; lower it to
                increase the gap between lines
    maxwidth    max line width in pixels before text wraps

    This module only rasterizes the glyph runs onto a transparent RGBA image;
    edit those constants in the example and rerun it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEJAVU_FONT = str(Path.home() / "Library/Fonts/DejaVuSans.ttf")
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
EMOJI_STRIKE = 64

EMOJI_RE = re.compile(
    r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B50\u2B55]"
    r"|[\uFE0F\u200D\u20E3]"
    r"|[#*0-9]\uFE0F\u20E3"
)


def _split_runs(text: str) -> list[tuple[str, bool]]:
    """Split text into (segment, is_emoji) runs."""
    runs: list[tuple[str, bool]] = []
    current = ""
    current_emoji: bool | None = None
    for ch in text:
        is_emoji = bool(EMOJI_RE.match(ch))
        if current_emoji is None or is_emoji == current_emoji:
            current += ch
        else:
            runs.append((current, current_emoji))
            current, current_emoji = ch, is_emoji
        if current_emoji is None:
            current_emoji = is_emoji
    if current:
        runs.append((current, current_emoji))
    return runs


def _expand(canvas: Image.Image, width: int, height: int) -> Image.Image:
    if canvas.width >= width and canvas.height >= height:
        return canvas
    new = Image.new("RGBA", (max(canvas.width, width), max(canvas.height, height)), (0, 0, 0, 0))
    new.paste(canvas, (0, 0))
    return new


def layout_text(text: str, size: int) -> Image.Image:
    """Layout text as an RGBA image: transparent background, black latin, color emoji."""
    latin = ImageFont.truetype(DEJAVU_FONT, size)
    emoji = ImageFont.truetype(EMOJI_FONT, EMOJI_STRIKE, index=0)
    ascent, descent = latin.getmetrics()
    scale = size / EMOJI_STRIKE
    canvas = Image.new("RGBA", (1, ascent + descent + 2), (0, 0, 0, 0))
    x = 0
    for seg, is_emoji in _split_runs(text):
        if is_emoji:
            width = int(emoji.getlength(seg)) + 8
            tile = Image.new("RGBA", (width, EMOJI_STRIKE + 8), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((4, EMOJI_STRIKE + 4), seg, font=emoji, embedded_color=True, anchor="ls")
            tile = tile.resize(
                (max(1, int(width * scale)), max(1, int((EMOJI_STRIKE + 8) * scale))),
                Image.LANCZOS,
            )
            canvas = _expand(canvas, x + tile.width, canvas.height)
            canvas.alpha_composite(tile, (x, ascent - int(EMOJI_STRIKE * scale)))
            x += int(emoji.getlength(seg) * scale)
        else:
            width = int(latin.getlength(seg))
            canvas = _expand(canvas, x + width, canvas.height)
            ImageDraw.Draw(canvas).text((x, ascent), seg, font=latin, fill=(0, 0, 0, 255), anchor="ls")
            x += width
    return canvas


def measure_text(text: str, size: int) -> tuple[int, int]:
    """Return (width, height) of the laid-out text."""
    canvas = layout_text(text, size)
    return canvas.width, canvas.height


def render_text(text: str, size: int, out: str | Path) -> Path:
    """Render text to a cropped transparent PNG at `out`; returns the path."""
    canvas = layout_text(text, size)
    canvas.crop(canvas.getbbox()).save(out)
    return Path(out)


def render_outlined_text(
    text: str,
    size: int,
    out: str | Path,
    *,
    stroke: int = 10,
    shadow_opacity: float = 0.6,
    shadow_blur: int = 4,
    shadow_offset: tuple[int, int] = (0, 4),
    canvas: tuple[int, int] = (700, 200),
) -> Path:
    """Render white text with a black outline and soft shadow to a transparent PNG.

    The text is centered on a transparent `canvas`. Returns the output path.
    """
    font = ImageFont.truetype(DEJAVU_FONT, size)
    layer = Image.new("RGBA", canvas, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    x = (canvas[0] - (box[2] - box[0])) // 2 - box[0]
    y = (canvas[1] - (box[3] - box[1])) // 2 - box[1]

    shadow = Image.new("RGBA", canvas, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text(
        (x + shadow_offset[0], y + shadow_offset[1]),
        text,
        font=font,
        fill=(0, 0, 0, int(255 * shadow_opacity)),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, int(255 * shadow_opacity)),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

    result = Image.alpha_composite(shadow, layer)
    ImageDraw.Draw(result).text((x, y), text, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
    result.save(out)
    return Path(out)


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    mode, text, size = argv[0], argv[1], int(argv[2])
    out = argv[3] if len(argv) > 3 else None
    if mode == "outlined":
        if out is None:
            raise SystemExit("outlined requires an output path")
        render_outlined_text(text, size, out)
        return
    canvas = layout_text(text, size)
    if mode == "render" and out:
        canvas.crop(canvas.getbbox()).save(out)
    print(canvas.width)


if __name__ == "__main__":
    main()
