"""TikTok-style outlined text: white text with a black outline and soft shadow.

Uses ``pixel_alchemy.text_stickers.render.render_outlined_text``. Writes
``tiktok_style.png`` next to this file.
"""

from pathlib import Path

from pixel_alchemy.text_stickers.render import render_outlined_text

out = render_outlined_text("Your text here", 52, Path(__file__).with_name("tiktok_style.png"))
print(f"wrote {out}")
