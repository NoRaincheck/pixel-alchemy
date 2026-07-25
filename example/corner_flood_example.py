from pathlib import Path

from pixel_alchemy.background_removal.corner_flood import corner_flood

img_path = Path(__file__).with_name("hello-summer.jpg")

corner_flood(img_path, "test.png", foreground=True)
