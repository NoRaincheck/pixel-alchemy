from pathlib import Path

from pixel_alchemy.super_resolution.upscayl import upscayl

img_path = Path(__file__).with_name("hello-world.jpg")

upscayl(img_path, "test.jpg")
