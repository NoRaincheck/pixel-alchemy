from pathlib import Path

from pixel_alchemy.pipeline.upscayl_pipeline import upscayl_pipeline

img_path = Path(__file__).with_name("hello-world.jpg")

upscayl_pipeline(img_path, "test_pipelined.png")
