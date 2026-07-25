from pathlib import Path

from pixel_alchemy.background_removal.birefnet import birefnet

img_path = Path(__file__).with_name("hello-summer.jpg")

birefnet(img_path, "test.png", foreground=True)
