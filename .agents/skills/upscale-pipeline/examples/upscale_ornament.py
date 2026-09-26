#!/usr/bin/env python3
"""Upscale ornament image to 2048×2048."""

import shutil
import subprocess
from pathlib import Path

IMG = Path("ornament_ceramic_engaged4-keep-the-background-remove-the-ornament.jpg")
bin_path = shutil.which("upscayl-bin")
MODEL_DIR = Path("/Users/crn/.local/share/upscayl-bin-20251207-174704-macos/models")

OUT = IMG.with_name(f"{IMG.stem}_2048.jpg")
TARGET = 2048

print(f"Upscaling to {TARGET}px → {OUT.name}")
subprocess.run(
    [
        "upscayl-bin",
        "-i",
        str(IMG),
        "-o",
        str(OUT.with_suffix(".png")),
        "-m",
        str(MODEL_DIR),
        "-n",
        "high-fidelity-4x",
        "-w",
        str(TARGET),
        "-c",
        "85",
        "-f",
        "jpg",
    ],
    check=True,
)
print(f"Done: {OUT.name}")
