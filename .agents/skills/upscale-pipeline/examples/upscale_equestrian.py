#!/usr/bin/env python3
"""Upscale equestrian image to 4k–8k in one pass each."""

import shutil
import subprocess
from pathlib import Path

IMG = Path("equestrian_training_horse_log_book_double_spread_5.725x9_300dpi.png")
bin_path = shutil.which("upscayl-bin")
MODEL_DIR = Path("/Users/crn/.local/share/upscayl-bin-20251207-174704-macos/models")


WIDTHS = [3840, 5120, 6144, 7168, 7680]
SUFFIXES = ["4k", "5k", "6k", "7k", "8k"]

for w, s in zip(WIDTHS, SUFFIXES):
    out = IMG.with_name(f"{IMG.stem}_{s}.jpg")
    print(f"Upscaling to {w}px width → {out.name}")
    subprocess.run(
        [
            "upscayl-bin",
            "-i",
            str(IMG),
            "-o",
            str(out.with_suffix(".png")),
            "-m",
            str(MODEL_DIR),
            "-n",
            "high-fidelity-4x",
            "-w",
            str(w),
            "-c",
            "85",
            "-f",
            "jpg",
        ],
        check=True,
    )
    print(f"  Done: {out.name}")

print("All done!")
