#!/usr/bin/env python3
"""Compose front/back cover images onto templates, split at a magenta spine strip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def find_spine_left(template: Image.Image) -> int:
    """Left edge of the magenta spine strip, or the horizontal center as fallback."""
    row = np.array(template.convert("RGB"))[template.height // 2]
    mask = (row[:, 0] > 200) & (row[:, 1] < 50) & (row[:, 2] > 200)
    return int(np.where(mask)[0][0]) if mask.any() else template.width // 2


def make_cover(template: Image.Image, front: Image.Image, back: Image.Image) -> Image.Image:
    tw, th = template.size
    spine_left = find_spine_left(template)

    def fill(img: Image.Image, w: int, h: int) -> Image.Image:
        return img.resize((w, h), Image.LANCZOS)  # ty:ignore[unresolved-attribute]

    cover = template.copy()
    cover.paste(fill(back.convert("RGB"), spine_left, th), (0, 0))
    cover.paste(fill(front.convert("RGB"), tw - spine_left, th), (spine_left, 0))
    return cover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="JSON list of {front, back} entries")
    parser.add_argument("--templates", type=Path, required=True, help="folder of template PNGs")
    parser.add_argument("--output", type=Path, default=Path("covers"))
    args = parser.parse_args()

    entries = json.loads(args.metadata.read_text())
    templates = sorted(args.templates.glob("*.png"))
    if not entries or not templates:
        raise SystemExit("Need at least one metadata entry and one template PNG")

    base = args.metadata.parent
    args.output.mkdir(exist_ok=True)
    print(f"Generating {len(entries) * len(templates)} covers")

    for entry in entries:
        front_path, back_path = base / entry["front"], base / entry["back"]
        if not front_path.exists() or not back_path.exists():
            print(f"  SKIP: missing image(s) for {entry}")
            continue
        for tmpl in templates:
            cover = make_cover(Image.open(tmpl), Image.open(front_path), Image.open(back_path))
            out = args.output / f"{front_path.stem}_{tmpl.stem}.png"
            cover.save(out, "PNG")
            print(f"  {out.name}")


if __name__ == "__main__":
    main()
