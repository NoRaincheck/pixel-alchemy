#!/usr/bin/env python3
"""Batch two-pass upscayl pipeline: detail pass, sharpening pass, Lanczos to target width."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Literal

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from pixel_alchemy.super_resolution.upscayl import upscayl
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from pixel_alchemy.super_resolution.upscayl import upscayl


def choose_scale(current_width: int, target_width: int) -> Literal[2, 3, 4]:
    needed = round(target_width / current_width)
    if needed >= 4:
        return 4
    if needed == 3:
        return 3
    return 2


def process(img_path: Path, args) -> dict | None:
    out_path = img_path.with_name(img_path.stem + args.suffix + ".jpg")
    if out_path.exists():
        print(f"Skipping (output exists): {img_path.name}")
        return None

    with Image.open(img_path) as img:
        orig_w, orig_h = img.size
    orig_kb = img_path.stat().st_size // 1024
    print(f"Processing: {img_path.name} ({orig_w}x{orig_h})")

    scale = choose_scale(orig_w, args.width)
    pass1 = img_path.with_name(img_path.stem + "_pass1.png")
    print(f"  Pass 1: upscayl {args.pass1_model} (scale={scale})")
    upscayl(img_path, pass1, model=args.pass1_model, scale=scale)

    current = pass1
    if not args.no_pass2:
        pass2 = img_path.with_name(img_path.stem + "_pass2.png")
        print(f"  Pass 2: upscayl {args.pass2_model} (scale=2)")
        upscayl(pass1, pass2, model=args.pass2_model, scale=2)
        current = pass2

    with Image.open(current) as img:
        new_h = round(args.width * img.height / img.width)
        result = img.convert("RGB").resize((args.width, new_h), Image.LANCZOS)  # ty:ignore[unresolved-attribute]
    result.save(out_path, "JPEG", quality=args.quality, optimize=True)

    pass1.unlink(missing_ok=True)
    current.unlink(missing_ok=True)

    final_kb = out_path.stat().st_size // 1024
    print(f"  -> {out_path.name} ({args.width}x{new_h}, {final_kb} KB)")
    return {
        "file": img_path.name,
        "original": f"{orig_w}x{orig_h}",
        "original_size_kb": orig_kb,
        "final": f"{args.width}x{new_h}",
        "final_size_kb": final_kb,
        "width_scale": round(args.width / orig_w, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".", type=Path)
    parser.add_argument("--width", type=int, required=True, help="exact target width in px")
    parser.add_argument("--pass1-model", default="high-fidelity-4x")
    parser.add_argument("--pass2-model", default="ultrasharp-4x")
    parser.add_argument("--no-pass2", action="store_true", help="skip the sharpening pass")
    parser.add_argument("--suffix", default="_enhanced")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    skip = ("_pass1", "_pass2", args.suffix)
    images = [
        p
        for p in sorted(args.directory.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not any(s in p.stem for s in skip)
    ]
    if not images:
        raise SystemExit(f"No images found in {args.directory}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [r for r in pool.map(lambda p: process(p, args), images) if r]

    report_path = args.directory / "pipeline_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nDone: {len(results)} images. Report: {report_path}")


if __name__ == "__main__":
    main()
