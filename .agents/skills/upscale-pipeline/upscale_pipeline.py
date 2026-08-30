#!/usr/bin/env python3
"""Batch two-pass upscayl pipeline: detail pass, sharpening pass, Lanczos to target width."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
from difflib import get_close_matches
from pathlib import Path
from typing import Literal

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _model_dir() -> Path:
    binary = shutil.which("upscayl-bin")
    if not binary:
        raise SystemExit("upscayl-bin not found on PATH")
    models = Path(binary).resolve().parent / "models"
    if not models.is_dir():
        raise SystemExit(f"models dir not found: {models}")
    return models


def upscayl(input_path: Path, output_path: Path, *, model: str, scale: Literal[2, 3, 4]) -> None:
    model_dir = _model_dir()
    available = [p.stem for p in model_dir.glob("*.bin")]
    if model not in available:
        close = get_close_matches(model, available)
        hint = f"did you mean: {close}" if close else f"available: {sorted(available)}"
        raise SystemExit(f"model:{model} not in allowable models, {hint}")
    subprocess.run(
        ["upscayl-bin", "-i", str(input_path), "-o", str(output_path), "-m", str(model_dir), "-n", model, "-s", str(scale)],
        check=True,
    )


def choose_scale(current_width: int, target_width: int) -> Literal[2, 3, 4]:
    needed = round(target_width / current_width)
    if needed >= 4:
        return 4
    if needed == 3:
        return 3
    return 2


def process(img_path: Path, args) -> dict | None:
    out_dir = args.directory / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (img_path.stem + args.suffix + ".jpg")
    if out_path.exists():
        print(f"Skipping (output exists): {img_path.name}")
        return None

    with Image.open(img_path) as img:
        orig_w, orig_h = img.size
    orig_kb = img_path.stat().st_size // 1024
    print(f"Processing: {img_path.name} ({orig_w}x{orig_h})")

    scale = choose_scale(orig_w, args.width)
    tmp = out_dir / (img_path.stem + "_tmp")
    print(f"  Pass 1: upscayl {args.pass1_model} (scale={scale})")
    upscayl(img_path, tmp.with_suffix(".png"), model=args.pass1_model, scale=scale)

    current = tmp.with_suffix(".png")
    if not args.no_pass2:
        print(f"  Pass 2: upscayl {args.pass2_model} (scale=2)")
        upscayl(current, tmp.with_suffix("_p2.png"), model=args.pass2_model, scale=2)
        current.unlink()
        current = tmp.with_suffix("_p2.png")

    with Image.open(current) as img:
        new_h = round(args.width * img.height / img.width)
        result = img.convert("RGB").resize((args.width, new_h), Image.LANCZOS)  # ty:ignore[unresolved-attribute]
    result.save(out_path, "JPEG", quality=args.quality, optimize=True)

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
    parser.add_argument("--output-dir", default="upscaled", help="output subfolder name")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (0-100)")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    skip = ("_pass1", "_pass2", args.suffix, "_tmp", "_p2")
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
