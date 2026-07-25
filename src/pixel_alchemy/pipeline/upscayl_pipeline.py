from __future__ import annotations

import math
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter

from pixel_alchemy.super_resolution.upscayl import upscayl


def _upscale_step(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    scale: int,
) -> Path:
    return upscayl(input_path, output_path, model=model, scale=scale)


def _blur_and_downscale(
    img: Image.Image,
    radius: float,
    target_width: int,
) -> Image.Image:
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    cur_w, cur_h = blurred.size
    new_h = round(target_width * cur_h / cur_w)
    return blurred.resize((target_width, new_h), Image.LANCZOS)  # ty:ignore[unresolved-attribute]


def upscayl_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    target_width: int = 2048,
    blur_multipliers: list[float] = [5, 3, 1],
    model: str = "upscayl-standard-4x",
    scale: int = 4,
) -> Path:
    """Multi-pass pipeline: upscayl → blur → downscale, repeated per blur_multipliers.

    Each pass upscales via upscayl, applies Gaussian blur with radius m * log(scale),
    then downscales to target_width via Lanczos. Output of pass N feeds into pass N+1.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_pipelined.png"
    output_path = Path(output_path)

    log_factor = math.log(scale)
    radii = [m * log_factor for m in blur_multipliers]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        current_input = input_path

        for i, radius in enumerate(radii, 1):
            upscaled = tmp / f"pass{i}_upscaled.png"
            _upscale_step(current_input, upscaled, model=model, scale=scale)

            img = Image.open(upscaled)
            result = _blur_and_downscale(img, radius, target_width)

            if i < len(radii):
                intermediate = tmp / f"pass{i}.png"
                result.save(intermediate, "PNG", optimize=True)
                current_input = intermediate
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(output_path, "PNG", optimize=True)

    return output_path
