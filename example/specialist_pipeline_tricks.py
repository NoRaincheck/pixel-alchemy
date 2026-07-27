"""Specialist pipeline tricks — a cookbook of reusable image-processing patterns.

Each trick is self-contained and can be adapted to your own pipeline.
These patterns emerged from real-world batch processing workflows and
cover upscaling, sharpening, inpainting, and batch orchestration.

See specialist_pipeline_tricks.md for the companion documentation.
"""

from __future__ import annotations

import asyncio
import math
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None  # allow images larger than the default limit


# ──────────────────────────────────────────────────────────────────────
# Trick 1: Skip already-processed files
# ──────────────────────────────────────────────────────────────────────

SKIP_PATTERNS = {"_enhanced", "_pass1", "_pass2", "_upscayled", "_pipelined"}


def is_original(path: Path) -> bool:
    """Check if a file is an original (not yet processed).

    Uses a set of known output suffixes to filter out files that have
    already been through a pipeline pass.  Extend the set as needed.
    """
    stem = path.stem.lower()
    return not any(skip in stem for skip in SKIP_PATTERNS)


def already_processed(path: Path, output_suffix: str = "_pass2.png") -> bool:
    """Return True if the output for this input already exists on disk."""
    return (path.with_name(path.stem + output_suffix)).exists()


# ──────────────────────────────────────────────────────────────────────
# Trick 2: Adaptive scale factor selection
# ──────────────────────────────────────────────────────────────────────


def choose_scale(width: int, target_width: int = 2000) -> int:
    """Pick the best scale factor (2, 3, or 4) to reach target_width.

    Instead of always using 4x (which overshoots for already-large images),
    pick the smallest scale that gets you close to the target.  This saves
    VRAM and time when the input is already >500px wide.

    """
    needed = target_width / width
    return min(max(round(needed), 2), 4)


# ──────────────────────────────────────────────────────────────────────
# Trick 3: Pre-downscale to destroy JPEG artifacts
# ──────────────────────────────────────────────────────────────────────


def downscale_destroy_artifacts(
    img: Image.Image,
    target_width: int = 1920,
) -> Image.Image:
    """Downscale with Lanczos to destroy JPEG compression blocks.

    JPEG artifacts (8x8 block boundaries, ringing around edges) are
    amplified by AI upscalers.  Pre-downscaling to a moderate resolution
    (e.g. 1920px) with Lanczos resampling smears the blocks together,
    giving the AI a clean canvas.

    """
    w, h = img.size
    new_h = round(target_width * h / w)
    return img.resize((target_width, new_h), Image.LANCZOS)


# ──────────────────────────────────────────────────────────────────────
# Trick 4: Gaussian blur before AI upscale
# ──────────────────────────────────────────────────────────────────────


def blur_before_upscale(
    img: Image.Image,
    radius: float = 1.5,
) -> Image.Image:
    """Apply Gaussian blur to smooth noise before feeding to AI upscaler.

    AI upscalers can hallucinate detail from noise.  A light blur (radius
    1.0–2.0) removes high-frequency noise while preserving structure,
    so the AI generates clean detail instead of amplifying grain.

    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ──────────────────────────────────────────────────────────────────────
# Trick 5: Upscale high → downscale for detail refinement
# ──────────────────────────────────────────────────────────────────────


def upscale_then_refine(
    img_path: Path,
    output_path: Path,
    target_width: int = 3840,
    *,
    model: str = "digital-art-4x",
    upscale_scale: int = 4,
) -> Path:
    """Upscale past the target, then downscale to embed AI detail naturally.

    Key insight: AI upscalers produce their best detail at high resolution.
    If you upscale directly to 4K, you get one pass of detail.  If you
    upscale to 8K (e.g. 1920 → 7680) and then downscale to 4K (3840),
    the AI-generated detail at 8K is "baked in" at a higher quality when
    the Lanczos downscaler subsamples it.

    """
    from pixel_alchemy.super_resolution.upscayl import upscayl

    intermediate_w = target_width // 2  # e.g. 1920
    upscale_target = intermediate_w * upscale_scale  # e.g. 7680

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        upscaled = tmp / "upscaled.png"

        upscayl(img_path, upscaled, model=model, scale=upscale_scale, format="png")

        with Image.open(upscaled) as ai_img:
            w, h = ai_img.size
            new_h = round(target_width * h / w)
            refined = ai_img.resize((target_width, new_h), Image.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        refined.save(output_path, "PNG", optimize=True)

    return output_path


# ──────────────────────────────────────────────────────────────────────
# Trick 6: Two-model sharpening (high-fidelity → ultrasharp)
# ──────────────────────────────────────────────────────────────────────


def two_model_sharpen(
    img_path: Path,
    output_path: Path,
    target_width: int = 2000,
) -> Path:
    """Use high-fidelity model for structure, then ultrasharp for detail.

    Different upscayl models have different strengths:
    - high-fidelity-4x: preserves the original look, faithful structure
    - ultrasharp-4x:    enhances edge detail and sharpness

    Running high-fidelity first (scale=2-4) gives you a faithful base,
    then ultrasharp (scale=2) on top sharpens the details without
    changing the overall composition.

    """
    from pixel_alchemy.super_resolution.upscayl import upscayl

    scale = choose_scale(Image.open(img_path).size[0], target_width)

    # Pass 1: faithful upscale
    pass1 = output_path.parent / f"{output_path.stem}_temp_pass1.png"
    upscayl(img_path, pass1, model="high-fidelity-4x", scale=scale)

    # Pass 2: sharpen
    upscayl(pass1, output_path, model="ultrasharp-4x", scale=2)

    # Resize to exact target
    with Image.open(output_path) as img:
        w, h = img.size
        new_h = round(target_width * h / w)
        img.resize((target_width, new_h), Image.LANCZOS).save(output_path, "PNG", optimize=True)

    pass1.unlink()  # clean up intermediate
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Trick 7: Double-pass unsharp mask
# ──────────────────────────────────────────────────────────────────────


def double_sharpen(img: Image.Image) -> Image.Image:
    """Two rounds of UnsharpMask with different radii for crisp detail.

    A single sharpening pass can only handle one scale of detail.  Using
    two passes — one broad (radius=0.8) and one fine (radius=0.4) —
    sharpens both large edges and fine texture.

    """
    # Broad sharpening: edges and structure
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=2))
    # Fine sharpening: texture and detail
    img = img.filter(ImageFilter.UnsharpMask(radius=0.4, percent=80, threshold=3))
    return img


# ──────────────────────────────────────────────────────────────────────
# Trick 8: Async concurrency with semaphore
# ──────────────────────────────────────────────────────────────────────


async def process_batch_concurrent(
    images: list[Path],
    process_fn,
    max_concurrent: int = 3,
) -> list[dict]:
    """Run a processing function over many images with bounded concurrency.

    Prevents OOM from processing too many images at once.  The semaphore
    limits how many images are being processed simultaneously.

    """
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(img_path: Path):
        async with sem:
            return await process_fn(img_path)

    tasks = [bounded(img) for img in images]
    return [r for r in await asyncio.gather(*tasks) if r is not None]


# ──────────────────────────────────────────────────────────────────────
# Trick 9: Inpainting to fix missing parts
# ──────────────────────────────────────────────────────────────────────


def inpaint_fix(
    init_image: Path,
    mask: Path,
    prompt: str,
    output: Path,
    *,
    diffusion_model: str,
    vae: str,
    llm: str,
    cfg_scale: float = 0.5,
    steps: int = 9,
) -> Path:
    """Use inpainting to regenerate a specific region of an image.

    The prompt should describe what you want in the masked area AND
    mention the surrounding style to ensure consistency:

      "a subject sitting, with details visible, soft watercolor style,
       consistent with the rest of the illustration, accessories"

    Tips:
    - Keep cfg_scale low (0.5–1.0) to avoid artifacts
    - Use 9 steps for inpainting (not too few, not too many)
    - Describe the style context in the prompt for consistency
    - Use PIL to create simple rectangle masks (white=inpaint region)
    """
    from pixel_alchemy.generation.sd_cli import inpaint

    inpaint(
        init_image=init_image,
        mask=mask,
        prompt=prompt,
        diffusion_model=diffusion_model,
        vae=vae,
        llm=llm,
        output=output,
        steps=steps,
        cfg_scale=cfg_scale,
    )
    return output


def make_rectangle_mask(
    image_path: Path,
    mask_path: Path,
    x1_frac: float,
    y1_frac: float,
    x2_frac: float,
    y2_frac: float,
) -> Path:
    """Create a simple rectangle mask from fractional coordinates.

    White area = region to inpaint.  Coordinates are 0.0–1.0 fractions
    of image width/height.

    """
    with Image.open(image_path) as img:
        w, h = img.size

    mask = Image.new("RGB", (w, h), "black")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        [int(w * x1_frac), int(h * y1_frac), int(w * x2_frac), int(h * y2_frac)],
        fill="white",
    )
    mask.save(mask_path)
    return mask_path


# ──────────────────────────────────────────────────────────────────────
# Trick 10: Multi-pass inpainting for shading consistency
# ──────────────────────────────────────────────────────────────────────


def multi_pass_inpaint_shading(
    image: Path,
    output: Path,
    mask: Path,
    prompt_pass1: str,
    prompt_pass2: str,
    *,
    diffusion_model: str,
    vae: str,
    llm: str,
) -> Path:
    """Two-pass inpainting: first fill, then fix shading consistency.

    Pass 1: fill in the missing/wrong region
    Pass 2: subtle shading correction on a targeted sub-region
    """
    from pixel_alchemy.generation.sd_cli import inpaint

    # Pass 1: fill the region
    pass1_out = output.parent / f"{output.stem}_pass1.png"
    inpaint(
        init_image=image,
        mask=mask,
        prompt=prompt_pass1,
        diffusion_model=diffusion_model,
        vae=vae,
        llm=llm,
        output=pass1_out,
        steps=9,
        cfg_scale=1.0,
    )

    # Pass 2: shading fix (optional, use a different mask)
    # Create a narrower mask targeting just the shading area
    with Image.open(pass1_out) as img:
        w, h = img.size
    shade_mask = Image.new("RGB", (w, h), "black")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(shade_mask)
    draw.rectangle([int(w * 0.65), int(h * 0.25), int(w * 0.95), int(h * 0.75)], fill="white")
    shade_mask_path = output.parent / f"{output.stem}_shade_mask.png"
    shade_mask.save(shade_mask_path)

    inpaint(
        init_image=pass1_out,
        mask=shade_mask_path,
        prompt=prompt_pass2,
        diffusion_model=diffusion_model,
        vae=vae,
        llm=llm,
        output=output,
        steps=9,
        cfg_scale=1.0,
    )

    pass1_out.unlink(missing_ok=True)
    return output


# ──────────────────────────────────────────────────────────────────────
# Trick 11: Pipeline report generation
# ──────────────────────────────────────────────────────────────────────


def save_pipeline_report(results: list[dict], report_path: Path) -> None:
    """Save a JSON report with before/after stats for each image.

    Tracks: original dimensions, final dimensions, file sizes, scale
    factors, and the pass sequence used.  Useful for batch auditing.
    """
    import json

    report_path.write_text(json.dumps(results, indent=2))
    print(f"Report saved: {report_path}")


# ──────────────────────────────────────────────────────────────────────
# Trick 12: Multi-pass upscayl pipeline
# ──────────────────────────────────────────────────────────────────────


def multi_pass_upscayl_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    target_width: int = 2048,
    blur_multipliers: list[float] = [5, 3, 1],
    model: str = "upscayl-standard-4x",
    scale: int = 4,
) -> Path:
    """The library's built-in multi-pass pipeline: upscayl → blur → downscale.

    Each pass:
      1. Upscale with upscayl
      2. Apply Gaussian blur (radius = multiplier × log(scale))
      3. Downscale to target_width via Lanczos

    The decreasing blur multipliers [5, 3, 1] mean:
      - Pass 1: heavy blur (aggressive noise removal)
      - Pass 2: medium blur (structural smoothing)
      - Pass 3: light blur (subtle refinement)

    """
    from pixel_alchemy.super_resolution.upscayl import upscayl

    log_factor = math.log(scale)
    radii = [m * log_factor for m in blur_multipliers]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        current_input = input_path

        for i, radius in enumerate(radii, 1):
            upscaled = tmp / f"pass{i}_upscaled.png"
            upscayl(current_input, upscaled, model=model, scale=scale)

            img = Image.open(upscaled)
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
            cur_w, cur_h = blurred.size
            new_h = round(target_width * cur_h / cur_w)
            result = blurred.resize((target_width, new_h), Image.LANCZOS)

            if i < len(radii):
                intermediate = tmp / f"pass{i}.png"
                result.save(intermediate, "PNG", optimize=True)
                current_input = intermediate
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(output_path, "PNG", optimize=True)

    return output_path


# ──────────────────────────────────────────────────────────────────────
# Summary of model choices
# ──────────────────────────────────────────────────────────────────────

"""
Model selection cheat sheet:

┌──────────────────────┬────────────────────────────────────────────┐
│ Use case             │ Model / strategy                           │
├──────────────────────┼────────────────────────────────────────────┤
│ Large JPEGs with     │ Downscale → blur → digital-art-4x (4x)    │
│ compression artifacts│ → downscale → double-sharpen               │
├──────────────────────┼────────────────────────────────────────────┤
│ Mid-size images,     │ high-fidelity-4x (scale 2-4) →            │
│ faithful upscale     │ ultrasharp-4x (scale 2) → resize          │
├──────────────────────┼────────────────────────────────────────────┤
│ Quick general upscale│ upscayl-standard-4x (scale 4)             │
├──────────────────────┼────────────────────────────────────────────┤
│ Missing parts fix    │ Inpaint with style-matching prompt,        │
│                      │ cfg_scale=0.5, steps=9                     │
├──────────────────────┼────────────────────────────────────────────┤
│ Shading consistency  │ Two-pass inpaint: fill → targeted shade    │
├──────────────────────┼────────────────────────────────────────────┤
│ Noise removal before │ Gaussian blur radius 1.0–2.0               │
│ AI upscale           │                                            │
└──────────────────────┴────────────────────────────────────────────┘

Key parameters:
  - cfg_scale: keep at 1.0 for sd-cli (higher causes artifacts)
  - inpainting steps: 9 is the sweet spot
  - blur radius: 1.0–2.0 for pre-upscale smoothing
  - Lanczos for all resize operations (sharpest downscale)
"""
