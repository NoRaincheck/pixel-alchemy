"""ControlNet lineart / HED + OpenCV dodge-blend pencil sketch.

Thin wrapper around ``controlnet_aux.LineartDetector`` and ``HEDdetector``
with a PIL / OpenCV-friendly API matching ``adaattn.py``, plus a
lightweight pure-OpenCV fallback (no model download) via the classic
grayscale-invert-blur-dodge trick.

Example:
    from pixel_alchemy.style_transfer.lineart import lineart, dodge_sketch
    from PIL import Image
    img = Image.open("wallpaper/wallpaper_enhanced.jpg")
    sketch = lineart(img, coarse=False)  # fine detail, white-on-black
    sketch_inverted = lineart(img, invert=True)  # black-on-white
    pencil = dodge_sketch(img, blur_kernel=21)  # pure OpenCV, no AI

    # path one-liner
    from pixel_alchemy.style_transfer.lineart import lineart_paths
    lineart_paths("wallpaper/wallpaper_enhanced.jpg", "wallpaper/lineart.jpg")

CLI:
    uv run python -m pixel_alchemy.style_transfer.lineart wallpaper/wallpaper_enhanced.jpg -o wallpaper/stylized/lineart.jpg
    uv run python -m pixel_alchemy.style_transfer.lineart wallpaper/wallpaper_enhanced.jpg --coarse --detect-resolution 768
    uv run python -m pixel_alchemy.style_transfer.lineart wallpaper/wallpaper_enhanced.jpg --detector hed -o hed.jpg
    uv run python -m pixel_alchemy.style_transfer.lineart wallpaper/wallpaper_enhanced.jpg --detector dodge -o pencil.jpg
    uv run python -m pixel_alchemy.style_transfer.lineart wallpaper/wallpaper_enhanced.jpg --invert  # black-on-white
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageOps

DEFAULT_ANNOTATOR = "lllyasviel/Annotators"
DEFAULT_HED = "lllyasviel/Annotators"


def _resolve_device(device: str | torch.device | None) -> str:
    if device is not None:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_pil(img: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img).convert("RGB")
        # assume BGR if 3ch from cv2, else RGB
        # heuristic: if ndarray from cv2.imread it's BGR; we treat as BGR
        # caller should pass RGB PIL when ambiguous; here convert BGR->RGB
        if img.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return Image.fromarray(img).convert("RGB")
    p = Path(img)
    if not p.exists():
        raise FileNotFoundError(p)
    return Image.open(p).convert("RGB")


# -- model cache -----------------------------------------------------------

_LINEART_CACHE: dict[tuple[str, str], object] = {}
_HED_CACHE: dict[str, object] = {}


def get_lineart_detector(
    pretrained: str = DEFAULT_ANNOTATOR,
    device: str | torch.device | None = None,
):
    """Load (and cache) LineartDetector."""
    from controlnet_aux import LineartDetector

    key = (pretrained, _resolve_device(device))
    if key in _LINEART_CACHE:
        return _LINEART_CACHE[key]
    det = LineartDetector.from_pretrained(pretrained)
    det.to(_resolve_device(device))
    _LINEART_CACHE[key] = det
    return det


def get_hed_detector(
    pretrained: str = DEFAULT_HED,
    device: str | torch.device | None = None,
):
    from controlnet_aux import HEDdetector

    key = _resolve_device(device) + ":" + pretrained
    if key in _HED_CACHE:
        return _HED_CACHE[key]
    det = HEDdetector.from_pretrained(pretrained)
    det.to(_resolve_device(device))
    _HED_CACHE[key] = det
    return det


# -- functional API --------------------------------------------------------

def lineart(
    image: str | Path | Image.Image | np.ndarray,
    *,
    coarse: bool = False,
    detect_resolution: int = 512,
    image_resolution: int | None = None,
    device: str | torch.device | None = None,
    output: str | Path | None = None,
    pretrained: str = DEFAULT_ANNOTATOR,
    invert: bool = False,
    dodge: bool = True,
    blur_kernel: int = 21,
) -> Image.Image:
    """Generate lineart sketch from an image.

    By default combines AI lineart (controlnet_aux) with the OpenCV
    dodge-blend pencil trick for softer, more realistic shading
    (pro-tip from instructions). Set dodge=False for pure AI edges.

    Args:
        image: PIL / path / BGR ndarray.
        coarse: True = thick sketchy lines, False = fine detailed.
        detect_resolution: model input size (512, 768, 1024).
        image_resolution: output size (defaults to input long-edge;
            pass 512/1024 to match controlnet_aux default, or None to
            keep input resolution).
        device: torch device (auto if None).
        output: if set, save to path.
        pretrained: HF repo for Annotators weights.
        invert: if True, black-on-white (print-friendly); else white-on-black.
        dodge: if True (default), multiply AI lineart with dodge_sketch for
            realistic shading; set False for pure AI.
        blur_kernel: dodge Gaussian size (21 default).
    """
    pil = _to_pil(image)
    ow, oh = pil.size
    if image_resolution is None:
        image_resolution = max(ow, oh)
        # cap huge wallpapers to avoid OOM; 2048 is plenty for lineart
        image_resolution = min(image_resolution, 2048)

    det = get_lineart_detector(pretrained, device)
    out_wob = det(
        pil,
        coarse=coarse,
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
        output_type="pil",
    )
    # resize back to original if we capped
    if out_wob.size != (ow, oh):
        out_wob = out_wob.resize((ow, oh), Image.LANCZOS)
    out_wob = out_wob.convert("RGB")

    if dodge:
        # blend in black-on-white space for realistic pencil
        dodge_bow = dodge_sketch(pil, blur_kernel=blur_kernel, invert=False)
        if dodge_bow.size != (ow, oh):
            dodge_bow = dodge_bow.resize((ow, oh), Image.LANCZOS)
        ai_bow = ImageOps.invert(out_wob)
        combined_bow = ImageChops.multiply(ai_bow, dodge_bow)
        out = combined_bow if invert else ImageOps.invert(combined_bow)
    else:
        out = ImageOps.invert(out_wob) if invert else out_wob

    if output is not None:
        o = Path(output)
        o.parent.mkdir(parents=True, exist_ok=True)
        out.save(o)
    return out


def hed(
    image: str | Path | Image.Image | np.ndarray,
    *,
    detect_resolution: int = 512,
    image_resolution: int | None = None,
    device: str | torch.device | None = None,
    output: str | Path | None = None,
    pretrained: str = DEFAULT_HED,
    safe: bool = False,
    scribble: bool = False,
    invert: bool = False,
    dodge: bool = True,
    blur_kernel: int = 21,
) -> Image.Image:
    pil = _to_pil(image)
    ow, oh = pil.size
    if image_resolution is None:
        image_resolution = min(max(ow, oh), 2048)

    det = get_hed_detector(pretrained, device)
    out_wob = det(
        pil,
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
        safe=safe,
        scribble=scribble,
        output_type="pil",
    )
    if out_wob.size != (ow, oh):
        out_wob = out_wob.resize((ow, oh), Image.LANCZOS)
    out_wob = out_wob.convert("RGB")

    if dodge:
        dodge_bow = dodge_sketch(pil, blur_kernel=blur_kernel, invert=False)
        if dodge_bow.size != (ow, oh):
            dodge_bow = dodge_bow.resize((ow, oh), Image.LANCZOS)
        ai_bow = ImageOps.invert(out_wob)
        combined_bow = ImageChops.multiply(ai_bow, dodge_bow)
        out = combined_bow if invert else ImageOps.invert(combined_bow)
    else:
        out = ImageOps.invert(out_wob) if invert else out_wob
    if output is not None:
        o = Path(output)
        o.parent.mkdir(parents=True, exist_ok=True)
        out.save(o)
    return out


def dodge_sketch(
    image: str | Path | Image.Image | np.ndarray,
    *,
    blur_kernel: int = 21,
    invert: bool = False,
    output: str | Path | None = None,
) -> Image.Image:
    """Pure-OpenCV pencil sketch via grayscale invert-blur dodge blend.

    No model download; ~10 ms on 1k image. Classic trick:
        gray = RGB->GRAY
        inv = 255 - gray
        blurred = GaussianBlur(inv, k, k)
        sketch = divide(gray, 255 - blurred, scale=256)

    Args:
        image: PIL / path / BGR ndarray.
        blur_kernel: Gaussian kernel size (odd, 21 default — larger = softer).
        invert: if True, return white-on-black (inverted) to match lineart default.
        output: if set, save to path.

    Returns:
        PIL RGB image (grayscale sketch in 3 channels).
    """
    pil = _to_pil(image)
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blur_kernel = max(3, blur_kernel)
    inv = 255 - gray
    blurred = cv2.GaussianBlur(inv, (blur_kernel, blur_kernel), 0)
    # dodge blend: gray / (255 - blurred) * 256
    denom = 255 - blurred
    # avoid division by zero — cv2.divide handles 0 as 0, but clip
    sketch_gray = cv2.divide(gray, denom, scale=256)
    sketch_gray = np.clip(sketch_gray, 0, 255).astype(np.uint8)
    if invert:
        sketch_gray = 255 - sketch_gray
    out = Image.fromarray(sketch_gray).convert("RGB")
    if output is not None:
        o = Path(output)
        o.parent.mkdir(parents=True, exist_ok=True)
        out.save(o)
    return out


def lineart_paths(
    input_path: str | Path,
    output_path: str | Path,
    *,
    coarse: bool = False,
    detect_resolution: int = 512,
    image_resolution: int | None = None,
    device: str | torch.device | None = None,
    invert: bool = False,
    dodge: bool = True,
    blur_kernel: int = 21,
) -> Path:
    out = lineart(
        input_path,
        coarse=coarse,
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
        device=device,
        invert=invert,
        dodge=dodge,
        blur_kernel=blur_kernel,
    )
    o = Path(output_path)
    o.parent.mkdir(parents=True, exist_ok=True)
    out.save(o)
    return o


# -- CLI -------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="ControlNet lineart / HED / OpenCV dodge via controlnet_aux (combined by default)")
    ap.add_argument("input", type=Path, help="Input image")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--detector", choices=["lineart", "hed", "dodge"], default="lineart", help="lineart/HED need controlnet_aux; dodge is pure OpenCV; lineart/hed are combined with dodge by default")
    ap.add_argument("--coarse", action="store_true", help="Lineart coarse (thick) mode")
    ap.add_argument("--invert", action="store_true", help="Invert colors (black-on-white instead of white-on-black; dodge default is black-on-white, so --invert gives white-on-black)")
    ap.add_argument("--no-dodge", action="store_true", help="Disable dodge-blend combination (pure AI edges only)")
    ap.add_argument("--blur-kernel", type=int, default=21, help="Dodge Gaussian kernel size (odd, default 21)")
    ap.add_argument("--detect-resolution", type=int, default=512)
    ap.add_argument("--image-resolution", type=int, default=None, help="Output resolution (default: input size capped at 2048)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--pretrained", default=None)
    return ap.parse_args()


def main():
    args = _parse_args()
    dodge = not args.no_dodge
    out = args.output
    if out is None:
        suffix = "coarse" if args.coarse and args.detector == "lineart" else args.detector
        if dodge and args.detector in ("lineart", "hed"):
            suffix += "-dodge"
        if args.invert:
            suffix += "-inverted"
        out = args.input.with_name(f"{args.input.stem}_lineart-{suffix}{args.input.suffix}")
    if args.detector == "hed":
        hed(args.input, detect_resolution=args.detect_resolution, image_resolution=args.image_resolution, device=args.device, pretrained=args.pretrained or DEFAULT_HED, invert=args.invert, dodge=dodge, blur_kernel=args.blur_kernel, output=out)
    elif args.detector == "dodge":
        dodge_sketch(args.input, blur_kernel=args.blur_kernel, invert=args.invert, output=out)
    else:
        lineart(args.input, coarse=args.coarse, detect_resolution=args.detect_resolution, image_resolution=args.image_resolution, device=args.device, pretrained=args.pretrained or DEFAULT_ANNOTATOR, invert=args.invert, dodge=dodge, blur_kernel=args.blur_kernel, output=out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
