"""BiRefNet + vtracer iterative expansion example.

Uses BiRefNet for foreground detection, then expands the mask outward
via iterative dilation + vtracer spline smoothing until it reaches
within 1px of the image edge. Outputs an RGBA image with transparent
background.

Dilation step scales dynamically with image size (dilation_pct).
No shape priors — relies purely on edge detection for the stopping
condition, making it work for any subject shape.

See remove_bg_loose.md for companion documentation.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import cv2
import numpy as np
import vtracer
from PIL import Image

from pixel_alchemy.background_removal.birefnet import birefnet


def remove_watermark(img: Image.Image) -> Image.Image:
    """Remove the NotebookLM watermark from the bottom right corner."""
    w, h = img.size
    corner = img.crop((int(w * 0.85), int(h * 0.95), w, h))
    arr = np.array(corner)
    dark = np.sum((arr[:, :, 0] < 200).astype(int))
    if dark / arr.size < 0.02:
        return img
    result = img.copy()
    result.paste((255, 255, 255), (int(w * 0.82), int(h * 0.92), w, h))
    return result


def get_birefnet_mask(img: Image.Image) -> np.ndarray:
    """Run BiRefNet and return the raw grayscale mask (uint8, 0-255)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.convert("RGB").save(tmp.name)
        mask_path = birefnet(tmp.name, foreground=False)
    mask = Image.open(mask_path).convert("L")
    os.unlink(tmp.name)
    os.unlink(str(mask_path))
    return np.array(mask)


def _mask_touches_edge(mask: np.ndarray, margin: int = 1) -> bool:
    """Check if any foreground pixel is within margin pixels of the image border."""
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    border[:margin, :] = True
    border[h - margin :, :] = True
    border[:, :margin] = True
    border[:, w - margin :] = True
    return bool((mask & border).any())


def _vtracer_smooth(binary_small: np.ndarray, tmp_dir: str) -> np.ndarray:
    """Vectorize a binary mask with vtracer, render back, return boolean mask."""
    h, w = binary_small.shape
    in_path = os.path.join(tmp_dir, "iter_input.png")
    svg_path = os.path.join(tmp_dir, "iter_output.svg")
    out_path = os.path.join(tmp_dir, "iter_output.png")

    Image.fromarray(binary_small, mode="L").save(in_path)
    vtracer.convert_image_to_svg_py(in_path, svg_path)
    subprocess.run(["magick", svg_path, "-resize", f"{w}x{h}!", out_path], check=True)

    rendered = np.array(Image.open(out_path).convert("L"))
    return rendered > 128


def expand_mask_to_edge(
    img: Image.Image,
    target_width: int = 500,
    dilation_pct: float = 0.04,
    max_iters: int = 50,
) -> np.ndarray:
    """Iteratively dilate + vtracer-smooth until the mask reaches the image edge.

    Args:
        dilation_pct: Dilation step as a fraction of the smaller dimension.
            Ensures the kernel scales with image size.

    Returns a boolean mask at the original image resolution.
    """
    h, w = img.size[1], img.size[0]
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    tight_mask = get_birefnet_mask(img)
    binary = (tight_mask > 128).astype(np.uint8) * 255

    scale = target_width / w
    small_w = target_width
    small_h = int(h * scale)
    dilation_step = max(3, int(min(small_w, small_h) * dilation_pct) | 1)
    small = (cv2.resize(binary, (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255

    tight_pct = (small > 0).sum() / small.size * 100
    print(f"  Tight mask: {tight_pct:.1f}%  ({small_w}x{small_h})")

    for i in range(max_iters):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_step, dilation_step))
        small = cv2.dilate(small, kernel, iterations=1)
        small = _vtracer_smooth(small, tmp_dir).astype(np.uint8) * 255
        if _mask_touches_edge(small > 0):
            pct = (small > 0).sum() / small.size * 100
            print(f"  Iter {i + 1}: hit edge at {pct:.1f}%")
            break
    else:
        pct = (small > 0).sum() / small.size * 100
        print(f"  Max iterations ({max_iters}) reached at {pct:.1f}%")

    final = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST) > 128
    full_pct = final.sum() / final.size * 100
    print(f"  Final: {full_pct:.1f}% foreground at {w}x{h}")
    return final


def process_image(input_path: str, out_dir: str = "output") -> None:
    """Full pipeline: watermark removal -> BiRefNet -> vtracer expansion -> RGBA output."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(input_path))
    if os.path.exists(out_path):
        print(f"Skipping (already done): {os.path.basename(input_path)}")
        return
    print(f"Processing: {os.path.basename(input_path)}")
    img = Image.open(input_path)
    img = remove_watermark(img)
    mask = expand_mask_to_edge(img)
    img_rgba = img.convert("RGBA")
    img_rgba.putalpha(Image.fromarray((mask * 255).astype(np.uint8), mode="L"))
    img_rgba.save(out_path, "PNG")
    print(f"  Done -> {out_path}")


if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), "hello-summer.jpg")
    process_image(img_path, out_dir=os.path.join(os.path.dirname(__file__), "output"))
