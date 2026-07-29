"""Corner flood + BiRefNet foreground protection example.

Demonstrates combining the fast corner-flood background removal with
BiRefNet foreground detection so that subject matter is preserved.
The flood-fill tolerance is dynamically chosen per-image so that
no more than a configurable fraction of foreground is lost.

See corner_flood_birefnet.md for the companion documentation.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pixel_alchemy.background_removal.birefnet import birefnet

CORNER_PATCH_SIZE = 5


# ──────────────────────────────────────────────────────────────────────
# Corner sampling + flood fill (adapted from corner_flood.py)
# ──────────────────────────────────────────────────────────────────────


def _sample_corners(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """Sample and average a small patch from each corner, return as RGB tuples."""
    h, w = arr.shape[:2]
    half = CORNER_PATCH_SIZE // 2
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    result = []
    for y, x in corners:
        y_lo, y_hi = max(0, y - half), min(h, y + half + 1)
        x_lo, x_hi = max(0, x - half), min(w, x + half + 1)
        avg = arr[y_lo:y_hi, x_lo:x_hi].mean(axis=(0, 1)).astype(int)
        result.append((int(avg[0]), int(avg[1]), int(avg[2])))
    return result


def _corner_color(corners: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Pick the most common corner color (majority vote)."""
    return Counter(corners).most_common(1)[0][0]


def _compute_diff_and_seeds(img: Image.Image):
    """Pre-compute the Euclidean distance map from background color and flood fill seeds."""
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    bg_color = _corner_color(_sample_corners(arr))

    diff = np.sqrt(
        (r - bg_color[0]) ** 2 + (g - bg_color[1]) ** 2 + (b - bg_color[2]) ** 2
    )
    h, w = arr.shape[:2]
    seeds = [(1, 1), (h + 1, 1), (1, w + 1), (h + 1, w + 1)]
    return diff, seeds


def _flood_fill_bg(diff, seeds, tolerance, w, h):
    """Threshold diff at tolerance, flood fill from seeds, return boolean bg mask."""
    bg_mask = (diff <= tolerance).astype(np.uint8) * 255
    bordered = cv2.copyMakeBorder(bg_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    for by, bx in seeds:
        by, bx = min(by, h), min(bx, w)
        if bordered[by, bx] == 255:
            cv2.floodFill(bordered, None, (bx, by), 0)
    return bordered[1:-1, 1:-1] == 0


# ──────────────────────────────────────────────────────────────────────
# BiRefNet foreground mask
# ──────────────────────────────────────────────────────────────────────


def get_birefnet_mask(img: Image.Image, threshold: int = 128) -> np.ndarray:
    """Run BiRefNet and return a boolean foreground mask."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.convert("RGB").save(tmp.name)
        mask_path = birefnet(tmp.name, foreground=False)
    mask = Image.open(mask_path).convert("L")
    Path(tmp.name).unlink()
    mask_path.unlink()
    return np.array(mask) > threshold


# ──────────────────────────────────────────────────────────────────────
# Core technique: dynamic-tolerance flood fill with foreground protection
# ──────────────────────────────────────────────────────────────────────


def remove_background_protected(
    img: Image.Image,
    fg_mask: np.ndarray | None = None,
    max_fg_loss: float = 0.10,
) -> Image.Image:
    """Remove background via corner flood fill with dynamic tolerance.

    When fg_mask is provided, binary searches for the highest tolerance
    (1–50) that does not remove more than max_fg_loss fraction of the
    BiRefNet-identified foreground.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]

    diff, seeds = _compute_diff_and_seeds(img)

    if fg_mask is None:
        is_bg = _flood_fill_bg(diff, seeds, 3, w, h)
    else:
        fg_count = int(fg_mask.sum())
        if fg_count == 0:
            is_bg = _flood_fill_bg(diff, seeds, 50, w, h)
        else:
            lo, hi = 1, 50
            best_tol = lo
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = _flood_fill_bg(diff, seeds, mid, w, h)
                loss = (candidate & fg_mask).sum() / fg_count
                if loss <= max_fg_loss:
                    best_tol = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            is_bg = _flood_fill_bg(diff, seeds, best_tol, w, h)
            fg_loss = (is_bg & fg_mask).sum() / fg_count * 100
            print(f"  Tolerance: {best_tol}  (fg loss: {fg_loss:.1f}%)")

    arr[:, :, 3] = np.where(is_bg, 0, 255)
    return Image.fromarray(arr, "RGBA")


# ──────────────────────────────────────────────────────────────────────
# Example: single image
# ──────────────────────────────────────────────────────────────────────


def process_image(
    input_path: Path,
    output_path: Path,
    max_fg_loss: float = 0.10,
) -> None:
    """Full pipeline: BiRefNet → protected flood fill."""
    img = Image.open(input_path)

    print(f"Running BiRefNet on {input_path.name}...")
    fg_mask = get_birefnet_mask(img)
    fg_pct = fg_mask.sum() / fg_mask.size * 100
    print(f"  Foreground: {fg_pct:.1f}% of pixels")

    result = remove_background_protected(img, fg_mask=fg_mask, max_fg_loss=max_fg_loss)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "PNG")
    print(f"  Saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Example: batch directory
# ──────────────────────────────────────────────────────────────────────


def process_directory(
    input_dir: Path,
    output_dir: Path,
    max_fg_loss: float = 0.10,
) -> None:
    """Process all PNGs in a directory."""
    png_files = sorted(input_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files → {output_dir}/")

    for i, path in enumerate(png_files, 1):
        print(f"\n[{i}/{len(png_files)}]")
        out = output_dir / path.name
        process_image(path, out, max_fg_loss=max_fg_loss)


if __name__ == "__main__":
    img_path = Path(__file__).with_name("hello-summer.jpg")
    out_path = Path(__file__).with_name("test_birefnet_protected.png")
    process_image(img_path, out_path)
