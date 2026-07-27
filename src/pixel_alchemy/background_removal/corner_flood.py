from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CORNER_PATCH_SIZE = 5


def _sample_corners(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """Sample and average a small patch from each corner, return as RGB tuples."""
    h, w = arr.shape[:2]
    half = CORNER_PATCH_SIZE // 2
    corners = [
        (0, 0),  # top-left
        (0, w - 1),  # top-right
        (h - 1, 0),  # bottom-left
        (h - 1, w - 1),  # bottom-right
    ]
    result = []
    for y, x in corners:
        y_lo = max(0, y - half)
        y_hi = min(h, y + half + 1)
        x_lo = max(0, x - half)
        x_hi = min(w, x + half + 1)
        patch = arr[y_lo:y_hi, x_lo:x_hi]
        avg = patch.mean(axis=(0, 1)).astype(int)
        result.append((int(avg[0]), int(avg[1]), int(avg[2])))
    return result


def _corner_color(corners: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Pick the most common corner color (majority vote)."""
    counts = Counter(corners)
    return counts.most_common(1)[0][0]


def _corners_match(color: tuple[int, int, int], corners: list[tuple[int, int, int]], tolerance: float) -> bool:
    """Check if all corners are within tolerance of the given color."""
    for c in corners:
        dist = np.linalg.norm(np.array(color, dtype=np.float32) - np.array(c, dtype=np.float32))
        if dist > tolerance:
            return False
    return True


def _flood_fill_mask(arr: np.ndarray, color: tuple[int, int, int], tolerance: float) -> np.ndarray:
    """Flood fill from all 4 corners, return combined binary mask (255 = filled)."""
    h, w = arr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    # Convert Euclidean tolerance to per-channel tolerance for OpenCV
    per_channel = tolerance / np.sqrt(3)

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

    for y, x in corners:
        # Fresh copy for each corner — floodFill modifies the image in-place
        img_copy = arr.copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(
            img_copy,
            mask,
            (x, y),
            newVal=(0, 0, 0),
            loDiff=(per_channel, per_channel, per_channel),
            upDiff=(per_channel, per_channel, per_channel),
            flags=cv2.FLOODFILL_MASK_ONLY,
        )
        combined |= (mask[1:-1, 1:-1] > 0).astype(np.uint8)

    return combined * 255


def _force_match_corners(arr: np.ndarray, corners: list[tuple[int, int, int]]) -> float:
    """Binary search for the smallest tolerance where all 4 corners flood-fill as the same region."""
    dominant = _corner_color(corners)
    lo, hi = 0.0, 441.67  # max Euclidean distance in RGB (255*sqrt(3))

    # Check if already matched at max tolerance
    if not _corners_match(dominant, corners, hi):
        return hi

    for _ in range(20):  # ~0.001 precision
        mid = (lo + hi) / 2
        if _corners_match(dominant, corners, mid):
            hi = mid
        else:
            lo = mid
    return hi


def corner_flood(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    tolerance: int = 30,
    force_match: bool = False,
    feather: int = 0,
    foreground: bool = False,
) -> Path:
    """Remove background by flood-filling from the four image corners.

    Samples the four corners of the image. If colors are roughly the same,
    flood-fills from all four corners and generates a mask removing those colors.

    Args:
        input_path: Path to the input image.
        output_path: Where to write the result. Defaults to <input>_mask.png or <input>_foreground.png.
        tolerance: Max RGB Euclidean distance (0-255) for a pixel to match the corner color.
            Ignored when force_match is True.
        force_match: When True, automatically tunes the tolerance so all four corners match.
        feather: Gaussian blur radius for soft mask edges. 0 = no feathering.
        foreground: If True, output the original image with background removed (RGBA PNG).

    Returns:
        The resolved output path as a pathlib.Path.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    image = Image.open(input_path).convert("RGB")
    arr = np.array(image, dtype=np.float32)
    corners = _sample_corners(arr)

    if force_match:
        tol = _force_match_corners(arr, corners)
    else:
        tol = float(tolerance)

    dominant = _corner_color(corners)
    mask_arr = _flood_fill_mask(arr, dominant, tol)

    # Feather the mask if requested
    if feather > 0:
        ksize = feather * 2 + 1
        mask_arr = cv2.GaussianBlur(mask_arr, (ksize, ksize), 0)

    mask = Image.fromarray(mask_arr, mode="L")

    if output_path is None:
        suffix = "_foreground.png" if foreground else "_mask.png"
        output_path = input_path.with_name(f"{input_path.stem}{suffix}")
    else:
        output_path = Path(output_path)

    if foreground:
        result = image.copy()
        # Invert: flood fill marks background as 255, alpha needs 255=opaque (foreground)
        alpha = Image.fromarray(255 - np.array(mask), mode="L")
        result.putalpha(alpha)
        result.save(output_path)
    else:
        mask.save(output_path)

    return output_path
