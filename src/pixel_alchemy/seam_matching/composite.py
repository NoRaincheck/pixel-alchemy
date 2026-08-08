from __future__ import annotations

import heapq
from typing import Literal

import cv2
import numpy as np

NSEW = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def find_best_fit(base: np.ndarray, segment: np.ndarray) -> tuple[int, int, float]:
    """Find the best-fit location for ``segment`` within ``base``.

    Uses normalized cross-correlation template matching, so it is robust to
    the segment containing re-rendered content that does not match ``base``.

    Args:
        base: RGB image the segment should be composed into.
        segment: RGB patch to locate; must be smaller than ``base``.

    Returns:
        (x, y, score) where (x, y) is the top-left corner of the best fit
        and score is the normalized cross-correlation at that location.
    """
    if base.shape[:2] < segment.shape[:2]:
        raise ValueError("segment is larger than base")
    result = cv2.matchTemplate(
        cv2.cvtColor(base, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(segment, cv2.COLOR_RGB2GRAY),
        cv2.TM_CCOEFF_NORMED,
    )
    _, score, _, (x, y) = cv2.minMaxLoc(result)
    return int(x), int(y), float(score)


def compose(
    base: np.ndarray,
    segment: np.ndarray,
    *,
    x: int | None = None,
    y: int | None = None,
    mask: np.ndarray | None = None,
    threshold: float = 40.0,
    seam: bool = True,
    margin: int = 8,
    method: Literal["paste", "feather"] = "paste",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Compose ``segment`` into ``base`` at its best-fit location.

    The segment is registered to the base image (position guessed when not
    given), the region where it actually replaces the base is determined,
    and the two are blended along a seam that follows the cheapest-difference
    pixels.  The segment is cropped if it hangs off the edge of the base.

    Args:
        base: RGB uint8 image to compose into.
        segment: RGB uint8 patch to compose.  An RGBA segment is used for
            its RGB content.
        x: Best-fit x position; found automatically when omitted.
        y: Best-fit y position; found automatically when omitted.
        mask: Optional binary region (same shape as segment) marking where
            the segment replaces the base.  Derived automatically from the
            pixel difference when omitted.
        threshold: Difference at which a pixel is considered modified, used
            when deriving the mask automatically.
        seam: When True, refine the blend region with a minimum-cost seam
            cut around the modified region.
        margin: How far (in pixels) the seam may grow beyond the modified
            region into agreeing pixels.
        method: How to blend -- "paste" (hard), or "feather" (alpha blend
            with a soft, feathered boundary).

    Returns:
        (composite, (x, y)) where composite is the resulting RGB uint8 image
        and (x, y) is the fitted segment position.
    """
    base = np.asarray(base)[:, :, :3]
    segment = np.asarray(segment)[:, :, :3]

    if x is None or y is None:
        x, y, _ = find_best_fit(base, segment)
    h, w = segment.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, base.shape[1]), min(y + h, base.shape[0])
    if x0 >= x1 or y0 >= y1:
        raise ValueError("segment does not overlap base at (x, y)")
    segment = segment[y0 - y : y1 - y, x0 - x : x1 - x]
    if mask is not None:
        mask = mask[y0 - y : y1 - y, x0 - x : x1 - x]
    x, y = x0, y0

    region = base[y : y + segment.shape[0], x : x + segment.shape[1]]
    diff = np.abs(segment.astype(np.float32) - region.astype(np.float32)).sum(axis=2)
    if mask is None:
        region_mask = diff > threshold
    else:
        region_mask = mask[:, :, 0] > 0 if mask.ndim == 3 else mask > 0

    if seam and region_mask.any():
        seam_mask = _seam_cut(diff, region_mask, margin)
    else:
        seam_mask = region_mask.astype(np.uint8) * 255

    return _blend(base, segment, x, y, seam_mask, method), (x, y)


def _blend(
    base: np.ndarray,
    segment: np.ndarray,
    x: int,
    y: int,
    seam_mask: np.ndarray,
    method: Literal["paste", "feather"],
) -> np.ndarray:
    """Paste or alpha-blend ``segment`` into ``base`` under ``seam_mask``."""
    h, w = segment.shape[:2]
    out = base.copy()
    region = out[y : y + h, x : x + w]

    if method == "paste":
        region[...] = np.where(seam_mask[..., None] > 0, segment, region)
        return out

    alpha = cv2.GaussianBlur(seam_mask.astype(np.float32) / 255.0, (0, 0), 8)
    region[...] = (region * (1 - alpha[..., None]) + segment * alpha[..., None]).astype(np.uint8)
    return out


def _seam_cut(cost: np.ndarray, region_mask: np.ndarray, margin: int) -> np.ndarray:
    """Find a minimum-cost seam around ``region_mask`` (adaptation of the
    Scene-Completion ``create_seam_cut``).

    Multi-source Dijkstra over the difference field gives every pixel the
    cheapest cumulative difference needed to reach it from the region.  The
    seam grows the region through pixels reachable at or below the typical
    cost in a surrounding band: it reaches the band where segment and base
    agree (low cost) and stops where they diverge.
    """
    h, w = cost.shape
    dist = np.full((h, w), np.inf)
    heap = [(0.0, int(y), int(x)) for y, x in np.argwhere(region_mask)]
    dist[region_mask] = 0.0
    heapq.heapify(heap)
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dy, dx in NSEW:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            nd = d + cost[ny, nx]
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(heap, (nd, ny, nx))

    kernel = np.ones((3, 3), np.uint8)
    grown = region_mask.astype(np.uint8)
    for _ in range(margin):
        grown = cv2.dilate(grown, kernel)
    band = grown.astype(bool) & ~region_mask
    budget = float(np.quantile(dist[band], 0.5)) if band.any() else 0.0
    seam = (dist <= budget) & band
    return ((seam | region_mask).astype(np.uint8)) * 255
