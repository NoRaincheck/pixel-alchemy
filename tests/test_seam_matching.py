from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel_alchemy.seam_matching.composite import compose, find_best_fit

DATA = Path(__file__).parent / "seam_matching"


def _load(name: str) -> np.ndarray:
    return np.array(Image.open(DATA / f"{name}.png").convert("RGB"))


@pytest.mark.parametrize(
    ("segment", "expected"),
    [("variant-input", (30, 123)), ("variant2-input", (70, 155))],
)
def test_find_best_fit(segment: str, expected: tuple[int, int]) -> None:
    x, y, score = find_best_fit(_load("base"), _load(segment))
    assert (x, y) == expected
    assert score > 0.8


@pytest.mark.parametrize(
    ("method", "tolerance"),
    [("paste", 0.5), ("feather", 2.0)],
)
@pytest.mark.parametrize(
    ("segment", "ground_truth"),
    [("variant-input", "variant"), ("variant2-input", "variant2")],
)
def test_compose_matches_reference(segment: str, ground_truth: str, method: str, tolerance: float) -> None:
    out, _ = compose(_load("base"), _load(segment), method=method)
    reference = _load(ground_truth)
    assert np.abs(out.astype(int) - reference.astype(int)).mean() < tolerance


def test_compose_finds_position_and_blends() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    segment = base[40:100, 50:110].copy()
    segment[20:40, 20:40] = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)

    out, (x, y) = compose(base, segment, method="paste")

    assert (x, y) == (50, 40)
    assert np.array_equal(
        out[40:100, 50:110][segment == base[40:100, 50:110]], base[40:100, 50:110][segment == base[40:100, 50:110]]
    )
    assert not np.array_equal(out[60:80, 70:90], base[60:80, 70:90])


def test_user_provided_mask() -> None:
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    segment = base.copy()
    segment[30:50, 40:60] = 0
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:50, 40:60] = True

    out, _ = compose(base, segment, mask=mask, method="paste")

    assert (out[30:50, 40:60] == 0).all()


def test_segment_overhangs_base_is_cropped() -> None:
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    segment = np.full((40, 60, 3), 100, dtype=np.uint8)

    out, (x, y) = compose(base, segment, x=130, y=100)

    assert out.shape == base.shape
    assert (x, y) == (130, 100)


def test_segment_larger_than_base() -> None:
    base = np.zeros((10, 10, 3), dtype=np.uint8)
    segment = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        find_best_fit(base, segment)
