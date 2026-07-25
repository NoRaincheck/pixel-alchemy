from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel_alchemy.background_removal.corner_flood import (
    _corner_color,
    _corners_match,
    _flood_fill_mask,
    _force_match_corners,
    _sample_corners,
    corner_flood,
)


def _solid_image(size: tuple[int, int] = (200, 100), bg: tuple[int, int, int] = (240, 240, 240)) -> np.ndarray:
    arr = np.full((*size[::-1], 3), bg, dtype=np.float32)
    # Add a distinct foreground rectangle in the center
    arr[25:75, 50:150] = (30, 30, 30)
    return arr


@pytest.fixture()
def solid_bg_image(tmp_path: Path) -> Path:
    arr = _solid_image()
    img = Image.fromarray(arr.astype(np.uint8))
    path = tmp_path / "solid.png"
    img.save(path)
    return path


@pytest.fixture()
def gradient_bg_image(tmp_path: Path) -> Path:
    arr = np.zeros((100, 200, 3), dtype=np.float32)
    # Gradient background — corners won't match
    for x in range(200):
        arr[:, x] = (x / 200 * 255, 100, 50)
    arr[25:75, 50:150] = (30, 30, 30)
    img = Image.fromarray(arr.astype(np.uint8))
    path = tmp_path / "gradient.png"
    img.save(path)
    return path


def test_sample_corners() -> None:
    arr = _solid_image(bg=(100, 150, 200))
    corners = _sample_corners(arr)
    assert len(corners) == 4
    for c in corners:
        assert c == (100, 150, 200)


def test_corner_color_majority() -> None:
    corners = [(10, 10, 10), (10, 10, 10), (10, 10, 10), (200, 200, 200)]
    assert _corner_color(corners) == (10, 10, 10)


def test_corners_match_true() -> None:
    corners = [(100, 100, 100), (102, 102, 102), (98, 98, 98), (101, 101, 101)]
    assert _corners_match((100, 100, 100), corners, tolerance=5.0) is True


def test_corners_match_false() -> None:
    corners = [(100, 100, 100), (200, 200, 200), (50, 50, 50), (150, 150, 150)]
    assert _corners_match((100, 100, 100), corners, tolerance=10.0) is False


def test_flood_fill_mask_solid_bg() -> None:
    arr = _solid_image(bg=(255, 255, 255))
    mask = _flood_fill_mask(arr, (255, 255, 255), tolerance=30.0)
    assert mask.shape == (100, 200)
    # Center foreground should NOT be filled
    assert mask[50, 100] == 0
    # Corners should be filled
    assert mask[0, 0] == 255
    assert mask[99, 199] == 255


def test_flood_fill_mask_no_match() -> None:
    # Image where corners are red but background is blue
    arr = np.full((100, 200, 3), (0, 0, 200), dtype=np.float32)
    # Make corners red
    arr[0, 0] = (200, 0, 0)
    arr[0, 199] = (200, 0, 0)
    arr[99, 0] = (200, 0, 0)
    arr[99, 199] = (200, 0, 0)
    # With tight tolerance on red seed, only the red corner pixel fills
    mask = _flood_fill_mask(arr, (200, 0, 0), tolerance=5.0)
    # Very little should be filled (just the 4 corner pixels)
    assert mask.sum() < 255 * 100 * 200 * 0.01


def test_force_match_corners() -> None:
    corners = [(100, 100, 100), (102, 102, 102), (98, 98, 98), (101, 101, 101)]
    tol = _force_match_corners(_solid_image(bg=(100, 100, 100)), corners)
    # Should find a small tolerance that works
    assert tol < 10.0


def test_solid_bg_default_mask(solid_bg_image: Path) -> None:
    result = corner_flood(solid_bg_image)
    assert result == solid_bg_image.with_name("solid_mask.png")
    img = Image.open(result)
    assert img.mode == "L"
    assert img.size == (200, 100)


def test_solid_bg_mask_content(solid_bg_image: Path) -> None:
    result = corner_flood(solid_bg_image)
    mask = np.array(Image.open(result))
    # Corners should be background (255)
    assert mask[0, 0] == 255
    assert mask[99, 199] == 255
    # Center foreground should be 0
    assert mask[50, 100] == 0


def test_custom_output(solid_bg_image: Path, tmp_path: Path) -> None:
    out = tmp_path / "my_mask.png"
    result = corner_flood(solid_bg_image, out)
    assert result == out
    assert out.exists()


def test_foreground_output(solid_bg_image: Path) -> None:
    result = corner_flood(solid_bg_image, foreground=True)
    assert result == solid_bg_image.with_name("solid_foreground.png")
    img = Image.open(result)
    assert img.mode == "RGBA"
    assert img.size == (200, 100)


def test_foreground_alpha(solid_bg_image: Path) -> None:
    result = corner_flood(solid_bg_image, foreground=True)
    img = Image.open(result)
    arr = np.array(img)
    # Background corners should have alpha = 0
    assert arr[0, 0, 3] == 0
    assert arr[99, 199, 3] == 0
    # Foreground center should have alpha = 255
    assert arr[50, 100, 3] == 255


def test_feather(tmp_path: Path) -> None:
    # Create a very sharp binary image: white background, black rectangle
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    arr[25:75, 50:150] = 0
    path = tmp_path / "sharp.png"
    Image.fromarray(arr).save(path)

    out_plain = tmp_path / "mask_plain.png"
    out_feathered = tmp_path / "mask_feathered.png"
    result_plain = corner_flood(path, out_plain, feather=0, tolerance=1)
    result_feathered = corner_flood(path, out_feathered, feather=10, tolerance=1)
    mask_plain = np.array(Image.open(result_plain))
    mask_feathered = np.array(Image.open(result_feathered))
    # Feathered mask should have more unique values (gradations from blur)
    assert len(np.unique(mask_feathered)) > len(np.unique(mask_plain))


def test_force_match_adapts(gradient_bg_image: Path) -> None:
    # Gradient image — corners don't match, force_match should still produce a mask
    result = corner_flood(gradient_bg_image, force_match=True)
    img = Image.open(result)
    assert img.mode == "L"
    assert img.size == (200, 100)


def test_missing_input() -> None:
    with pytest.raises(FileNotFoundError):
        corner_flood(Path("/no/such/file.png"))


def test_tolerance_parameter(solid_bg_image: Path) -> None:
    # Very tight tolerance should fill less
    result_tight = corner_flood(solid_bg_image, tolerance=1)
    result_loose = corner_flood(solid_bg_image, tolerance=100)
    mask_tight = np.array(Image.open(result_tight))
    mask_loose = np.array(Image.open(result_loose))
    assert mask_tight.sum() <= mask_loose.sum()
