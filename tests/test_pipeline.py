from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pixel_alchemy.pipeline.upscayl_pipeline import _blur_and_downscale, upscayl_pipeline

SMALL_PNG = Image.new("RGB", (4, 4), color="blue")


def _write_mock_upscayl(inp: Path, out: Path, **kw: object) -> Path:
    SMALL_PNG.save(out, "PNG")
    return out


@pytest.fixture()
def fake_image(tmp_path: Path) -> Path:
    img = tmp_path / "photo.png"
    SMALL_PNG.save(img, "PNG")
    return img


@patch("pixel_alchemy.pipeline.upscayl_pipeline.upscayl", side_effect=_write_mock_upscayl)
def test_defaults(mock_upscayl: MagicMock, fake_image: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.png"
    result = upscayl_pipeline(fake_image, out)
    assert result == out
    assert mock_upscayl.call_count == 3


@patch("pixel_alchemy.pipeline.upscayl_pipeline.upscayl", side_effect=_write_mock_upscayl)
def test_default_output_name(mock_upscayl: MagicMock, fake_image: Path) -> None:
    result = upscayl_pipeline(fake_image)
    assert result.name == "photo_pipelined.png"


@patch("pixel_alchemy.pipeline.upscayl_pipeline.upscayl", side_effect=_write_mock_upscayl)
def test_custom_blur_multipliers(mock_upscayl: MagicMock, fake_image: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.png"
    upscayl_pipeline(fake_image, out, blur_multipliers=[3, 1])
    assert mock_upscayl.call_count == 2


@patch("pixel_alchemy.pipeline.upscayl_pipeline.upscayl", side_effect=_write_mock_upscayl)
def test_custom_model_and_scale(mock_upscayl: MagicMock, fake_image: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.png"
    upscayl_pipeline(fake_image, out, model="upscayl-lite-4x", scale=2)
    for call in mock_upscayl.call_args_list:
        assert call.kwargs["model"] == "upscayl-lite-4x"
        assert call.kwargs["scale"] == 2


def test_missing_input() -> None:
    with pytest.raises(FileNotFoundError):
        upscayl_pipeline(Path("/no/such/file.png"))


def test_blur_and_downscale() -> None:
    img = Image.new("RGB", (400, 200), color="red")
    result = _blur_and_downscale(img, radius=2.0, target_width=100)
    assert result.size[0] == 100
    assert result.size[1] == 50
