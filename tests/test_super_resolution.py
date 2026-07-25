from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pixel_alchemy.super_resolution.upscayl import upscayl

MODELS_DIR = Path("/fake/upscayl/models")

ALLOWABLE = ["upscayl-standard-4x", "upscayl-lite-4x", "ultramix-balanced-4x"]


@pytest.fixture(autouse=True)
def _patch_model_dir() -> Generator[None]:
    with (
        patch("pixel_alchemy.super_resolution.upscayl._default_model_dir", return_value=MODELS_DIR),
        patch("pixel_alchemy.super_resolution.upscayl._allowable_models", return_value=ALLOWABLE),
    ):
        yield


@pytest.fixture()
def fake_image(tmp_path: Path) -> Path:
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\x00" * 10)
    return img


@patch("pixel_alchemy.super_resolution.upscayl.sh.upscayl_bin", new_callable=MagicMock)
def test_defaults(mock_bin: MagicMock, fake_image: Path) -> None:
    result = upscayl(fake_image)
    mock_bin.assert_called_once_with(
        "-i",
        str(fake_image),
        "-o",
        str(fake_image.with_name("photo_upscayled.jpg")),
        "-m",
        str(MODELS_DIR),
        "-n",
        "upscayl-standard-4x",
        "-s",
        "4",
    )
    assert result == fake_image.with_name("photo_upscayled.jpg")


@patch("pixel_alchemy.super_resolution.upscayl.sh.upscayl_bin", new_callable=MagicMock)
def test_custom_output(mock_bin: MagicMock, fake_image: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.png"
    result = upscayl(fake_image, out)
    assert result == out
    mock_bin.assert_called_once_with(
        "-i",
        str(fake_image),
        "-o",
        str(out),
        "-m",
        str(MODELS_DIR),
        "-n",
        "upscayl-standard-4x",
        "-s",
        "4",
        "-f",
        "png",
    )


@patch("pixel_alchemy.super_resolution.upscayl.sh.upscayl_bin", new_callable=MagicMock)
def test_save_as_png(mock_bin: MagicMock, fake_image: Path) -> None:
    result = upscayl(fake_image, format="png")
    assert result.suffix == ".png"
    args = mock_bin.call_args.args
    assert "-f" in args
    assert "png" in args


@patch("pixel_alchemy.super_resolution.upscayl.sh.upscayl_bin", new_callable=MagicMock)
def test_format_override(mock_bin: MagicMock, fake_image: Path) -> None:
    upscayl(fake_image, format="jpg")
    args = mock_bin.call_args.args
    assert "-f" in args
    assert "jpg" in args


@patch("pixel_alchemy.super_resolution.upscayl.sh.upscayl_bin", new_callable=MagicMock)
def test_custom_model_and_scale(mock_bin: MagicMock, fake_image: Path) -> None:
    upscayl(fake_image, model="ultramix-balanced-4x", scale=2)
    args = mock_bin.call_args.args
    assert args == (
        "-i",
        str(fake_image),
        "-o",
        str(fake_image.with_name("photo_upscayled.jpg")),
        "-m",
        str(MODELS_DIR),
        "-n",
        "ultramix-balanced-4x",
        "-s",
        "2",
    )


@patch("pixel_alchemy.super_resolution.upscayl.sh.upscayl_bin", new_callable=MagicMock)
def test_custom_model_dir(mock_bin: MagicMock, fake_image: Path, tmp_path: Path) -> None:
    custom = tmp_path / "my-models"
    upscayl(fake_image, model_dir=custom)
    args = mock_bin.call_args.args
    assert "-m" in args
    assert args[args.index("-m") + 1] == str(custom)


def test_missing_input() -> None:
    with pytest.raises(FileNotFoundError):
        upscayl(Path("/no/such/file.png"))
