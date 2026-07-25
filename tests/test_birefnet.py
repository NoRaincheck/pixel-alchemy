from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pixel_alchemy.background_removal.birefnet import (
    _postprocess,
    _preprocess,
    birefnet,
)


@pytest.fixture()
def fake_image(tmp_path: Path) -> Path:
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    path = tmp_path / "photo.jpg"
    img.save(path)
    return path


def test_preprocess_shape() -> None:
    image = Image.new("RGB", (400, 300))
    result = _preprocess(image)
    assert result.shape == (1, 3, 1024, 1024)
    assert result.dtype == np.float32


def test_preprocess_normalization() -> None:
    image = Image.new("RGB", (100, 100), color=(128, 128, 128))
    result = _preprocess(image)
    pixel = result[0, :, 0, 0]
    expected = (np.array([128, 128, 128], dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    np.testing.assert_allclose(pixel, expected, atol=1e-6)


def test_postprocess_shape() -> None:
    output = np.random.randn(1, 1024, 1024).astype(np.float32)
    mask = _postprocess(output, (400, 300))
    assert mask.size == (400, 300)
    assert mask.mode == "L"


def test_postprocess_sigmoid_range() -> None:
    output = np.zeros((1, 1024, 1024), dtype=np.float32)
    mask = _postprocess(output, (100, 100))
    arr = np.array(mask)
    # sigmoid(0) = 0.5 -> 127.5, rounded to uint8
    assert arr.min() >= 127
    assert arr.max() <= 128


@patch("pixel_alchemy.background_removal.birefnet._predict")
@patch("pixel_alchemy.background_removal.birefnet._model_path")
def test_defaults(mock_model_path: MagicMock, mock_predict: MagicMock, fake_image: Path) -> None:
    mock_model_path.return_value = fake_image.parent / "models" / "model_fp16.onnx"
    mock_predict.return_value = np.zeros((1, 1024, 1024), dtype=np.float32)

    result = birefnet(fake_image)
    assert result == fake_image.with_name("photo_mask.png")
    assert result.suffix == ".png"


@patch("pixel_alchemy.background_removal.birefnet._predict")
@patch("pixel_alchemy.background_removal.birefnet._model_path")
def test_custom_output(mock_model_path: MagicMock, mock_predict: MagicMock, fake_image: Path, tmp_path: Path) -> None:
    mock_model_path.return_value = fake_image.parent / "models" / "model_fp16.onnx"
    mock_predict.return_value = np.zeros((1, 1024, 1024), dtype=np.float32)

    out = tmp_path / "result.png"
    result = birefnet(fake_image, out)
    assert result == out


@patch("pixel_alchemy.background_removal.birefnet._predict")
@patch("pixel_alchemy.background_removal.birefnet._model_path")
def test_fp32_model(mock_model_path: MagicMock, mock_predict: MagicMock, fake_image: Path) -> None:
    mock_model_path.return_value = Path("/fake/models/model.onnx")
    mock_predict.return_value = np.zeros((1, 1024, 1024), dtype=np.float32)

    birefnet(fake_image, fp16=False)
    mock_model_path.assert_called_once_with(False)
    assert mock_predict.call_args[0][1].name == "model.onnx"


@patch("pixel_alchemy.background_removal.birefnet._predict")
@patch("pixel_alchemy.background_removal.birefnet._model_path")
def test_fp16_model(mock_model_path: MagicMock, mock_predict: MagicMock, fake_image: Path) -> None:
    mock_model_path.return_value = Path("/fake/models/model_fp16.onnx")
    mock_predict.return_value = np.zeros((1, 1024, 1024), dtype=np.float32)

    birefnet(fake_image)
    mock_model_path.assert_called_once_with(True)
    assert mock_predict.call_args[0][1].name == "model_fp16.onnx"


@patch("pixel_alchemy.background_removal.birefnet._predict")
@patch("pixel_alchemy.background_removal.birefnet._model_path")
def test_foreground_default_name(mock_model_path: MagicMock, mock_predict: MagicMock, fake_image: Path) -> None:
    mock_model_path.return_value = fake_image.parent / "models" / "model_fp16.onnx"
    mock_predict.return_value = np.zeros((1, 1024, 1024), dtype=np.float32)

    result = birefnet(fake_image, foreground=True)
    assert result == fake_image.with_name("photo_foreground.png")


@patch("pixel_alchemy.background_removal.birefnet._predict")
@patch("pixel_alchemy.background_removal.birefnet._model_path")
def test_foreground_output_is_rgba(mock_model_path: MagicMock, mock_predict: MagicMock, fake_image: Path) -> None:
    mock_model_path.return_value = fake_image.parent / "models" / "model_fp16.onnx"
    mock_predict.return_value = np.random.randn(1, 1024, 1024).astype(np.float32)

    result = birefnet(fake_image, foreground=True)
    img = Image.open(result)
    assert img.mode == "RGBA"
    assert img.size == (200, 100)


@patch("pixel_alchemy.background_removal.birefnet._predict")
@patch("pixel_alchemy.background_removal.birefnet._model_path")
def test_mask_output_is_grayscale(mock_model_path: MagicMock, mock_predict: MagicMock, fake_image: Path) -> None:
    mock_model_path.return_value = fake_image.parent / "models" / "model_fp16.onnx"
    mock_predict.return_value = np.random.randn(1, 1024, 1024).astype(np.float32)

    result = birefnet(fake_image)
    img = Image.open(result)
    assert img.mode == "L"


def test_missing_input() -> None:
    with pytest.raises(FileNotFoundError):
        birefnet(Path("/no/such/file.png"))
