from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from PIL import Image

MODEL_REPO = "onnx-community/BiRefNet-ONNX"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ort.set_default_logger_severity(3)  # suppress onnxruntime warnings


def _model_path(fp16: bool) -> Path:
    model_file = "model_fp16.onnx" if fp16 else "model.onnx"
    try:
        local_dir = snapshot_download(
            MODEL_REPO,
            allow_patterns=[f"onnx/{model_file}"],
            local_files_only=True,
        )
    except FileNotFoundError:
        local_dir = snapshot_download(
            MODEL_REPO,
            allow_patterns=[f"onnx/{model_file}"],
        )
    return Path(local_dir) / "onnx" / model_file


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((1024, 1024), Image.BICUBIC)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis]


def _predict(input_data: np.ndarray, model_path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]


def _postprocess(output: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
    mask = 1 / (1 + np.exp(-output[0]))
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask.squeeze(), mode="L")
    mask = mask.resize(original_size, Image.BICUBIC)
    return mask


def birefnet(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    fp16: bool = True,
    foreground: bool = False,
) -> Path:
    """Generate an alpha matte / segmentation mask for an image using BiRefNet ONNX.

    Args:
        input_path: Path to the input image.
        output_path: Where to write the result. Defaults to <input>_mask.png or <input>_foreground.png.
        fp16: Use the FP16 model (smaller, ~490MB). Set to False for FP32 (~973MB).
        foreground: If True, apply mask to original image and output foreground with transparent background (RGBA PNG).

    Returns:
        The resolved output path as a pathlib.Path.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    image = Image.open(input_path).convert("RGB")
    original_size = image.size

    model_path = _model_path(fp16)

    input_data = _preprocess(image)
    output = _predict(input_data, model_path)
    mask = _postprocess(output, original_size)

    if output_path is None:
        suffix = "_foreground.png" if foreground else "_mask.png"
        output_path = input_path.with_name(f"{input_path.stem}{suffix}")
    else:
        output_path = Path(output_path)

    if foreground:
        result = image.copy()
        result.putalpha(mask)
        result.save(output_path)
    else:
        mask.save(output_path)

    return output_path
