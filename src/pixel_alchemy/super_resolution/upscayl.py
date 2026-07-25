from __future__ import annotations

import shutil
from difflib import get_close_matches
from pathlib import Path
from typing import Literal

import sh


def _default_model_dir() -> Path:
    binary = Path(shutil.which("upscayl-bin") or "")
    if not binary.exists():
        raise FileNotFoundError("upscayl-bin not found on PATH")
    return binary.resolve().parent / "models"


def _allowable_models() -> list[str]:
    model_dir = _default_model_dir()
    return [x.stem for x in model_dir.glob("*.bin")]


def upscayl(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    model: str = "upscayl-standard-4x",
    model_dir: str | Path | None = None,
    scale: Literal[2, 3, 4] = 4,
    format: Literal["png", "jpg"] | None = None,
    width: int | None = None,
    compress: int = 0,
    enable_tta: bool = False,
) -> Path:
    """Upscale an image using upscayl-bin.

    Args:
        input_path: Path to the input image.
        output_path: Where to write the result.  Defaults to ``<input>_upscayled.<ext>``.
        model: Model name passed to ``-n``, typical options are 'upscayl-standard-4x' 'upscayl-lite-4x'.
        model_dir: Directory containing model files.  Defaults to the ``models``
            folder next to the resolved ``upscayl-bin`` binary.
        scale: Upscale factor (``-s``).
        format: Output format override (``-f``).
        width: resize output to a width (default=W:default)
        compress: a number between 0-100, compression of the output image, typically used with jpg
        enable_tta: When True, uses Test Time Augmentation, which may remove artifacts but increases processing by 8x

    Returns:
        The resolved output path as a :class:`pathlib.Path`.
    """
    if model not in (allowable := _allowable_models()):
        raise ValueError(f"model:{model}, not in allowable models, did you mean: {get_close_matches(model, allowable)}")
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if format is None and output_path is not None:
        inferred = Path(output_path).suffix[1:]
        assert inferred in ("png", "jpg")
        format = inferred

    if output_path is None and format:
        stem = f"{input_path.stem}_upscayled"
        output_path = input_path.with_name(stem).with_suffix(f".{format}")
    elif output_path is not None:
        output_path = Path(output_path)
        assert output_path.suffix[1:] == format, f"{output_path.suffix[1:]}, {format}"
    else:
        stem = f"{input_path.stem}_upscayled"
        output_path = input_path.with_name(stem).with_suffix(input_path.suffix)

    if format == "png" and compress > 0:
        raise ValueError("If png is selected, no compression is applied, change format to jpg for compression")

    resolved_model_dir = Path(model_dir) if model_dir else _default_model_dir()

    args: list[str] = ["-i", str(input_path), "-o", str(output_path)]
    args += ["-m", str(resolved_model_dir)]
    args += ["-n", model]
    args += ["-s", str(scale)]
    if format is not None:
        args += ["-f", format]
    if enable_tta:
        args.append("-x")
    if format == "jpg" and compress > 0:
        args += ["-c", str(compress)]
    if width:
        args += ["-w", str(width)]

    sh.upscayl_bin(*args)
    return output_path
