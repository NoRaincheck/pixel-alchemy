#!/usr/bin/env python3
"""BiRefNet background removal via onnxruntime + OpenCV."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort

MODEL_SIZE = 1024
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_default_model() -> Path:
    # skill lives at .agents/skills/background-removal/remove_background.py -> project root is 4 levels up
    candidates = [
        Path(__file__).resolve().parents[3] / "models" / "birefnet.onnx",
        Path(__file__).resolve().parent.parent.parent.parent / "models" / "birefnet.onnx",
        Path.cwd() / "models" / "birefnet.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def collect_images(inputs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for item in inputs:
        if item.is_dir():
            for p in sorted(item.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                    images.append(p)
        elif item.is_file():
            images.append(item)
        else:
            print(f"Skipping (not found): {item}")
    return images


def remove_background(
    image_path: Path,
    sess: ort.InferenceSession,
    input_name: str,
    output_dir: Path,
    *,
    size: int = MODEL_SIZE,
    write_alpha: bool = True,
    write_comparison: bool = True,
) -> Path:
    img = cv.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")
    h, w = img.shape[:2]

    blob = cv.dnn.blobFromImage(
        img, scalefactor=1.0 / 255.0, size=(size, size), mean=IMAGE_NET_MEAN, swapRB=True, crop=False
    )
    blob[0, 0] /= IMAGE_NET_STD[0]
    blob[0, 1] /= IMAGE_NET_STD[1]
    blob[0, 2] /= IMAGE_NET_STD[2]

    logits = sess.run(None, {input_name: blob})[0]
    alpha = 1.0 / (1.0 + np.exp(-logits[0, 0]))
    alpha = cv.resize(alpha, (w, h), interpolation=cv.INTER_LINEAR)
    alpha = (alpha * 255).astype(np.uint8)

    bgra = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    transparent = output_dir / f"{stem}_transparent.png"
    cv.imwrite(str(transparent), bgra)

    if write_alpha:
        cv.imwrite(str(output_dir / f"{stem}_alpha.png"), alpha)

    if write_comparison:
        alpha_3ch = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR)
        white_bg = np.ones_like(img, dtype=np.uint8) * 255
        mask_f = alpha.astype(np.float32)[:, :, np.newaxis] / 255.0
        result_on_white = (img.astype(np.float32) * mask_f + white_bg * (1.0 - mask_f)).astype(np.uint8)
        comparison = np.hstack(
            [
                cv.resize(img, (400, 400)),
                cv.resize(alpha_3ch, (400, 400)),
                cv.resize(result_on_white, (400, 400)),
            ]
        )
        cv.imwrite(str(output_dir / f"{stem}_comparison.jpg"), comparison)

    return transparent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="image files and/or directories")
    parser.add_argument("-o", "--output-dir", type=Path, help="output directory (default: ./outputs)")
    parser.add_argument("--model", type=Path, default=find_default_model(), help="path to birefnet.onnx")
    parser.add_argument("--size", type=int, default=MODEL_SIZE, help="model input size (default: 1024)")
    parser.add_argument("--no-alpha", action="store_true", help="skip alpha matte output")
    parser.add_argument("--no-comparison", action="store_true", help="skip comparison.jpg output")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}\nExpected at models/birefnet.onnx")

    images = collect_images(args.inputs)
    if not images:
        raise SystemExit("No images found")

    output_dir = args.output_dir or (Path.cwd() / "outputs")
    # if single directory input and no explicit output, keep default; else use cwd/outputs
    # also support project-root fallback when running from skill dir
    if args.output_dir is None:
        root_outputs = Path(__file__).resolve().parents[3] / "outputs"
        if root_outputs.parent.exists():
            output_dir = root_outputs

    print(f"Loading BiRefNet model from {args.model}")
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"  Input: {input_name} {sess.get_inputs()[0].shape}  size={args.size}")

    for image_path in images:
        print(f"\n{image_path}")
        t0 = time.time()
        out = remove_background(
            image_path,
            sess,
            input_name,
            output_dir,
            size=args.size,
            write_alpha=not args.no_alpha,
            write_comparison=not args.no_comparison,
        )
        print(f"  -> {out}  ({time.time() - t0:.2f}s)")

    print(f"\nDone: {len(images)} image(s) -> {output_dir}/")


if __name__ == "__main__":
    main()
