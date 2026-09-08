---
name: background-removal
description: Remove backgrounds from images using BiRefNet ONNX via onnxruntime. Produces transparent PNGs, alpha mattes, and comparison previews. Use when isolating subjects, creating transparent cutouts, or generating alpha masks for compositing.
---

# Background Removal (BiRefNet)

BiRefNet (Bilateral Reference Network) ONNX inference via `onnxruntime` + OpenCV postprocessing. Sigmoid on logits + resize to original resolution for the alpha matte.

Model: `models/birefnet.onnx` (local, ~490 MB). OpenCV's DNN backend cannot run this model (ENGINE_NEW segfaults, ENGINE_CLASSIC lacks FP16 support) — onnxruntime is used instead.

## Usage

```bash
./remove_background.py image.jpg                          # -> outputs/ with transparent + alpha + comparison
./remove_background.py image.jpg -o ./out                 # custom output dir
./remove_background.py photos/ -o ./cutouts               # batch: every image in a directory
./remove_background.py a.jpg b.jpg c.png --no-comparison  # multiple explicit files
./remove_background.py image.jpg --model /path/birefnet.onnx --size 1024
```

## Options

| option | default | meaning |
|---|---|---|
| `INPUT...` | required | one or more image files and/or directories |
| `-o, --output-dir DIR` | `outputs/` next to project root (or `./outputs` fallback) | where to write results |
| `--model PATH` | `models/birefnet.onnx` | path to BiRefNet ONNX file |
| `--size N` | `1024` | model input size (square) |
| `--no-alpha` | off | skip writing `*_alpha.png` matte |
| `--no-comparison` | off | skip writing `*_comparison.jpg` preview |

## Outputs (per input image)

| file | contents |
|---|---|
| `<stem>_transparent.png` | BGRA image with alpha from BiRefNet |
| `<stem>_alpha.png` | grayscale alpha matte (unless `--no-alpha`) |
| `<stem>_comparison.jpg` | side-by-side `[original \| mask \| on white]` (unless `--no-comparison`) |

## Requirements

- `onnxruntime`, `opencv-python-headless`, `numpy`
- Model file at `models/birefnet.onnx` (or pass `--model`)

## Notes

- Preprocessing: `blobFromImage` with `1/255`, `swapRB`, ImageNet `mean=[0.485,0.456,0.406]` / `std=[0.229,0.224,0.225]`, sigmoid `1/(1+exp(-logits))`.
- CPU only (`CPUExecutionProvider`) by default; add CUDA provider if available.
