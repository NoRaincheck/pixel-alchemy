---
name: upscale-pipeline
description: Batch-upscale folders of images with a two-pass upscayl pipeline (high-fidelity detail pass, then ultrasharp sharpening pass) followed by Lanczos resize to an exact target width. Use when enlarging or sharpening images for print/web, e.g. "upscale these images to 8000px".
---

# Upscale Pipeline

Two-pass upscayl batch pipeline, generalized from the `run_pipeline.py` scripts
in `baby/`, `image-interior/`, and friends:

1. Pass 1 — `high-fidelity-4x` at scale 2–4 (auto-chosen from target width).
2. Pass 2 — `ultrasharp-4x` scale=2 (sharpening pass).
3. Lanczos downscale to exactly `--width` px (aspect preserved).

Skips images whose output already exists, so it is resumable. Writes a
`pipeline_report.json` next to the images.

## Usage

```bash
# From repo root (so pixel_alchemy is importable), or anywhere with it installed
uv run .agents/skills/upscale-pipeline/upscale_pipeline.py baby/title-2026-08-14 --width 7000

uv run .agents/skills/upscale-pipeline/upscale_pipeline.py DIR --width 8000 --suffix _enhanced --format jpg
uv run .agents/skills/upscale-pipeline/upscale_pipeline.py DIR --width 3840 --pass1-model digital-art-4x --no-pass2
```

## Options

| option | default | meaning |
|---|---|---|
| `DIR` | `.` | folder of `.jpg/.jpeg/.png` images to process in place |
| `--width N` | required | exact final width in px |
| `--pass1-model M` | `high-fidelity-4x` | detail pass model (`digital-art-4x` for illustrations) |
| `--pass2-model M` | `ultrasharp-4x` | sharpening pass model |
| `--no-pass2` | off | single-pass mode |
| `--suffix S` | `_enhanced` | output becomes `<stem><suffix>.jpg` |
| `--quality Q` | `95` | JPEG quality of outputs |
| `--workers N` | `3` | concurrent upscayl processes |

Requires `upscayl-bin` on PATH and the repo's `pixel_alchemy` package.
