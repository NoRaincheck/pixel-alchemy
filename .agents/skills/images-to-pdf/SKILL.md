---
name: images-to-pdf
description: Combine folders of images (PNG/JPEG) into print-ready PDFs at a given page size and DPI, either one PDF per folder or one PDF per subfolder. Use when building book interiors, spreads, or any image-sequence PDF.
---

# Images to PDF

Images are naturally sorted and become one page each. With a page size given,
each image is resized to the exact page pixels; without one, pages take each
image's own size at `--dpi`.

## Usage

```bash
# One PDF from a folder of images
./images_to_pdf.py pages/ -o book.pdf --page 6.125x9.25 --dpi 300

# One PDF per subfolder of PARENT/, written as PARENT/<subfolder>.pdf
./images_to_pdf.py PARENT/ --per-folder --page 8.5x11 --dpi 300

# No resize: keep original pixel size (fast, lossless-ish via JPEG q95)
./images_to_pdf.py pages/ -o book.pdf
```

## Options

| option | default | meaning |
|---|---|---|
| `DIR` | required | folder containing images |
| `-o OUT.pdf` | `<DIR>.pdf` | output path (single-PDF mode) |
| `--per-folder` | off | one PDF per subfolder instead of one total |
| `--page WINxHIN` | image's own size | page size in inches, e.g. `6.125x9.25`; images are resized to fill |
| `--fit stretch\|contain` | `stretch` | `contain` letterboxes on white instead of stretching |
| `--dpi N` | `300` | DPI for page sizing and PDF metadata |
| `--quality Q` | `95` | JPEG quality inside the PDF |

Requires Pillow.
