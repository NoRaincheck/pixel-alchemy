---
name: pdf-to-images
description: Render PDF pages to PNG (or JPEG) images at a chosen DPI using pdftoppm, one output folder per PDF. Use when converting a PDF back into page images for re-processing.
---

# PDF to Images

Wraps `pdftoppm` (poppler) for the recurring "render this PDF back to page
images" step. Each PDF gets its own output folder named after it.

## Usage

```bash
./pdf_to_images.py book.pdf                      # -> book/0001.png at 300 dpi
./pdf_to_images.py *.pdf --dpi 200 --fmt jpeg    # batch, JPEG output
./pdf_to_images.py some_dir_with_pdfs/           # every PDF in the dir
```

## Options

| option | default | meaning |
|---|---|---|
| `INPUT...` | required | one or more PDFs and/or directories of PDFs |
| `--dpi N` | `300` | render resolution |
| `--fmt png\|jpeg` | `png` | image format |
| `-o DIR` | `<pdf stem>/` next to each PDF | output folder |

Requires poppler (`pdftoppm`) on PATH: `brew install poppler`.
