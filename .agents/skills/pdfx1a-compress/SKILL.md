---
name: pdfx1a-compress
description: Create print-ready PDF/X-1a:2001 files with minimal size. Compresses PDF images (lossless jpegtran or configurable lossy re-encoding) and injects PDF/X-1a conformance metadata. Use when a PDF needs to be print-ready, PDF/X-1a compliant, and/or reduced in file size.
---

# PDF/X-1a Compress

## Setup

```bash
cd /path/to/skill
# Optional dependencies: ghostscript (auto RGB→CMYK), jpegtran (lossless savings),
# Pillow (lossy --quality), qpdf + poppler (verify)
```

## Usage

### Compress

```bash
./pdfx1a_compress.py compress INPUT.pdf OUTPUT.pdf                     # lossless + PDF/X-1a
./pdfx1a_compress.py compress rgb.pdf OUTPUT.pdf                       # auto RGB→CMYK via Ghostscript
./pdfx1a_compress.py compress in.pdf out.pdf --quality p90             # re-encode @ quality 90
./pdfx1a_compress.py compress in.pdf out.pdf --quality p75 --subsampling auto
```

### Verify

```bash
./pdfx1a_compress.py verify OUTPUT.pdf
```

Checks PDF/X-1a conformance: header, output intent + ICC profile, Info & XMP consistency,
TrimBox on every page, font embedding, no RGB/transparency, CMYK images, ≥295 dpi.

## Modes

| mode | command | image handling |
|---|---|---|
| lossless (default) | `compress in.pdf out.pdf` | pixels untouched; `jpegtran -optimize` only, kept if smaller |
| lossy | `compress in.pdf out.pdf --quality pNN` | Pillow re-encode at quality 10..95 (`pNN` or int); kept only if smaller |
| lossy + chroma | `... --quality p75 --subsampling auto` | tries C-channel 4:2:2 / 4:2:0; keeps smaller |

`--subsampling`: `none` (default), `422`, `420`, `auto`. Only C channel subsampled.

## PDF/X-1a Conformance

Injects what ISO 15930-1 requires:
- `/TrimBox` = `/MediaBox` on every page
- OutputIntent: `/S /GTS_PDFX`, vendored CMYK ICC (`vendor/default_cmyk.icc`)
- Info dict: `GTS_PDFXVersion (PDF/X-1a:2001)`, `/Trapped /False`, dates
- XMP metadata with document/instance UUIDs

Override ICC: `--icc /path/to/custom.icc`
Override condition ID: `--condition-id "FOGRA39"`

## Reproducibility

Same input + options + toolchain → byte-identical output. No timestamps, deterministic IDs from SHA-256 of input.

Pin dates: `--date D:YYYYMMDDHHMMSSZ` or `SOURCE_DATE_EPOCH` env var.

## Dependencies

| dependency | needed for |
|---|---|
| python3 stdlib | core pipeline |
| jpegtran | lossless savings (auto-detected, skipped if absent) |
| Pillow | lossy `--quality` |
| ghostscript | auto RGB→CMYK conversion |
| qpdf, poppler | verify extras |
