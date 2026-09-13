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
./pdfx1a_compress.py compress in.pdf out.pdf --pdfx-version PDF/X-4    # keep RGB/transparency, no conversion
```

Already-CMYK input stays lossless unless `--quality` is given. Input needing
Ghostscript conversion is optimised at `--quality 80 --subsampling auto` by
default (the conversion is already a lossy generation, so keeping its large
streams verbatim buys nothing); pass `--quality` explicitly to override, or
`--no-convert` to skip conversion (output will not be X-1a conforming).

### Verify

```bash
./pdfx1a_compress.py verify OUTPUT.pdf
```

Checks PDF/X conformance: header, output intent + ICC profile, Info & XMP consistency,
TrimBox on every page, font embedding, and — for X-1a only — no RGB/transparency,
CMYK images, ≥295 dpi. The expected standard is auto-detected from the file's
`GTS_PDFXVersion`. (RGB/transparency are allowed and reported informationally
for PDF/X-4.)

## When NOT to use

If the artwork must keep RGB, live transparency, or layers (no CMYK flattening),
use the `pdfx4-embed` skill instead: it only embeds fonts and finalises to PDF/X-4.

## Modes

| mode | command | image handling |
|---|---|---|
| lossless (default) | `compress in.pdf out.pdf` | pixels untouched; `jpegtran -optimize` only, kept if smaller |
| lossy | `compress in.pdf out.pdf --quality pNN` | Pillow re-encode at quality 10..95 (`pNN` or int); kept only if smaller |
| lossy + chroma | `... --quality p75 --subsampling auto` | tries C-channel 4:2:2 / 4:2:0; keeps smaller |

`--subsampling`: `none` (default), `422`, `420`, `auto`. Only C channel subsampled.

## PDF/X Conformance

Injects what the standard requires (`--pdfx-version PDF/X-1a:2001` default,
also `:2003` and `PDF/X-4`):
- `/TrimBox` = `/MediaBox` on every page
- OutputIntent: `/S /GTS_PDFX`, vendored CMYK ICC (`vendor/default_cmyk.icc`)
- Info dict: `GTS_PDFXVersion`, `/Trapped /False`, dates
- XMP metadata with document/instance UUIDs

PDF/X-4 skips the Ghostscript CMYK conversion entirely, so RGB, transparency
and layers survive; only fonts must already be embedded (see `pdfx4-embed`).

Override ICC: `--icc /path/to/custom.icc`
Override condition ID: `--condition-id "FOGRA39"`
Skip conversion even when needed: `--no-convert`

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
