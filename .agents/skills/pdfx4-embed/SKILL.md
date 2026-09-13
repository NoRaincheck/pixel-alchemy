---
name: pdfx4-embed
description: Embed fonts and finalise to PDF/X-4 without flattening anything. Keeps RGB, live transparency and layers intact — only font subsets and conformance metadata change. Use when artwork must not be converted to CMYK, or when the printer accepts PDF/X-4.
---

# PDF/X-4 Embed

Font-embedding-only alternative to `pdfx1a-compress`. Nothing is flattened,
nothing is re-coloured.

## Setup

```bash
# Requires ghostscript + the sibling pdfx1a-compress skill (used as finaliser)
```

## Usage

```bash
./pdfx4_embed.py embed INPUT.pdf OUTPUT.pdf
./pdfx4_embed.py embed rgb.pdf out.pdf --condition-id "FOGRA39"
./pdfx4_embed.py embed in.pdf out.pdf --date D:20260823120000Z --title "My Book"
```

Verify with the sibling skill (standard is auto-detected from the file):

```bash
../pdfx1a-compress/pdfx1a_compress.py verify OUTPUT.pdf
```

## What it does

1. **Stage 1 — Ghostscript font pass**: embeds + subsets all fonts
   (`CompatibilityLevel 1.6`, `LeaveColorUnchanged`, JPEG/JPX passthrough,
   no downsampling). RGB, transparency, and layer structure survive.
2. **Stage 2 — qpdf normalisation**: unpacks object streams to classic objects
   (all bytes preserved) so the finaliser can parse the file.
3. **Stage 3 — deterministic X-4 finalise**: delegates to
   `pdfx1a-compress` with `--pdfx-version PDF/X-4 --no-jpeg-opt`, so image
   pixels are untouched. Dates/IDs derive from your original via `--id-source`.

## When to use which

| situation | skill |
|---|---|
| Printer demands PDF/X-1a (CMYK, flattened) | `pdfx1a-compress` |
| Artwork must keep RGB / transparency / layers, printer accepts X-4 | `pdfx4-embed` |
| Smallest file wins, quality negotiable | `pdfx1a-compress --quality pNN` |

## Limitations

- Output is PDF/X-4, not X-1a: confirm the printer accepts it.
- Non-JPEG rasters may be re-encoded once by the Ghostscript pass; JPEGs pass through.
- Encrypted or object-stream PDFs may need a `qpdf --decrypt --object-streams=disable` pre-pass.
