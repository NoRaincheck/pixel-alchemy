# pdfx1a-compress

Turn any print PDF into a **conforming, reproducible PDF/X-1a (CMYK)** that is
as small as the artwork allows.

```
./pdfx1a_compress.py compress INPUT.pdf OUTPUT.pdf [options]
./pdfx1a_compress.py verify   OUTPUT.pdf
```

## Quick start

```sh
# already-CMYK book — keep pixels untouched, add real PDF/X-1a conformance
./pdfx1a_compress.py compress book_cmyk.pdf out.pdf

# RGB book — auto-converts to DeviceCMYK, then optimises (default q80 + 4:2:2 auto)
./pdfx1a_compress.py compress book_rgb.pdf out.pdf

# explicit quality ladder
./pdfx1a_compress.py compress in.pdf out.pdf --quality p90            # finest
./pdfx1a_compress.py compress in.pdf out.pdf --quality p75 --subsampling auto   # smallest

# check the result
./pdfx1a_compress.py verify out.pdf
```

## Modes

| situation | what happens |
|---|---|
| input is colour-clean (CMYK/Gray only) | finalised directly; **pixels untouched** unless `--quality` given |
| input has RGB/Lab images, RGB fills or transparency | **automatic Ghostscript stage** converts everything to DeviceCMYK (fonts embedded, transparency flattened, images re-encoded as JPEG @300 dpi) |
| after any conversion | the size optimiser always runs — default `--quality 80 --subsampling auto` unless you pass `--quality` explicitly (conversion is already a lossy generation, so keeping Ghostscript's large streams verbatim buys nothing) |

### Image optimisation

* `jpegtran -optimize` pass for every JPEG — **lossless**, kept only if smaller.
* With `--quality pNN` (or an int 10–95): Pillow re-encode candidates; a
  candidate replaces the current best **only when strictly smaller**. Nothing
  is degraded without a size win; if nothing shrinks you get an honest note.
* `--subsampling none|422|420|auto` — chroma subsampling of the **C channel
  only**; M, Y and K stay full-resolution (Adobe CMYK JPEG convention).
  `auto` tries all and keeps whichever is smallest.

Why so conservative on already-CMYK input? Benchmarks against Quartz-produced
books show Apple's JPEG encoder is near-optimal: every full-quality re-encode
(Ghostscript QFactor 0.15/0.40/0.90, Pillow q90/q80/q75 at 4:4:4) produced
**larger** files than the originals (109–236%). Only C-channel subsampling
actually shrinks them (−15…−35%), at the cost of some colour sharpness.

## What "PDF/X-1a" gets injected

The finaliser adds everything ISO 15930-1 requires that is missing:

* `/TrimBox` = `/MediaBox` on every page
* Output intent `/OutputIntents` with `/S /GTS_PDFX`, CMYK ICC profile
  (`vendor/default_cmyk.icc`, N=4), `OutputConditionIdentifier`
  (`--condition-id`, default `CGATS TR001`; swap in your printer's profile via
  `--icc`, e.g. FOGRA39 or GRACoL)
* Info dict: `GTS_PDFXVersion (PDF/X-1a:2001)` (or `:2003`),
  `/Trapped /False`, pinned `CreationDate`/`ModDate`
* XMP metadata consistent with the Info dict (`pdfxid:GTS_PDFXVersion`, dates,
  document/instance UUIDs)

Colour rules are enforced, not assumed: the built-in audit refuses content
with RGB operators, transparency or RGB images — that content is exactly what
triggers the automatic conversion stage.

## Reproducibility

Same input bytes + same options + same toolchain ⇒ **byte-identical output**.

* canonical object order / whitespace / xref layout,
* document & instance IDs derived from SHA-256 of the *input* file,
* dates resolved deterministically: `--date D:YYYYMMDDHHMMSSZ` >
  `SOURCE_DATE_EPOCH` > the source's own CreationDate > fixed epoch —
  never the wall clock, even when Ghostscript runs as a stage.

Verified by double-run MD5 comparison on all reference books, including the
Ghostscript-conversion path.

## Verify

```
./pdfx1a_compress.py verify out.pdf
```

Checks header version, output intent (+ICC `acsp` magic, N=4), Info/XMP
consistency, TrimBox on every page, font embedding, absence of RGB operators /
transparency / RGB images, CMYK-ness of all large images, ≥295 dpi, plus
`qpdf --check` when installed. Exit code 0 = conforming.

Note: Apple-produced JPEGs sometimes contain padded entropy segments;
strict libjpeg-based tools (`qpdf`) warn about them even on untouched
originals. Preview/Acrobat/poppler/Ghostscript read them fine — verify
reports this as WARN, not FAIL.

## Dependencies

| tool | needed for |
|---|---|
| python3 (stdlib) | core pipeline & verify |
| ghostscript | automatic RGB→CMYK conversion stage |
| jpegtran (libjpeg-turbo) | lossless savings (auto-detected, optional) |
| Pillow | `--quality` lossy mode & converted-input optimisation |
| qpdf, poppler-utils | extra checks in `verify` |

Tested with: python 3.9, Ghostscript 10.07.1, libjpeg-turbo 3.x,
Pillow 11.3, qpdf 12.4, mutool 1.28 (macOS 15, arm64).

## Results on the reference books

All outputs CONFORMING PDF/X-1a:2001, pages still 1838×2775 px @ 300 dpi.

| input | command | output size | delta |
|---|---|---|---|
| `grandpa_allthelittle_pdfx.pdf` (127.5 MB, CMYK) | *(lossless)* | 124.9 MB | −2.09 % |
| `grandpa_allthelittle_pdfx.pdf` | `--quality p75 --subsampling auto` | 83.4 MB | −34.59 % |
| `grandpa_allthelittle.pdf` (55.3 MB, **RGB**) | *(auto-convert, default q80+auto)* | 91.2 MB | +64.9 %¹ |
| `grandpa_allthelittle.pdf` | *(auto-convert)* `--quality p75 --subsampling auto` | 82.9 MB | +49.9 %¹ |

¹ Growth is inherent: RGB→CMYK adds a 4th channel (~+33 % raw pixels) plus a
print-grade re-encode. Colour fidelity of the conversion was checked by
render-diffing: RMSE ≈ 1–3 of 255 per channel vs the RGB original.
Typical runtimes: ~12 s lossless finalisation, ~2.5 min with conversion
(100 pages).

## Limitations

* Parser targets classic-xref PDFs (no object streams / encryption). Inputs
  using newer structures are routed through the Ghostscript stage automatically,
  which normalises them first.
* Conversion uses Ghostscript's colour management into the *output intent*
  profile family; for exact press targeting, supply your printer's ICC with
  `--icc` and matching `--condition-id`.
