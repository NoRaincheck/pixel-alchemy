# pdfx1a-compress

Make a print-ready **PDF/X-1a** that is as small as the artwork allows,
fully reproducible.

```
./pdfx1a_compress.py compress INPUT.pdf OUTPUT.pdf
./pdfx1a_compress.py verify   OUTPUT.pdf
```

## What it does

### Modes

| mode | invocation | image handling |
|---|---|---|
| lossless (default) | `compress in.pdf out.pdf` | pixels untouched; `jpegtran -optimize` only, kept if smaller |
| lossy, configurable | `compress in.pdf out.pdf --quality p90` | Pillow re-encode at quality 10..95 (`pNN` or plain int); kept only if smaller |
| lossy + chroma tradeoff | `... --quality p75 --subsampling auto` | also tries C-channel 4:2:2 / 4:2:0; `auto` keeps whichever is smaller |

`--subsampling` choices: `none` (default, max colour fidelity), `422`, `420`,
`auto`. Only the **C** channel is ever subsampled — M, Y and K stay
full-resolution (matches Adobe CMYK JPEG convention).

A candidate always replaces the current best only when it is *strictly smaller*;
otherwise the original bytes are kept and reported. If nothing shrinks you get:

```
note            : no image got smaller at this quality — the originals were
                  already better compressed. Try a lower --quality or --subsampling auto/422/420.
```


Why so conservative? This pipeline was benchmarked against the alternative on a
real Quartz-produced book (99 full-page 300 dpi DeviceCMYK JPEGs):

| method | size vs original JPEGs |
|---|---|
| Ghostscript re-encode, QFactor 0.15 / 0.40 / 0.90 | 236% / 164% / 109% |
| Pillow re-encode q90 / q80 / q75 (4:4:4) | 167% / 132% / 121% |
| Pillow q70–80 with 4:2:2 chroma subsampling | 76–96% (visible colour softening) |
| `jpegtran -optimize` (this tool) | **~97.5% — pixel-identical** |

Apple's Quartz encoder already produces near-optimal CMYK JPEGs, so a plain
re-encode (`--quality p90` … `p70`, 4:4:4) either grows the file or costs
fidelity. Real size cuts on this class of input come from `--quality` **plus**
`--subsampling auto`. If you explicitly want the size cut and accept softer
C-channel colour:

```
./pdfx1a_compress.py compress in.pdf out.pdf --quality p75 --subsampling auto
```

### PDF/X-1a conformance injection
The source "PDFX" file exported by macOS Quartz is usually *not* conforming.
This tool adds everything ISO 15930-1 requires that is missing:

* `/TrimBox` = `/MediaBox` on every page (required; X-1a has no bleed concept here)
* Output intent: `/OutputIntents [ ... ]`, `/S /GTS_PDFX`,
  `DestOutputProfile` = vendored CMYK ICC profile (`vendor/default_cmyk.icc`,
  N=4), `OutputConditionIdentifier` (default `CGATS TR001`, override with
  `--condition-id`; swap the ICC for your printer's, e.g. FOGRA39/GRACoL, via `--icc`)
* Info dict: `GTS_PDFXVersion (PDF/X-1a:2001)` (`:2003` optional),
  `/Trapped /False`, `CreationDate`, `ModDate`
* XMP metadata consistent with the Info dict (`pdfxid:GTS_PDFXVersion`, dates,
  document/instance UUIDs)
* Document colour audit: refuses content with RGB operators, transparency,
  or RGB images (PDF/X-1a forbids them). Documents that need *conversion*
  should first be run through Ghostscript:
  `gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dPDFX=1 -sColorConversionStrategy=CMYK -dProcessColorModel=/DeviceCMYK -o flat.pdf in.pdf`
  then finalise `flat.pdf` with this tool.

## Reproducibility

Same input bytes + same options + same toolchain ⇒ **byte-identical output**
(no wall-clock timestamps anywhere).

* Object order, whitespace, xref layout are canonicalised.
* `/ID`, XMP `DocumentID`/`InstanceID` are derived from SHA-256 of the input.
* Dates resolve in a fixed order: `--date D:YYYYMMDDHHMMSSZ` >
  `SOURCE_DATE_EPOCH` env var > the source file's own CreationDate > fixed epoch.

## Verify

```
./pdfx1a-compress verify out.pdf
```
Checks header version, output intent (+ICC `acsp`, N=4), Info & XMP
consistency, TrimBox on every page, font embedding, absence of RGB operators /
transparency / RGB images, CMYK-ness of large images, ≥295 dpi, and runs
`qpdf --check` when installed. Exit code 0 = conforming.

Note: Apple-produced JPEGs sometimes contain padded entropy segments;
strict libjpeg-based checkers (e.g. `qpdf`) emit warnings about them even on
the untouched original. Tolerant decoders (Preview/Acrobat/poppler/Ghostscript)
read them fine. The verify step reports this as a WARN, not a failure.

## Dependencies

* `python3` — stdlib only for the core pipeline
* `jpegtran` (libjpeg-turbo) — enables the lossless savings; auto-detected,
  gracefully skipped with `--no-jpeg-opt`
* `Pillow` — only for `--quality` / lossy re-encoding
* `qpdf`, poppler (`pdfimages`) — optional extras for `verify`

Tested with: python 3.9, jpegtran/libjpeg-turbo 3.x, Pillow 11.3, qpdf 12.4.

## Results on the reference book

`grandpa_allthelittle_pdfx.pdf` (100 pages, 300 dpi DeviceCMYK, 127,526,089 bytes):

| output | command | size | delta |
|---|---|---|---|
| `…_pdfx1a_compressed.pdf` | *(lossless default)* | 124,863,161 B | **−2.09 %** |
| `…_pdfx1a_p75.pdf` | `--quality p75 --subsampling auto` | 83,410,108 B | **−34.59 %** |

Both are CONFORMING PDF/X-1a:2001 (0 failures), still 1838×2775 px @ 300 dpi
CMYK per page; the p75 variant trades C-channel resolution for size.
Lossless run: ~12 s; p75 run: ~35 s.
The honest takeaway: this book was already within ~2% of the smallest possible
file at 300 dpi full-quality CMYK. The tool guarantees it stays print-safe
(real PDF/X-1a conformance it previously lacked) while shaving what is
available losslessly — and documents exactly how much more compression would cost.
