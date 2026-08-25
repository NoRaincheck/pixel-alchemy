---
name: cover-compose
description: Compose print book cover spreads by placing front and back cover images onto template PNGs, auto-detecting the magenta spine marker strip to split front/back regions. Use when generating book cover mockups or print wraps from templates.
---

# Cover Compose

For each entry in a metadata JSON and each template PNG, produces a composed
cover with the **back** image filling the area left of the spine and the
**front** image covering from the spine rightwards.

The spine position is detected per template as the left edge of a horizontal
**magenta strip** (`R>200, G<50, B>200` on the middle row); falls back to
center if absent. Images are resized to exactly fill their region.

## Usage

```bash
./make_covers.py --metadata metadata.json --templates tmpl/ --output covers/
```

Metadata format (paths relative to the metadata file):

```json
[
  {"front": "front_v1.png", "back": "back_v1.png"},
  {"front": "front_v2.png", "back": "back_v2.png"}
]
```

Outputs `<output>/<front-stem>_<template-stem>.png` for every entry x template
combination, skipping entries whose images are missing.

Requires Pillow + numpy.
