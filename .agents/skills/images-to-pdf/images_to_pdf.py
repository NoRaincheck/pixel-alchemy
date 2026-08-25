#!/usr/bin/env python3
"""Combine folders of images into PDFs at a given page size and DPI."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name.lower())]


def load_pages(folder: Path, args) -> list[Image.Image]:
    images = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
    if not images:
        return []
    pages = []
    for path in images:
        with Image.open(path) as img:
            img = img.convert("RGB")
        if args.page:
            pw, ph = round(args.page[0] * args.dpi), round(args.page[1] * args.dpi)
            if args.fit == "contain":
                img.thumbnail((pw, ph), Image.LANCZOS)  # ty:ignore[unresolved-attribute]
                canvas = Image.new("RGB", (pw, ph), "white")
                canvas.paste(img, ((pw - img.width) // 2, (ph - img.height) // 2))
                img = canvas
            else:
                img = img.resize((pw, ph), Image.LANCZOS)  # ty:ignore[unresolved-attribute]
        pages.append(img)
    return pages


def save_pdf(pages: list[Image.Image], output: Path, args) -> None:
    pages[0].save(output, "PDF", save_all=True, append_images=pages[1:], resolution=args.dpi, quality=args.quality)
    print(f"{output} ({len(pages)} pages)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="output PDF (default: <DIR>.pdf)")
    parser.add_argument("--per-folder", action="store_true", help="one PDF per subfolder of DIR")
    parser.add_argument("--page", type=lambda s: tuple(map(float, s.split("x"))), metavar="WINxHIN")
    parser.add_argument("--fit", choices=("stretch", "contain"), default="stretch")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    if args.per_folder:
        for folder in sorted(p for p in args.directory.iterdir() if p.is_dir()):
            pages = load_pages(folder, args)
            if pages:
                save_pdf(pages, args.directory / f"{folder.name}.pdf", args)
    else:
        pages = load_pages(args.directory, args)
        if not pages:
            raise SystemExit(f"No images found in {args.directory}")
        output = args.output or args.directory.with_suffix(".pdf")
        save_pdf(pages, output, args)


if __name__ == "__main__":
    main()
