#!/usr/bin/env python3
"""Render PDF pages to images at a chosen DPI via pdftoppm."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def collect(inputs: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        if item.is_dir():
            pdfs.extend(sorted(item.glob("*.pdf")))
        elif item.suffix.lower() == ".pdf":
            pdfs.append(item)
        else:
            raise SystemExit(f"Not a PDF or directory: {item}")
    if not pdfs:
        raise SystemExit("No PDFs found")
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fmt", choices=("png", "jpeg"), default="png")
    parser.add_argument("-o", "--out-dir", type=Path, help="output folder (default: <pdf stem>/)")
    args = parser.parse_args()

    for pdf in collect(args.inputs):
        out_dir = args.out_dir or pdf.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{pdf} -> {out_dir}/ ({args.dpi} dpi)")
        subprocess.run(
            ["pdftoppm", "-" + args.fmt, "-r", str(args.dpi), str(pdf), str(out_dir / pdf.stem)],
            check=True,
        )


if __name__ == "__main__":
    main()
