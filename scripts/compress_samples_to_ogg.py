"""Compress samples/sounds/*.{mp3,wav} -> .ogg (Opus) for consistency.

Uses ffmpeg libopus. Keeps directory structure. Verifies with soundfile.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "samples" / "sounds"


def convert_one(src: Path, dst: Path, bitrate: str, binaural_bitrate: str) -> tuple[str, bool, str]:
    br = binaural_bitrate if "binaural" in src.parts else bitrate
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-c:a",
        "libopus",
        "-b:a",
        br,
        "-vbr",
        "on",
        "-compression_level",
        "10",
        "-application",
        "audio",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return (str(src), True, f"-> {dst.name} @ {br}")
    except subprocess.CalledProcessError as e:
        return (str(src), False, str(e))
    except FileNotFoundError:
        return (str(src), False, "ffmpeg not found")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compress samples to OGG Opus")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="samples/sounds dir")
    ap.add_argument("--bitrate", default="64k", help="opus bitrate for most samples (default 64k)")
    ap.add_argument("--binaural-bitrate", default="96k", help="opus bitrate for binaural/ (default 96k)")
    ap.add_argument("--keep-original", action="store_true", help="keep .mp3/.wav alongside .ogg")
    ap.add_argument("--jobs", type=int, default=8, help="parallel ffmpeg jobs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_root: Path = args.src
    if not src_root.exists():
        print(f"missing {src_root}", file=sys.stderr)
        sys.exit(1)

    files = [p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in (".mp3", ".wav")]
    files.sort()
    if not files:
        print("no .mp3/.wav found")
        return

    print(f"found {len(files)} files in {src_root} | {args.bitrate} (binaural {args.binaural_bitrate})")
    if args.dry_run:
        for f in files:
            print(f"  {f.relative_to(src_root)} -> {f.with_suffix('.ogg').relative_to(src_root)}")
        return

    oggs: list[Path] = []
    failed: list[tuple[str, str]] = []
    total_in = 0
    total_out = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(convert_one, f, f.with_suffix(".ogg"), args.bitrate, args.binaural_bitrate): f for f in files}
        for fut in concurrent.futures.as_completed(futs):
            src, ok, msg = fut.result()
            src_p = Path(src)
            print(f"{'OK' if ok else 'FAIL'} {src_p.relative_to(src_root)} {msg}")
            if ok:
                oggs.append(src_p.with_suffix(".ogg"))
                total_in += src_p.stat().st_size
            else:
                failed.append((src, msg))

    if failed:
        print(f"\n{len(failed)} failed:", file=sys.stderr)
        for s, m in failed:
            print(f"  {s}: {m}", file=sys.stderr)
        sys.exit(1)

    for o in oggs:
        total_out += o.stat().st_size

    print(f"\nconverted {len(oggs)}/{len(files)} | {total_in/1024/1024:.1f} MB -> {total_out/1024/1024:.1f} MB ({100*total_out/max(1,total_in):.0f}%)")

    if not args.keep_original:
        for f in files:
            f.unlink()
        print(f"removed {len(files)} originals")

    # verify with soundfile
    try:
        import soundfile as sf

        bad = []
        for o in sorted(oggs):
            try:
                sf.info(str(o))
            except Exception as e:
                bad.append((o, e))
        if bad:
            print(f"soundfile verify: {len(bad)} bad", file=sys.stderr)
            for p, e in bad:
                print(f"  {p}: {e}", file=sys.stderr)
        else:
            print(f"soundfile verify: {len(oggs)} OK (Vorbis/Opus readable)")
    except ImportError:
        pass

    # gitignore hint
    gi = src_root.parent / ".gitignore"
    if gi.exists() and "# *" in gi.read_text():
        print(f"note: {gi} contains '# *' (commented) — no action needed. If you want to track .ogg, ensure not ignored.")


if __name__ == "__main__":
    main()
