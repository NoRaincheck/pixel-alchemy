#!/usr/bin/env python3
"""Mix moodist ambient sounds into a soundscape of exact duration."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Presets — id:volume layers + optional effect defaults
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "cafe": {
        "desc": "busy cafe, chatter + kitchen",
        "sounds": {"cafe": 0.85, "keyboard": 0.35, "bubbles": 0.20, "crowd": 0.15},
        "lowpass": 3500,
    },
    "city-walk": {
        "desc": "walking in the city",
        "sounds": {"busy-street": 0.70, "traffic": 0.50, "crowd": 0.40, "wind": 0.25, "road": 0.20},
    },
    "office": {
        "desc": "working in the office",
        "sounds": {"office": 0.80, "keyboard": 0.50, "paper": 0.30, "ceiling-fan": 0.25, "clock": 0.15},
    },
    "library": {
        "desc": "quiet study",
        "sounds": {"library": 0.80, "paper": 0.35, "clock": 0.20, "keyboard": 0.25, "rain-on-window": 0.20},
        "lowpass": 2500,
    },
    "rainy-night": {
        "desc": "rain + night calm",
        "sounds": {"light-rain": 0.70, "rain-on-window": 0.40, "thunder": 0.20, "wind-in-trees": 0.35, "crickets": 0.30},
        "lowpass": 4500,
    },
    "fireplace": {
        "desc": "cabin / hygge",
        "sounds": {"campfire": 0.80, "wind": 0.30, "rain-on-tent": 0.25, "owl": 0.15, "crickets": 0.20},
        "lowpass": 4000,
        "reverb": True,
    },
    "deep-focus": {
        "desc": "brown noise focus bed",
        "sounds": {"brown-noise": 0.60, "light-rain": 0.30, "keyboard": 0.15},
        "lowpass": 3000,
    },
    "night-village": {
        "desc": "village evening",
        "sounds": {"night-village": 0.80, "crickets": 0.40, "wind": 0.30, "church": 0.20},
        "reverb": True,
    },
    "construction": {
        "desc": "urban work site",
        "sounds": {"construction-site": 0.70, "traffic": 0.40, "busy-street": 0.30},
    },
    "underwater": {
        "desc": "submerged dream",
        "sounds": {"underwater": 0.80, "whale": 0.40, "waves": 0.30, "droplets": 0.20},
        "lowpass": 2000,
        "reverb": True,
    },
    "train-ride": {
        "desc": "inside a moving train",
        "sounds": {"inside-a-train": 0.80, "rain-on-window": 0.30, "clock": 0.15},
        "highpass": 80,
        "lowpass": 5000,
    },
    "jungle": {
        "desc": "dense forest",
        "sounds": {"jungle": 0.80, "birds": 0.50, "droplets": 0.30, "wind": 0.20},
    },
}

DEFAULT_SOURCE = Path(__file__).resolve().parents[4] / "moodist" / "public" / "sounds"
# fallback for pip/git layouts — walk up until moodist found
if not DEFAULT_SOURCE.is_dir():
    for p in Path(__file__).resolve().parents:
        cand = p / "moodist" / "public" / "sounds"
        if cand.is_dir():
            DEFAULT_SOURCE = cand
            break


def parse_duration(s: str) -> int:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    # 1h30m, 1h, 30m, 90s, etc.
    total = 0
    for val, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", s):
        n = float(val)
        if unit == "h":
            total += n * 3600
        elif unit == "m":
            total += n * 60
        else:
            total += n
    if total == 0:
        raise argparse.ArgumentTypeError(f"bad duration: {s} (try 30s, 5m, 1h)")
    return round(total)


def find_sound(sound_id: str, source: Path) -> Path | None:
    # allow category/id or bare id
    if "/" in sound_id:
        for ext in (".mp3", ".wav", ".ogg", ".m4a", ".flac"):
            cand = source / f"{sound_id}{ext}"
            if cand.is_file():
                return cand
        # also try bare stem under subdirs
        stem = sound_id.split("/")[-1]
        sound_id = stem
    # bare id: search recursively
    for ext in (".mp3", ".wav", ".ogg", ".m4a", ".flac"):
        for p in source.rglob(f"{sound_id}{ext}"):
            return p
    # fuzzy: glob any file whose stem == id
    for p in source.rglob("*.*"):
        if p.stem == sound_id and p.suffix.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".flac"}:
            return p
    return None


def list_sounds(source: Path) -> list[Path]:
    exts = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
    return sorted(p for p in source.rglob("*.*") if p.suffix.lower() in exts)


def parse_sounds_spec(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not spec.strip():
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token or "=" in token:
            sep = ":" if ":" in token else "="
            sid, vol_s = token.rsplit(sep, 1)
            sid, vol_s = sid.strip(), vol_s.strip()
            try:
                vol = float(vol_s)
            except ValueError:
                raise SystemExit(f"bad volume '{vol_s}' in '{token}' (expect 0-1)")
        else:
            sid, vol = token, 0.5
        out[sid] = max(0.0, min(1.0, vol))
    return out


def build_ffmpeg_cmd(layers: list[tuple[Path, float]], duration: int, args, out: Path) -> list[str]:
    n = len(layers)
    # filter_complex
    filters: list[str] = []
    for i, (_, vol) in enumerate(layers):
        # volume + ensure stereo 44.1k for consistent mixing
        filters.append(f"[{i}:a]aformat=sample_fmts=fltp:channel_layouts=stereo,volume={vol:.3f}[a{i}]")

    if n == 1:
        last = "[a0]"
    else:
        amix_in = "".join(f"[a{i}]" for i in range(n))
        filters.append(f"{amix_in}amix=inputs={n}:duration=longest:dropout_transition=0:normalize=0[mix]")
        last = "[mix]"

    # global effect chain on last
    chain: list[str] = []
    if args.highpass is not None:
        chain.append(f"highpass=f={int(args.highpass)}")
    # preset lowpass if not overridden; args.lowpass may be None -> use preset default
    if args.lowpass is not None:
        chain.append(f"lowpass=f={int(args.lowpass)}")
    if args.reverb:
        chain.append("aecho=0.8:0.88:60:0.4")
    if args.fade and args.fade > 0:
        f = min(args.fade, duration // 2)
        chain.append(f"afade=t=in:st=0:d={f}")
        chain.append(f"afade=t=out:st={duration - f}:d={f}")
    if args.normalize:
        chain.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    if chain:
        filters.append(f"{last}{','.join(chain)}[out]")
        map_label = "[out]"
    else:
        # rename last to out if no chain
        if last != "[out]":
            filters.append(f"{last}anull[out]")
            map_label = "[out]"
        else:
            map_label = last

    fc = ";".join(filters)

    cmd = ["ffmpeg", "-y"]
    for p, _ in layers:
        cmd += ["-stream_loop", "-1", "-i", str(p)]
    cmd += ["-t", str(duration), "-filter_complex", fc, "-map", map_label]

    # codec by extension
    ext = out.suffix.lower()
    if ext == ".wav":
        cmd += ["-c:a", "pcm_s16le"]
    elif ext == ".ogg":
        cmd += ["-c:a", "libvorbis", "-q:a", "4"]
    elif ext == ".m4a":
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    elif ext == ".flac":
        cmd += ["-c:a", "flac"]
    else:  # mp3 default
        cmd += ["-c:a", "libmp3lame", "-q:a", "2"]
        if out.suffix == "":
            out = out.with_suffix(".mp3")

    cmd.append(str(out))
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), help="preset name")
    p.add_argument("--sounds", default="", help="comma list id:vol, e.g. cafe:0.8,keyboard:0.3")
    p.add_argument("--duration", default="5m", help="target length: seconds or 30s/5m/1h (default 5m)")
    p.add_argument("-o", "--output", type=Path, help="output file (ext picks codec)")
    p.add_argument("--fade", type=float, default=3, help="fade in/out seconds (0 to disable)")
    p.add_argument("--reverb", action="store_true", help="light room reverb")
    p.add_argument("--lowpass", type=float, default=None, help="lowpass freq Hz")
    p.add_argument("--highpass", type=float, default=None, help="highpass freq Hz")
    p.add_argument("--normalize", action="store_true", help="loudness normalize")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="moodist sounds folder")
    p.add_argument("--list-presets", action="store_true", help="list presets and exit")
    p.add_argument("--list-sounds", action="store_true", help="list available sound ids and exit")
    p.add_argument("--dry-run", action="store_true", help="print ffmpeg command and exit")
    args = p.parse_args()

    if args.list_presets:
        for name in sorted(PRESETS):
            info = PRESETS[name]
            layers = ", ".join(f"{k}:{v}" for k, v in info["sounds"].items())
            eff = []
            if info.get("lowpass"):
                eff.append(f"lowpass {info['lowpass']}")
            if info.get("highpass"):
                eff.append(f"highpass {info['highpass']}")
            if info.get("reverb"):
                eff.append("reverb")
            eff_s = f"  [{', '.join(eff)}]" if eff else ""
            print(f"{name:15s} {info['desc']:30s}  {layers}{eff_s}")
        return

    if args.list_sounds:
        if not args.source.is_dir():
            raise SystemExit(f"source not found: {args.source}")
        for fp in list_sounds(args.source):
            rel = fp.relative_to(args.source)
            dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", str(fp)]
            try:
                r = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=5)
                dur = f"{float(r.stdout.strip()):5.1f}s" if r.stdout.strip() else "  ?  "
            except Exception:
                dur = "  ?  "
            print(f"{dur}  {rel}  (id: {fp.stem})")
        return

    if not args.preset and not args.sounds:
        p.error("need --preset and/or --sounds")

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH (brew install ffmpeg)")

    duration = parse_duration(args.duration)
    if duration < 1:
        raise SystemExit("duration must be >=1s")

    # merge preset + custom sounds
    merged: dict[str, float] = {}
    preset_lowpass = None
    preset_highpass = None
    preset_reverb = False
    if args.preset:
        info = PRESETS[args.preset]
        merged.update(info["sounds"])
        preset_lowpass = info.get("lowpass")
        preset_highpass = info.get("highpass")
        preset_reverb = bool(info.get("reverb"))

    custom = parse_sounds_spec(args.sounds)
    merged.update(custom)  # custom overrides preset

    if not merged:
        raise SystemExit("no sounds resolved")

    # resolve lowpass/highpass: CLI overrides preset; if CLI not given, use preset default
    if args.lowpass is None and preset_lowpass is not None:
        args.lowpass = float(preset_lowpass)
    if args.highpass is None and preset_highpass is not None:
        args.highpass = float(preset_highpass)
    if not args.reverb and preset_reverb:
        args.reverb = True

    if not args.source.is_dir():
        raise SystemExit(f"source not found: {args.source} (use --source)")

    layers: list[tuple[Path, float]] = []
    missing: list[str] = []
    for sid, vol in merged.items():
        fp = find_sound(sid, args.source)
        if fp is None:
            missing.append(sid)
        else:
            layers.append((fp, vol))

    if missing:
        avail = sorted(fp.stem for fp in list_sounds(args.source))
        raise SystemExit(f"sounds not found: {', '.join(missing)}\ntry --list-sounds (available: {', '.join(avail[:10])}...)")

    # deterministic output name if not given
    if args.output is None:
        base = args.preset if args.preset else "mix"
        # include duration in name
        m, s = divmod(duration, 60)
        tag = f"{m}m{s:02d}s" if m else f"{s}s"
        if duration % 60 == 0 and duration >= 60:
            tag = f"{duration // 60}m"
            if duration >= 3600 and duration % 3600 == 0:
                tag = f"{duration // 3600}h"
        args.output = Path(f"{base}_{tag}.mp3")

    # ensure parent exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_ffmpeg_cmd(layers, duration, args, args.output)

    print(f"mix: {', '.join(f'{sid}:{vol}' for sid, vol in merged.items())}")
    print(f"duration: {duration}s  fade:{args.fade}s  -> {args.output}")
    for fp, vol in layers:
        print(f"  {vol:.2f}  {fp.relative_to(args.source) if fp.is_relative_to(args.source) else fp}")

    if args.dry_run:
        print("\n" + " ".join(shlex_quote(c) for c in cmd))
        return

    print(f"\nfilter: {cmd[cmd.index('-filter_complex')+1]}")
    subprocess.run(cmd, check=True)
    # verify duration
    probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(args.output)]
    r = subprocess.run(probe, capture_output=True, text=True)
    if r.stdout.strip():
        actual = float(r.stdout.strip())
        print(f"done: {args.output} ({actual:.1f}s)")


def shlex_quote(s: str) -> str:
    if re.search(r"[ \t\n\"'\\]", s):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


if __name__ == "__main__":
    main()
