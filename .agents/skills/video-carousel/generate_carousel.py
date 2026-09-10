#!/usr/bin/env python3
"""Generate 9:16 images + text overlays + video carousels from carousel JSONs.

Reads every JSON matching ``--input`` (List[List[{text, visual_description}]]),
generates Z-images with the chosen style palette, burns centered Courier
typewriter text via ffmpeg (grain + vintage, same as
generate_z_videos.py), and concatenates each carousel into an MP4.

Generic reusable pipeline — no topic-specific
hardcoding; use ``--style noir|soft`` to switch palettes.

Usage::

    .agents/skills/video-carousel/generate_carousel.py --input "input/*.json" --output-root output/
    .agents/skills/video-carousel/generate_carousel.py --input "input/my-topic.json" --dry-run
    .agents/skills/video-carousel/generate_carousel.py --input "input/my-topic.json" --only my-topic --carousel 2
    .agents/skills/video-carousel/generate_carousel.py --input "input/*.json" --skip-generate --force
    .agents/skills/video-carousel/generate_carousel.py --input "input/*.json" --no-video
    .agents/skills/video-carousel/generate_carousel.py --input "input/*.json" --no-overlay
    .agents/skills/video-carousel/generate_carousel.py --input "input/*.json" --style noir

"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pixel_alchemy.generation.sd_cli import generate

# ---------------------------------------------------------------------------
# Paths / model config (same as generate_z_images.py)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_GLOB_DEFAULT = "*.json"
OUTPUT_ROOT_DEFAULT = BASE_DIR / "output"

DIFFUSION_MODEL = "/Users/crn/.local/share/stable-diffusion.cpp/build/models/z_image_turbo-Q3_K.gguf"
VAE = "/Users/crn/.local/share/stable-diffusion.cpp/build/models/flux1_schnell_diffusion_pytorch_model.safetensors"
LLM = "/Users/crn/.local/share/stable-diffusion.cpp/build/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# ---------------------------------------------------------------------------
# Style hints
# ---------------------------------------------------------------------------
STYLE_HINT_NOIR = (
    "Use a bold avant-garde composition with fragmented geometric shapes and dramatic asymmetry. "
    "Create a portrait-oriented, full-bleed retro avant-garde noir painting. "
    "100% hand-painted by an artist — not digital, not photorealistic, not CGI. "
    "Use a bold avant-garde composition with fragmented geometric shapes and dramatic asymmetry. "
    "Light it like film noir: deep inky blacks and stark chiaroscuro contrast, with a warm amber "
    "and burnt-orange glow piercing through deep crimson, maroon and velvety shadows. Paint with "
    "visible expressive brushwork and palette-knife texture, wet-on-wet washes that bleed and pool "
    "at the edges, and hand-drawn ink outlines. Use a mid-century retro colour palette for a "
    "melancholic, cinematic noir mood. Emphasize painterly, analogue, artist-painted texture."
)

# Soft palette — muted cool shadows + warm candlelight accent (generic, not topic-specific)
STYLE_HINT_SOFT = (
    "Create a portrait-oriented, full-bleed painterly illustration. "
    "100% hand-painted by an artist — not digital, not photorealistic, not CGI. "
    "Use a soft, gentle composition with balanced, intimate framing. "
    "Light it softly: muted slate blues, dusty greys and deep indigo shadows warmed by "
    "a gentle amber, honey-gold and candlelight glow. Soft chiaroscuro with tender transitions "
    "between cool, desaturated shadows and warm golden highlights, never harsh. "
    "Paint with visible soft brushwork, delicate watercolor washes that bloom and feather, "
    "subtle paper texture and fine hand-drawn ink outlines. "
    "Use a muted, comforting palette — dusty rose, sage, warm cream, soft gold, pale lavender, "
    "weathered linen — for a melancholic yet hopeful, reverent and intimate mood. "
    "Emphasize painterly, analogue, artist-painted texture, tender, quiet and deeply human."
)

# ---------------------------------------------------------------------------
# Render defaults — 9:16
# ---------------------------------------------------------------------------
WIDTH = 768
HEIGHT = 1366  # 1365 is odd -> libx264 needs even; 1366 is nearest even for 9:16
STEPS = 8
FPS = 8
DURATION_DEFAULT = 3.0
FONTSIZE_DEFAULT = 52
CROP_DEFAULT = 0.04

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Courier.ttc",
    "/Users/crn/Library/Fonts/DejaVuSansMono.ttf",
    "/Users/crn/Library/Fonts/DejaVuSansMono-Bold.ttf",
    "/System/Library/Fonts/Supplemental/PTMono.ttc",
    "/System/Library/Fonts/Monaco.ttf",
]

TEAR_POS = [
    (0.22, 4, 0.78, 3),
    (0.35, 5, 0.65, 2),
    (0.48, 3, 0.52, 4),
    (0.15, 6, 0.85, 2),
    (0.28, 4, 0.72, 3),
    (0.40, 5, 0.60, 2),
    (0.18, 3, 0.82, 4),
    (0.30, 4, 0.70, 3),
    (0.45, 5, 0.55, 2),
    (0.20, 6, 0.80, 3),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_font() -> str | None:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    try:
        r = subprocess.run(["fc-match", "Courier New", "--format", "%{file}\n"], capture_output=True, text=True)
        cand = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        if cand and Path(cand).exists():
            return cand
    except Exception:
        pass
    return None


def find_ffmpeg() -> str:
    cands = [
        "/opt/homebrew/Cellar/ffmpeg-full/8.1.2_2/bin/ffmpeg",
        shutil.which("ffmpeg-full"),
        shutil.which("ffmpeg"),
    ]
    for c in cands:
        if not c:
            continue
        try:
            r = subprocess.run([c, "-filters"], capture_output=True, text=True)
            if "drawtext" in r.stdout:
                return c
        except Exception:
            continue
    for p in Path("/opt/homebrew/Cellar/ffmpeg-full").glob("*/bin/ffmpeg"):
        try:
            r = subprocess.run([str(p), "-filters"], capture_output=True, text=True)
            if "drawtext" in r.stdout:
                return str(p)
        except Exception:
            continue
    return shutil.which("ffmpeg") or "ffmpeg"


def escape_drawtext_textfile(text: str, wrap_width: int = 30) -> str:
    return textwrap.fill(text.strip(), width=wrap_width, break_long_words=False)


def _wrap_for_fs(fontsize: int) -> int:
    # Courier is monospaced: char_w ≈ 0.60 * fontsize.
    # Usable width ≈ 680px (≈768 - 88px margin). max_chars = usable / char_w.
    # Previous values (32 for 52pt) overflowed 768px (e.g. 28 chars ≈ 868px).
    if fontsize >= 80:
        return 14
    if fontsize >= 72:
        return 16
    if fontsize >= 64:
        return 18
    if fontsize >= 56:
        return 20
    return 22


def build_prompt(visual_description: str, style_hint: str) -> str:
    return f"{visual_description.strip()}\n\n{style_hint}"


def load_carousels(json_path: Path) -> list[list[dict]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{json_path}: expected non-empty list")
    # detect flat List[Frame] vs List[List[Frame]]
    first = data[0]
    if isinstance(first, dict) and "text" in first:
        # flat -> wrap
        return [data]  # type: ignore
    if isinstance(first, list):
        return data  # type: ignore
    raise ValueError(f"{json_path}: unexpected structure: {type(first)} {first!r:.200}")


# ---------------------------------------------------------------------------
# Overlay: single image -> overlay.png with grain/vintage/tears/drawtext
# ---------------------------------------------------------------------------

def _build_grain_cmd(
    raw_png: Path,
    out_png: Path,
    crop: float,
    width: int,
    height: int,
    ffmpeg_bin: str,
    tear_idx: int = 0,
) -> list[str]:
    """Grain + vintage only (no text) — used as first stage of PIL overlay."""
    filter_complex = (
        f"[0:v]crop=w='iw*(1-2*{crop})':h='ih*(1-2*{crop})':x='iw*{crop}':y='ih*{crop}',"
        f"scale={width}:{height}:flags=lanczos,setsar=1,curves=vintage,"
        f"colorbalance=rs=0.015:gs=0.004:bs=-0.015:rm=0.008:gm=0.002:bm=-0.008,"
        f"eq=contrast=1.015:brightness=0:saturation=0.97,format=yuv444p,setsar=1[base];"
        f"[1:v]noise=alls=80:allf=t:all_seed={tear_idx*11},"
        f"scale=w='iw/(2+random({tear_idx*2})*3)':h='ih/(2+random({tear_idx*2+1})*3)':eval=frame:flags=bilinear,"
        f"scale={width}:{height}:flags=neighbor:eval=frame,format=gray,setsar=1[g];"
        f"[base][g]blend=c0_mode=addition:c1_mode=normal:c2_mode=normal:c0_opacity=0.22,format=yuv444p[tmp]"
    )
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(raw_png),
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=1:d=1",
        "-filter_complex", filter_complex,
        "-map", "[tmp]",
        "-frames:v", "1",
        "-q:v", "2",
        str(out_png),
    ]
    return cmd


def _draw_centered_text_pil(
    base_png: Path,
    text: str,
    out_png: Path,
    fontfile: str,
    fontsize: int,
    width: int,
    height: int,
) -> None:
    """Draw Courier New centered per-line (not left-justified block) onto base_png."""
    wrapped = escape_drawtext_textfile(text, wrap_width=_wrap_for_fs(fontsize))
    lines = [l for l in wrapped.split("\n") if l != ""]
    if not lines:
        # no text — just copy
        Image.open(base_png).save(out_png)
        return
    font = ImageFont.truetype(fontfile, fontsize)
    line_spacing = 10
    # Measure with stroke to get accurate total height
    # Use a dummy draw to measure bbox with stroke
    dummy = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    heights: list[int] = []
    for l in lines:
        bbox = d.textbbox((0, 0), l, font=font, stroke_width=4, anchor="lt")
        h = bbox[3] - bbox[1]
        heights.append(h)
    total_h = sum(heights) + (len(lines) - 1) * line_spacing
    y0 = (height - total_h) // 2

    im = Image.open(base_png).convert("RGBA")
    draw = ImageDraw.Draw(im)
    y = y0
    for idx, line in enumerate(lines):
        # shadow first (offset 2,2, black, no stroke, translucent)
        draw.text(
            (width // 2 + 2, y + 2),
            line,
            font=font,
            fill=(0, 0, 0, 180),
            anchor="mt",
        )
        # main: white with black stroke 4
        draw.text(
            (width // 2, y),
            line,
            font=font,
            fill="white",
            stroke_width=4,
            stroke_fill="black",
            anchor="mt",
        )
        h = heights[idx]
        y += h + line_spacing
    # flatten to RGB for PNG (keep quality)
    im.convert("RGB").save(out_png, "PNG")


def build_overlay_cmd(
    raw_png: Path,
    text: str,
    out_png: Path,
    fontfile: str,
    fontsize: int,
    crop: float,
    width: int,
    height: int,
    ffmpeg_bin: str,
    tmpdir: Path,
    tear_idx: int = 0,
) -> list[str]:
    """Legacy ffmpeg drawtext cmd (left-justified block) — kept for shell script export.
    Actual overlay_image now uses PIL centered rendering. This cmd is only written to
    create_overlay_frame_*.sh for reference; prefer running the Python path for centered output.
    """
    wrapped = escape_drawtext_textfile(text, wrap_width=_wrap_for_fs(fontsize))
    tf = tmpdir / "text.txt"
    tf.write_text(wrapped, encoding="utf-8")
    ff = fontfile.replace("'", r"\'")
    tf_str = str(tf).replace("'", r"\'")
    filter_complex = (
        f"[0:v]crop=w='iw*(1-2*{crop})':h='ih*(1-2*{crop})':x='iw*{crop}':y='ih*{crop}',"
        f"scale={width}:{height}:flags=lanczos,setsar=1,curves=vintage,"
        f"colorbalance=rs=0.015:gs=0.004:bs=-0.015:rm=0.008:gm=0.002:bm=-0.008,"
        f"eq=contrast=1.015:brightness=0:saturation=0.97,format=yuv444p,setsar=1[base];"
        f"[1:v]noise=alls=80:allf=t:all_seed={tear_idx*11},"
        f"scale=w='iw/(2+random({tear_idx*2})*3)':h='ih/(2+random({tear_idx*2+1})*3)':eval=frame:flags=bilinear,"
        f"scale={width}:{height}:flags=neighbor:eval=frame,format=gray,setsar=1[g];"
        f"[base][g]blend=c0_mode=addition:c1_mode=normal:c2_mode=normal:c0_opacity=0.22,format=yuv444p,format=yuv444p[tmp];"
        f"[tmp]drawtext=fontfile='{ff}':textfile='{tf_str}':"
        f"fontcolor=white:fontsize={fontsize}:borderw=4:bordercolor=black:"
        f"shadowx=2:shadowy=2:box=0:line_spacing=10:"
        f"x=(w-text_w)/2:y=(h-text_h)/2,format=yuv444p[v]"
    )
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(raw_png),
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=1:d=1",
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-frames:v", "1",
        "-q:v", "2",
        str(out_png),
    ]
    return cmd


def overlay_image(
    raw_png: Path,
    text: str,
    out_png: Path,
    fontfile: str,
    fontsize: int,
    crop: float,
    width: int,
    height: int,
    ffmpeg_bin: str,
    tear_idx: int,
) -> bool:
    # Two-stage: ffmpeg grain+vintage -> temp, then PIL centered text
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        tmp_base = td_p / "base.png"
        cmd = _build_grain_cmd(raw_png, tmp_base, crop, width, height, ffmpeg_bin, tear_idx)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [overlay error grain] {raw_png.name}: {r.stderr[-2000:]}")
            return False
        if not tmp_base.exists():
            print(f"  [overlay error] tmp_base missing for {raw_png.name}")
            return False
        try:
            _draw_centered_text_pil(tmp_base, text, out_png, fontfile, fontsize, width, height)
        except Exception as e:
            print(f"  [overlay error PIL] {raw_png.name}: {e}")
            return False
        return out_png.exists()


# ---------------------------------------------------------------------------
# Video: concat overlay pngs into MP4 (per carousel)
# ---------------------------------------------------------------------------

def build_carousel_video_cmd(
    overlay_pngs: list[Path],
    texts: list[str],
    out_mp4: Path,
    fontfile: str,
    fontsize: int,
    crop: float,
    width: int,
    height: int,
    fps: int,
    duration: float,
    ffmpeg_bin: str,
    tmpdir: Path,
) -> list[str]:
    n = len(overlay_pngs)
    # overlay pngs already have text — but we rebuild grain+text per segment with timing
    # so video shows each image for `duration` seconds with the same vintage filter as overlays.
    # Simpler: reuse overlay images as already-burned frames, just concat.
    # However to keep consistent vintage, we re-apply same per-segment pipeline here using raw images.
    # For simplicity we concat the overlay pngs (no re-burn).
    # Build concat demuxer approach: use -loop 1 per image.
    cmd: list[str] = [ffmpeg_bin, "-y"]
    for png in overlay_pngs:
        cmd += ["-loop", "1", "-t", f"{duration:.3f}", "-framerate", str(fps), "-i", str(png)]

    # concat n streams (overlay pngs already centered)
    total_dur = n * duration
    parts: list[str] = []
    for i in range(n):
        parts.append(f"[{i}:v]scale={width}:{height}:flags=lanczos,setsar=1,fps={fps}:round=near,setpts=PTS-STARTPTS[vs{i}]")
    vs_inputs = "".join(f"[vs{i}]" for i in range(n))
    parts.append(f"{vs_inputs}concat=n={n}:v=1:a=0,tpad=stop_mode=clone:stop_duration=0.36,format=yuv420p,fps={fps}[v]")
    filter_complex = ";\n".join(parts)
    cmd += ["-filter_complex", filter_complex, "-map", "[v]", "-r", str(fps), "-c:v", "libx264", "-crf", "19", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-t", f"{total_dur:.3f}", str(out_mp4)]
    return cmd


def build_carousel_video_from_raw_cmd(
    raw_pngs: list[Path],
    texts: list[str],
    out_mp4: Path,
    fontfile: str,
    fontsize: int,
    crop: float,
    width: int,
    height: int,
    fps: int,
    duration: float,
    ffmpeg_bin: str,
    tmpdir: Path,
) -> list[str]:
    """Per-segment grain+vintage+text like z_videos, but duration is uniform."""
    n = len(raw_pngs)

    def _wrap(fs: int) -> int:
        return _wrap_for_fs(fs)

    textfiles: list[Path] = []
    for i, txt in enumerate(texts):
        wrapped = escape_drawtext_textfile(txt, wrap_width=_wrap(fontsize))
        tf = tmpdir / f"seg_{i:03d}.txt"
        tf.write_text(wrapped, encoding="utf-8")
        textfiles.append(tf)

    cmd: list[str] = [ffmpeg_bin, "-y"]
    for png in raw_pngs:
        cmd += ["-loop", "1", "-t", f"{duration:.3f}", "-framerate", str(fps), "-i", str(png)]
    for i in range(n):
        cmd += ["-f", "lavfi", "-t", f"{duration:.3f}", "-r", str(fps), "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration:.3f}"]

    parts: list[str] = []
    for i in range(n):
        grain_idx = n + i
        parts.append(
            f"[{grain_idx}:v]noise=alls=80:allf=t:all_seed={i*11},"
            f"scale=w='iw/(2+random({i*2})*3)':h='ih/(2+random({i*2+1})*3)':eval=frame:flags=bilinear,"
            f"scale={width}:{height}:flags=neighbor:eval=frame,format=gray,setsar=1,fps={fps}[g{i}]"
        )
    for i in range(n):
        tf = textfiles[i]
        ff = fontfile.replace("'", r"\'")
        tf_str = str(tf).replace("'", r"\'")
        parts.append(
            f"[{i}:v]crop=w='iw*(1-2*{crop})':h='ih*(1-2*{crop})':x='iw*{crop}':y='ih*{crop}',"
            f"scale={width}:{height}:flags=lanczos,setsar=1,curves=vintage,"
            f"colorbalance=rs=0.015:gs=0.004:bs=-0.015:rm=0.008:gm=0.002:bm=-0.008,"
            f"eq=eval=frame:contrast=1.015:brightness='random({10+i})*0.03-0.015':saturation=0.97,"
            f"format=yuv444p,setsar=1,fps={fps}:round=near,setpts=PTS-STARTPTS[base{i}]"
        )
        parts.append(
            f"[base{i}][g{i}]blend=c0_mode=addition:c1_mode=normal:c2_mode=normal:c0_opacity=0.22,format=yuv444p,"
            f"setsar=1,fps={fps}:round=near,setpts=PTS-STARTPTS[tmp{i}]"
        )
        parts.append(
            f"[tmp{i}]drawtext=fontfile='{ff}':textfile='{tf_str}':"
            f"fontcolor=white:fontsize={fontsize}:borderw=4:bordercolor=black:"
            f"shadowx=2:shadowy=2:box=0:line_spacing=10:"
            f"x=(w-text_w)/2:y=(h-text_h)/2[vs{i}]"
        )
    concat_inputs = "".join(f"[vs{i}]" for i in range(n))
    total_dur = n * duration
    # ensure even dims for libx264 (768x1366 already even, but handle any odd custom size)
    parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0,scale=trunc(iw/2)*2:trunc(ih/2)*2,tpad=stop_mode=clone:stop_duration=0.36,format=yuv420p,fps={fps}[v]")
    filter_complex = ";\n".join(parts)
    cmd += ["-filter_complex", filter_complex, "-map", "[v]", "-r", str(fps), "-c:v", "libx264", "-crf", "19", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-t", f"{total_dur:.3f}", str(out_mp4)]
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate video-carousel images + overlays + MP4s from carousel JSONs")
    ap.add_argument("--input", type=str, default=INPUT_GLOB_DEFAULT, help="glob for carousel JSONs (relative to skill dir if not absolute)")
    ap.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT_DEFAULT), help="output root")
    ap.add_argument("--width", type=int, default=WIDTH, help="image/video width (default 768 for 9:16)")
    ap.add_argument("--height", type=int, default=HEIGHT, help="image/video height (default 1365 for 9:16)")
    ap.add_argument("--steps", type=int, default=STEPS, help="diffusion steps")
    ap.add_argument("--duration", type=float, default=DURATION_DEFAULT, help="seconds per frame in video")
    ap.add_argument("--fps", type=int, default=FPS, help="video fps")
    ap.add_argument("--fontsize", type=int, default=FONTSIZE_DEFAULT, help="overlay fontsize")
    ap.add_argument("--crop", type=float, default=CROP_DEFAULT, help="crop fraction per edge")
    ap.add_argument("--fontfile", type=str, default=None, help="Courier ttf path (auto-detected if omitted)")
    ap.add_argument("--style", choices=["soft", "noir"], default="soft", help="style hint palette (soft = muted candlelight, noir = Retro Avant-Garde Noir)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="print what would be done without generating")
    ap.add_argument("--skip-generate", action="store_true", help="skip sd-cli image generation, only overlay/video")
    ap.add_argument("--no-overlay", action="store_true", help="skip text overlay step")
    ap.add_argument("--no-video", action="store_true", help="skip MP4 concatenation")
    ap.add_argument("--only", type=str, default=None, help="only process JSON whose filename contains this substring")
    ap.add_argument("--carousel", type=str, default=None, help="only process carousel indices containing this substring (e.g. 0, 2-4)")
    args = ap.parse_args()

    style_hint = STYLE_HINT_SOFT if args.style == "soft" else STYLE_HINT_NOIR
    print(f"[style] {args.style}")
    if args.style == "soft":
        print(f"  hint: {style_hint[:90]}...")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        # allow relative to BASE_DIR
        output_root = (BASE_DIR / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)

    # discover inputs
    raw_glob = args.input
    if not Path(raw_glob).is_absolute():
        # input is relative to BASE_DIR
        search = BASE_DIR / raw_glob
        # glob with pattern
        import glob as globmod

        json_paths = sorted(Path(p) for p in globmod.glob(str(search)))
    else:
        import glob as globmod

        json_paths = sorted(Path(p) for p in globmod.glob(raw_glob))

    if args.only:
        json_paths = [p for p in json_paths if args.only in p.name]
        print(f"[filter] only={args.only!r} -> {len(json_paths)} file(s)")

    if not json_paths:
        print(f"No JSON found for pattern {raw_glob!r} (BASE_DIR={BASE_DIR})")
        raise SystemExit(1)

    # font / ffmpeg
    if not args.no_overlay:
        fontfile = args.fontfile or find_font()
        if not fontfile:
            print("[error] No monospace font found. Install Courier New or DejaVu Sans Mono, or pass --fontfile")
            raise SystemExit(1)
        if not Path(fontfile).exists():
            print(f"[error] fontfile not found: {fontfile}")
            raise SystemExit(1)
        print(f"[font] {fontfile} fontsize={args.fontsize}")
        ffmpeg_bin = find_ffmpeg()
        print(f"[ffmpeg] {ffmpeg_bin}")
        try:
            r = subprocess.run([ffmpeg_bin, "-filters"], capture_output=True, text=True)
            if "drawtext" not in r.stdout:
                print(f"[error] {ffmpeg_bin} lacks drawtext (need libfreetype). Try brew install ffmpeg-full")
                raise SystemExit(1)
        except Exception as e:
            print(f"[warn] could not verify drawtext: {e}")
    else:
        fontfile = args.fontfile or find_font() or ""
        ffmpeg_bin = find_ffmpeg()

    for json_path in json_paths:
        stem = json_path.stem
        try:
            carousels = load_carousels(json_path)
        except Exception as e:
            print(f"[skip] {json_path.name}: {e}")
            continue
        print(f"\n[{stem}] {len(carousels)} carousel(s)")
        # optional carousel filter
        indices = list(range(len(carousels)))
        if args.carousel is not None:
            # support "0", "2,4", "1-3"
            wanted: set[int] = set()
            for part in args.carousel.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    try:
                        for k in range(int(a), int(b) + 1):
                            wanted.add(k)
                    except ValueError:
                        pass
                else:
                    try:
                        wanted.add(int(part))
                    except ValueError:
                        pass
            indices = [i for i in indices if i in wanted]
            print(f"  [carousel filter] {args.carousel!r} -> {indices}")

        for c_idx in indices:
            frames = carousels[c_idx]
            if not frames:
                print(f"  [skip] carousel {c_idx:03d}: empty")
                continue
            carousel_dir = output_root / stem / f"carousel_{c_idx:03d}"
            carousel_dir.mkdir(parents=True, exist_ok=True)

            # persist carousel meta for debugging
            meta_path = carousel_dir / "carousel.json"
            if not meta_path.exists() or args.force:
                meta_path.write_text(json.dumps(frames, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"\n  [carousel {c_idx:03d}] {len(frames)} frame(s) -> {carousel_dir.relative_to(BASE_DIR) if carousel_dir.is_relative_to(BASE_DIR) else carousel_dir}")
            for j, fr in enumerate(frames):
                print(f"    {j}: {fr.get('text','')[:60]!r}")

            # --- Phase 1: generate raw images ---
            raw_pngs: list[Path] = []
            texts: list[str] = []
            for j, fr in enumerate(frames):
                text = (fr.get("text") or "").strip()
                visual = (fr.get("visual_description") or "").strip()
                if not visual:
                    print(f"    [warn] frame {j} has no visual_description, skipping gen")
                    continue
                texts.append(text)
                raw_png = carousel_dir / f"frame_{j:02d}.png"
                raw_pngs.append(raw_png)
                # write sidecars
                (carousel_dir / f"frame_{j:02d}.txt").write_text(text, encoding="utf-8")
                (carousel_dir / f"frame_{j:02d}.desc.txt").write_text(visual, encoding="utf-8")

                if args.dry_run:
                    print(f"    [dry-run gen] {raw_png.name}")
                    continue
                if args.skip_generate:
                    if not raw_png.exists():
                        print(f"    [skip gen] missing {raw_png.name} (use without --skip-generate)")
                    else:
                        print(f"    [skip gen] {raw_png.name} exists")
                    continue
                if raw_png.exists() and not args.force:
                    print(f"    [skip gen] {raw_png.name} exists (use --force)")
                    continue
                prompt = build_prompt(visual, style_hint)
                print(f"    [gen] {raw_png.name} steps={args.steps} {args.width}x{args.height}")
                try:
                    generate(
                        prompt,
                        diffusion_model=DIFFUSION_MODEL,
                        vae=VAE,
                        llm=LLM,
                        output=raw_png,
                        width=args.width,
                        height=args.height,
                        steps=args.steps,
                    )
                except Exception as e:
                    print(f"    [gen error] {raw_png.name}: {e}")
                    continue
                if raw_png.exists():
                    sz = raw_png.stat().st_size / 1024
                    print(f"      -> {sz:.0f} KB")

            # --- Phase 2: overlay ---
            overlay_pngs: list[Path] = []
            if not args.no_overlay:
                subs_dir = carousel_dir / "_subs_txt"
                subs_dir.mkdir(exist_ok=True)
                for j, (raw_png, text) in enumerate(zip(raw_pngs, texts)):
                    overlay_png = carousel_dir / f"frame_{j:02d}_overlay.png"
                    overlay_pngs.append(overlay_png)
                    wrapped = escape_drawtext_textfile(text, wrap_width=_wrap_for_fs(args.fontsize))
                    (subs_dir / f"seg_{j:03d}.txt").write_text(wrapped, encoding="utf-8")

                    if args.dry_run:
                        print(f"    [dry-run overlay] {overlay_png.name}")
                        continue
                    if not raw_png.exists():
                        print(f"    [overlay skip] raw missing {raw_png.name}")
                        continue
                    if overlay_png.exists() and not args.force:
                        print(f"    [skip overlay] {overlay_png.name} exists")
                        continue
                    print(f"    [overlay] {raw_png.name} -> {overlay_png.name}")
                    ok = overlay_image(raw_png, text, overlay_png, fontfile, args.fontsize, args.crop, args.width, args.height, ffmpeg_bin, tear_idx=c_idx * 10 + j)
                    if ok:
                        sz = overlay_png.stat().st_size / 1024
                        print(f"      -> {sz:.0f} KB")
                    # write per-frame overlay shell script
                    sh_path = carousel_dir / f"create_overlay_frame_{j:02d}.sh"
                    with tempfile.TemporaryDirectory() as td:
                        td_p = Path(td)
                        cmd = build_overlay_cmd(raw_png, text, overlay_png, fontfile, args.fontsize, args.crop, args.width, args.height, ffmpeg_bin, td_p, tear_idx=c_idx * 10 + j)
                        # replace tmp text path with subs_dir for reproducibility
                        cmd_str = " ".join(shlex.quote(c.replace(str(td_p / "text.txt"), str(subs_dir / f"seg_{j:03d}.txt"))) for c in cmd)
                        sh_path.write_text("#!/bin/bash\nset -e\n" + cmd_str + f'\necho "done {overlay_png}"\n', encoding="utf-8")
                        sh_path.chmod(0o755)
            else:
                # if no-overlay, overlay list is raw list for video fallback
                overlay_pngs = raw_pngs
                print("  [no-overlay] skipping text burn")

            # --- Phase 3: video ---
            if args.no_video:
                print("  [no-video] skipping MP4")
                continue
            if args.dry_run:
                print(f"    [dry-run video] carousel_{c_idx:03d}.mp4 {len(raw_pngs)}x{args.duration:.1f}s fps={args.fps} {args.width}x{args.height}")
                continue
            out_mp4 = carousel_dir / f"carousel_{c_idx:03d}.mp4"
            if out_mp4.exists() and not args.force:
                print(f"  [skip video] {out_mp4.name} exists (use --force)")
                continue
            # Prefer centered overlay PNGs (PIL, per-line centered) when available.
            # Falls back to raw re-burn only if overlays missing / no-overlay mode.
            use_overlays = (
                not args.no_overlay
                and len(overlay_pngs) == len(texts)
                and overlay_pngs
                and all(p.exists() for p in overlay_pngs)
            )
            if use_overlays:
                print(f"  [video] {len(overlay_pngs)} frames x{args.duration:.1f}s -> {out_mp4.name} {args.width}x{args.height} fps={args.fps} (from centered overlays)")
                with tempfile.TemporaryDirectory() as td:
                    td_p = Path(td)
                    cmd = build_carousel_video_cmd(overlay_pngs, texts, out_mp4, fontfile, args.fontsize, args.crop, args.width, args.height, args.fps, args.duration, ffmpeg_bin, td_p)
                    sh_path = carousel_dir / "create_video.sh"
                    sh_lines = [
                        "#!/bin/bash",
                        "set -e",
                        f"# auto-generated carousel {c_idx:03d} from {json_path.name} (centered PIL overlays)",
                        f"# font: {fontfile} fontsize={args.fontsize} crop={args.crop} fps={args.fps} {args.width}x{args.height} dur={args.duration}",
                        " ".join(shlex.quote(c) for c in cmd),
                        f'echo "done {out_mp4}"',
                        f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "{out_mp4}" || true',
                        f'ls -lh "{out_mp4}" || true',
                    ]
                    sh_path.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
                    sh_path.chmod(0o755)
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  [video error] {out_mp4.name}: {r.stderr[-4000:]}")
                        continue
                    if out_mp4.exists():
                        sz = out_mp4.stat().st_size / (1024 * 1024)
                        print(f"  [video done] {out_mp4.name} {sz:.1f} MB")
                        try:
                            r2 = subprocess.run([str(Path(ffmpeg_bin).with_name("ffprobe")) if Path(str(Path(ffmpeg_bin).with_name("ffprobe"))).exists() else "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out_mp4)], capture_output=True, text=True)
                            if r2.stdout.strip():
                                print(f"    duration: {r2.stdout.strip()}s")
                        except Exception:
                            pass
            else:
                existing_raw = [(p, t) for p, t in zip(raw_pngs, texts) if p.exists()]
                if not existing_raw:
                    print(f"  [video skip] no raw images in {carousel_dir.name}")
                    continue
                raw_for_video, texts_for_video = zip(*existing_raw)  # type: ignore
                raw_for_video = list(raw_for_video)
                texts_for_video = list(texts_for_video)
                print(f"  [video] {len(raw_for_video)} frames x{args.duration:.1f}s -> {out_mp4.name} {args.width}x{args.height} fps={args.fps}")
                with tempfile.TemporaryDirectory() as td:
                    td_p = Path(td)
                    subs_dir = carousel_dir / "_subs_txt"
                    for k in range(len(texts_for_video)):
                        src = subs_dir / f"seg_{k:03d}.txt"
                        if src.exists():
                            shutil.copy(src, td_p / f"seg_{k:03d}.txt")
                    cmd = build_carousel_video_from_raw_cmd(raw_for_video, texts_for_video, out_mp4, fontfile, args.fontsize, args.crop, args.width, args.height, args.fps, args.duration, ffmpeg_bin, td_p)
                    sh_path = carousel_dir / "create_video.sh"
                    cmd_shell = []
                    for c in cmd:
                        if str(td_p) in c:
                            c = c.replace(str(td_p), str(subs_dir))
                        cmd_shell.append(c)
                    sh_lines = [
                        "#!/bin/bash",
                        "set -e",
                        f"# auto-generated carousel {c_idx:03d} from {json_path.name}",
                        f"# font: {fontfile} fontsize={args.fontsize} crop={args.crop} fps={args.fps} {args.width}x{args.height} dur={args.duration}",
                        f"# subs: {subs_dir}",
                        " ".join(shlex.quote(c) for c in cmd_shell),
                        f'echo "done {out_mp4}"',
                        f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "{out_mp4}" || true',
                        f'ls -lh "{out_mp4}" || true',
                    ]
                    sh_path.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
                    sh_path.chmod(0o755)
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  [video error] {out_mp4.name}: {r.stderr[-4000:]}")
                        continue
                    if out_mp4.exists():
                        sz = out_mp4.stat().st_size / (1024 * 1024)
                        print(f"  [video done] {out_mp4.name} {sz:.1f} MB")
                        try:
                            r2 = subprocess.run([str(Path(ffmpeg_bin).with_name("ffprobe")) if Path(str(Path(ffmpeg_bin).with_name("ffprobe"))).exists() else "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out_mp4)], capture_output=True, text=True)
                            if r2.stdout.strip():
                                print(f"    duration: {r2.stdout.strip()}s")
                        except Exception:
                            pass


if __name__ == "__main__":
    main()
