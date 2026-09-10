---
name: video-carousel
description: Generate vertical 9:16 video carousels from text+visual JSON — prompt ideas in thefallenpoet noir style, render with Z-image/SD, burn centered typewriter overlays, and concat to MP4. Use when creating philosophical/poetic reel carousels.
---

# Video Carousel

End-to-end pipeline for **text + image + video carousels** (Instagram Reels / TikTok 9:16):

1. **Prompt** an LLM to generate carousel ideas (JSON `List[List[Frame]]`) in the style of `thefallenpoet` — melancholic, philosophical, noir — for any topic.
2. **Render** each `visual_description` to a 768×1366 painterly image via `stable-diffusion.cpp` Z-image (`sd_cli.generate`) + style hint.
3. **Overlay** centered Courier New typewriter text with grain + vintage (`ffmpeg` curves/colorbalance + PIL centered draw) → `frame_*_overlay.png`.
4. **Concat** overlays into `carousel_*.mp4` (8 fps, ~3 s/frame, optional audio via `soundscape` skill).

Derived from `video-carousel/PROMPT_GENERATE_CAROUSEL*.md` + `video-carousel/*.py` (notably `generate_bible_grief_images.py`) with topic-specific references removed and generalized for any subject.

## Input JSON format

Each file is `List[Carousel]` where `Carousel = List[Frame]` (3–7 frames, ideal 5):

```json
[
  [
    {"text": "short overlay text — 1 line, poetic hook", "visual_description": "Flux/SD image prompt — shot + subject + lighting + palette + mood + style"},
    {"text": "...", "visual_description": "..."}
  ],
  [
    {"text": "...", "visual_description": "..."}
  ]
]
```

Also accepts flat `List[Frame]` (single carousel, auto-wrapped). `text` is burned centered; `visual_description` is appended to the style hint and sent to diffusion.

## Step 1 — Generate carousel ideas (LLM prompt)

Copy the prompt below, set `{{TOPIC}}`, attach `thefallenpoet/keyframes-flattened/combined.json` (or paste `REFERENCE_JSON`), and save the LLM output as `input/<topic>.json`.

See `PROMPT_GENERATE_CAROUSEL.md` for the full prompt (with embedded `combined.json` fallback). Quick template:

```
You are generating video carousels in the style of `thefallenpoet` — melancholic, philosophical, noir.
Reference: the attached combined.json — 6 exemplar carousels, 32 frames total, deduped.
Task: Create 20+ NEW carousel ideas about {{TOPIC}}. Do not copy text or visuals verbatim — match the DNA, not the content.
Each carousel is Array<Frame> (3-7 frames, ideal 5): {"text": "...", "visual_description": "..."}
Style DNA: Text is intimate 2nd-person or quoted philosopher, line-broken for pacing. Visual is Retro Avant-Garde Noir — hand-painted, film-noir chiaroscuro, deep blacks + warm amber/burnt-orange through crimson/maroon, vertical portrait, solitary/distanced pair, pensive mood.
Visual rules: Usable as Flux/SD prompt. Always include shot, pose/clothing, background, lighting, palette, mood, style keywords. Vary compositions but keep palette coherent.
Output: Single JSON — list of 20+ carousels: [[{"text":"...","visual_description":"..."}, ...], ...]. No prose outside JSON. Validate JSON.
```

Keep `noir` visuals even when the topic is soft — switch only the `--style soft` palette if you want muted candlelight instead of harsh noir.

## Step 2 — Render images + overlays + video

```bash
# from repo root, requires sd_cpp models + ffmpeg-full with drawtext

# dry-run: preview what would be generated
./.agents/skills/video-carousel/generate_carousel.py --input "input/*.json" --output-root output/ --dry-run

# render all carousels in a file (768x1366, 8 steps, noir palette)
./.agents/skills/video-carousel/generate_carousel.py --input "input/my-topic.json" --output-root output/

# softer palette (muted slate/indigo + candlelight)
./.agents/skills/video-carousel/generate_carousel.py --input "input/my-topic.json" --output-root output/ --style soft

# single carousel index, overwrite, or skip phases
./.agents/skills/video-carousel/generate_carousel.py --input "input/my-topic.json" --carousel 2 --force
./.agents/skills/video-carousel/generate_carousel.py --input "input/my-topic.json" --skip-generate --no-video
./.agents/skills/video-carousel/generate_carousel.py --input "input/my-topic.json" --no-overlay

# custom dimensions / timing / font
./.agents/skills/video-carousel/generate_carousel.py --input "input/*.json" --width 768 --height 1366 --duration 3.0 --fps 8 --fontsize 52 --crop 0.04
```

Inputs are discovered by glob relative to the skill dir (or absolute). Outputs mirror the original `output/<stem>/carousel_*/` layout but generic:

```
<output-root>/<json-stem>/carousel_000/
  carousel.json
  frame_00.png            # raw diffusion output
  frame_00.txt / .desc.txt
  frame_00_overlay.png    # grain+vintage+centered text
  _subs_txt/seg_000.txt
  create_overlay_frame_00.sh
  create_video.sh
  carousel_000.mp4
```

Pass `--only <substring>` to filter JSON filenames, `--carousel 0,2-4` to filter indices.

For the `thefallenpoet` extraction workflow (transcribe → keyframes + visual descriptions → Z-images → grain videos), run the originals in `video-carousel/`:

```bash
uv run python video-carousel/extract_keyframes.py            # transcribe + keyframes + tau visuals
uv run python video-carousel/generate_z_images.py --force    # .desc.txt -> z-image/*.png
uv run python video-carousel/generate_z_videos.py --force    # z-image + manifest -> _grain_subs.mp4 + layered soundscape audio
```

## Options (`generate_carousel.py`)

| option | default | meaning |
|---|---|---|
| `--input GLOB` | `*.json` | glob for carousel JSONs (relative to skill dir if not absolute) |
| `--output-root DIR` | `./output` | root for `carousel_*/` folders |
| `--style noir\|soft` | `soft` | `noir` = Retro Avant-Garde Noir (harsh amber/chiaroscuro), `soft` = muted cool shadows + candlelight |
| `--width N` | `768` | output width (9:16) |
| `--height N` | `1366` | output height (even, for libx264) |
| `--steps N` | `8` | diffusion steps |
| `--duration SEC` | `3.0` | seconds per frame in MP4 |
| `--fps N` | `8` | video fps |
| `--fontsize N` | `52` | Courier overlay size (auto-wraps via `_wrap_for_fs`) |
| `--crop F` | `0.04` | crop fraction per edge before scale |
| `--fontfile PATH` | auto | Courier TTF; auto-detects Courier New / DejaVu / Monaco |
| `--force` | off | overwrite existing PNG/MP4 |
| `--dry-run` | off | print actions, no generation |
| `--skip-generate` | off | skip `sd_cli.generate`, only overlay/video |
| `--no-overlay` | off | skip PIL+ffmpeg text burn |
| `--no-video` | off | skip MP4 concat |
| `--only SUBSTR` | — | only JSONs whose filename contains SUBSTR |
| `--carousel SPEC` | — | only carousel indices e.g. `0`, `2,4`, `1-3` |

## Style hints

Appended to every `visual_description` before `generate()`:

- **noir**: `Retro Avant-Garde Noir — Atmospheric & Moody` — fragmented geometry, 100% hand-painted, film-noir chiaroscuro, deep blacks + warm amber/burnt-orange through crimson/maroon, visible brushwork + ink outlines, mid-century retro palette.
- **soft**: muted slate blues / dusty greys / indigo shadows warmed by amber/honey candlelight, soft chiaroscuro, watercolor feathering, dusty rose/sage/cream/gold palette, intimate reverent mood.

Both emphasize *painterly, analogue, artist-painted texture — not CGI/photo*.

## Requirements

- `stable-diffusion.cpp` models at `~/.local/share/stable-diffusion.cpp/build/models/` (`z_image_turbo-Q3_K.gguf`, `flux1_schnell_diffusion_pytorch_model.safetensors`, `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`) or edit `DIFFUSION_MODEL/VAE/LLM` in script.
- `ffmpeg-full` with `drawtext` (`brew install ffmpeg-full`), Pillow, `pixel_alchemy.generation.sd_cli`, `uv` runner.
- Optional: `soundscape` skill for audio beds; `generate_z_videos.py` already layers mood-aware preset + diegetic sounds.

## Reference

- Prompt source: `video-carousel/PROMPT_GENERATE_CAROUSEL_embedded.md` (179 lines, 6 exemplars / 32 frames).
- Generation source: `video-carousel/*.py` — `generate_bible_grief_images.py` (795 lines), `generate_z_images.py`, `generate_z_videos.py` (910 lines), `extract_keyframes.py` — generalized here as `generate_carousel.py`.
- Original output example: `output/<stem>/carousel_*/{*.png,*_overlay.png,*.mp4,*.sh}` (e.g. 20 carousels per JSON, each `carousel_*/` with `frame_*.png`, `_overlay.png`, `carousel_*.mp4`, `create_*.sh`).
