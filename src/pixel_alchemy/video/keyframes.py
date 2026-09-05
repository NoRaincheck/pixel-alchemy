"""Video keyframe extraction and transcript-based renaming.

Combines ffprobe I-frame detection with transcript segment midpoints for
carousel-ready frames, plus a slug helper for renaming videos from
transcripts. Each frame gets a .txt sidecar and a manifest.json with
its associated transcript text. Optional visual descriptions via tau-ai
+ LMStudio (Qwen) are saved as .desc.txt and in the manifest as
visual_description.

Example::

    from pixel_alchemy.video.keyframes import extract_keyframes, slugify
    from pixel_alchemy.transcribe.english_transcribe import transcribe_english

    result = transcribe_english("video.mp4")
    slug = slugify(result["full_text"])
    n = extract_keyframes("video.mp4", f"keyframes/{slug}", segments=result["segments"])
    # writes frame_*.jpg + frame_*.txt + manifest.json
    # with visual descriptions: extract_keyframes(..., describe_visual=True)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import subprocess
from pathlib import Path

import cv2
from PIL import Image

VISUAL_PROMPT = (
    "Describe this image in precise visual detail so an AI image generator could recreate it. "
    "Include: subject(s), pose, clothing, facial expression, lighting, colors, background, "
    "composition, style, mood, and camera framing. "
    "Ignore and do not mention, transcribe, or describe any overlaid text, subtitles, captions, or watermarks. "
    "Focus only on the visual scene. Be concise but thorough (80-120 words)."
)


def slugify(text: str, max_words: int = 8, max_len: int = 60) -> str:
    """Turn transcript text into a filesystem-safe slug."""
    low = text.lower().strip()
    m = re.match(r"how true it was when ([a-z ]+) (said|asked)[:,\s]*", low)
    author = ""
    if m:
        author = re.sub(r"[^a-z0-9]+", "-", m.group(1).strip()).strip("-")
        low = re.sub(r"^how true it was when [^:,\"]+ (said|asked)[:,\s]*", "", low)
    low = low.split(".")[0]
    low = re.sub(r'"', "", low)
    low = re.sub(r"[^a-z0-9\s-]", "", low)
    words = low.split()
    slug = "-".join(words[:max_words])
    if author:
        slug = f"{author}-{slug}"
    slug = slug[:max_len].rstrip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug or "untitled"


def get_iframe_times(video: str | Path) -> list[float]:
    """Return I-frame timestamps via ffprobe."""
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-skip_frame",
            "nokey",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    times: list[float] = []
    for line in r.stdout.splitlines():
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                times.append(float(parts[1]))
            except ValueError:
                continue
    return sorted(times)


def _text_for_ts(ts: float, segments: list[dict] | None) -> tuple[str, dict | None]:
    """Return (text, segment) for ts — exact hit or nearest segment."""
    if not segments:
        return "", None
    for seg in segments:
        if seg["start"] - 0.05 <= ts <= seg["end"] + 0.05:
            return seg["text"], seg
    # fallback: nearest by midpoint distance
    best = min(segments, key=lambda s: abs(ts - (s["start"] + s["end"]) / 2))
    return best["text"], best


def save_frame_at(video: str | Path, out_path: str | Path, ts: float) -> bool:
    """Seek to ts seconds and write a JPEG frame."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), frame)


def _image_to_base64(image_path: Path, max_size: int = 512) -> tuple[str, str]:
    """Resize image and return (b64, mime_type)."""
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


async def _describe_with_tau(
    image_path: Path,
    *,
    prompt: str = VISUAL_PROMPT,
    base_url: str = "http://localhost:1234/v1",
    model: str = "qwen3.6-35b-a3b-mtp",
    api_key: str = "lm-studio",
    timeout_seconds: float = 120,
) -> str:
    """Describe image via tau-ai + LMStudio OpenAI-compatible endpoint."""
    try:
        from tau_agent.messages import ImageContent, TextContent, UserMessage
        from tau_ai import OpenAICompatibleProvider
        from tau_ai.env import OpenAICompatibleConfig
    except ImportError as e:
        raise RuntimeError("tau-ai not installed; install with pip install 'pixel-alchemy[tau]'") from e

    b64, mime = _image_to_base64(image_path)
    config = OpenAICompatibleConfig(
        api_key=api_key,
        base_url=base_url,
        supports_images=True,
        timeout_seconds=timeout_seconds,
    )
    provider = OpenAICompatibleProvider(config)
    msg = UserMessage(content=[TextContent(text=prompt), ImageContent(data=b64, mime_type=mime)])
    full = ""
    try:
        async for event in provider.stream_response(
            model=model,
            system="You are a helpful image describer for AI generation.",
            messages=[msg],
            tools=[],
        ):
            if event.__class__.__name__ == "TextDeltaEvent":
                full += event.delta  # type: ignore[attr-defined]
    finally:
        await provider.aclose()
    return full.strip()


def describe_image(
    image_path: str | Path,
    *,
    base_url: str = "http://localhost:1234/v1",
    model: str = "qwen3.6-35b-a3b-mtp",
    prompt: str = VISUAL_PROMPT,
) -> str:
    """Sync wrapper for _describe_with_tau."""
    return asyncio.run(
        _describe_with_tau(Path(image_path), prompt=prompt, base_url=base_url, model=model)
    )


def extract_keyframes(
    video: str | Path,
    out_dir: str | Path,
    *,
    segments: list[dict] | None = None,
    extra_times: list[float] | None = None,
    dedup_threshold: float = 0.4,
    describe_visual: bool = False,
    visual_base_url: str = "http://localhost:1234/v1",
    visual_model: str = "qwen3.6-35b-a3b-mtp",
) -> int:
    """Extract I-frames + segment midpoints to out_dir.

    Each frame gets a .txt sidecar with its transcript text and a
    manifest.json is written for the whole directory. When describe_visual
    is True, each frame also gets a .desc.txt with a tau-ai/LMStudio
    visual description (ignoring overlaid text) suitable for AI image
    generation.

    Args:
        video: Input video path.
        out_dir: Directory for JPEG frames.
        segments: Transcription segments with start/end keys.
        extra_times: Additional timestamps to include.
        dedup_threshold: Merge timestamps closer than this (seconds).
        describe_visual: Whether to generate visual descriptions via tau-ai.
        visual_base_url: LMStudio OpenAI-compatible base URL.
        visual_model: Model name for visual description.

    Returns:
        Number of frames written.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iframe_times = get_iframe_times(video)
    seg_times = [(s["start"] + s["end"]) / 2 for s in segments] if segments else []
    if extra_times:
        seg_times.extend(extra_times)

    all_times = sorted(set(iframe_times + seg_times))
    dedup: list[float] = []
    for t in all_times:
        if not dedup or abs(t - dedup[-1]) > dedup_threshold:
            dedup.append(t)

    if len(dedup) < 3:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        dur = frames / fps if fps else 0
        for frac in [0.25, 0.5, 0.75]:
            t = dur * frac
            if all(abs(t - d) > dedup_threshold for d in dedup):
                dedup.append(t)
        dedup.sort()

    manifest: list[dict] = []
    count = 0
    for i, ts in enumerate(dedup):
        out = out_dir / f"frame_{i:03d}_{ts:06.2f}s.jpg"
        if out.exists():
            # resume: don't overwrite existing frame
            saved = True
        else:
            saved = save_frame_at(video, out, ts)
            if not saved:
                continue
        count += 1
        text, seg = _text_for_ts(ts, segments)
        txt_path = out.with_suffix(".txt")
        if txt_path.exists() and txt_path.stat().st_size > 0:
            # don't overwrite existing transcript sidecar
            text = txt_path.read_text(encoding="utf-8")
        else:
            txt_path.write_text(text, encoding="utf-8")
        visual_desc = ""
        desc_path = out.parent / f"{out.stem}.desc.txt"
        if describe_visual:
            if desc_path.exists() and desc_path.stat().st_size > 20:
                existing = desc_path.read_text(encoding="utf-8").strip()
                if existing and not existing.startswith("[visual description failed"):
                    visual_desc = existing
                else:
                    # retry failed placeholder
                    try:
                        visual_desc = describe_image(out, base_url=visual_base_url, model=visual_model)
                    except Exception as e:  # noqa: BLE001
                        visual_desc = f"[visual description failed: {e}]"
                        print(f"    [warn] visual describe failed for {out.name}: {e}")
                    desc_path.write_text(visual_desc, encoding="utf-8")
            else:
                try:
                    visual_desc = describe_image(out, base_url=visual_base_url, model=visual_model)
                except Exception as e:  # noqa: BLE001
                    visual_desc = f"[visual description failed: {e}]"
                    print(f"    [warn] visual describe failed for {out.name}: {e}")
                desc_path.write_text(visual_desc, encoding="utf-8")
        elif desc_path.exists():
            # keep existing visual description in manifest even when not regenerating
            visual_desc = desc_path.read_text(encoding="utf-8").strip()
        manifest.append(
            {
                "file": out.name,
                "timestamp": round(ts, 2),
                "text": text,
                "visual_description": visual_desc,
                "segment": seg,
            }
        )

    full_text = " ".join(s["text"] for s in segments) if segments else ""
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {"video": video.name, "full_text": full_text, "segments": segments or [], "frames": manifest},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return count


def rename_by_transcript(video: str | Path, transcript: str) -> Path:
    """Rename video to slug derived from transcript.

    Returns new path (no-op if already correct, adds -2 suffix on collision).
    """
    video = Path(video)
    slug = slugify(transcript)
    target = video.with_name(f"{slug}.mp4")
    if target.resolve() == video.resolve():
        return target
    base = slug
    counter = 1
    while target.exists():
        counter += 1
        slug = f"{base}-{counter}"
        target = video.with_name(f"{slug}.mp4")
        if target.resolve() == video.resolve():
            break
    video.rename(target)
    return target
