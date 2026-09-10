"""MCP server exposing pixel-alchemy image tools over Streamable HTTP.

Tools:
- generate_image  — sd_cli generate with Z-image defaults (image generation only)
- remove_background — BiRefNet background removal
- remove_background_flood — flood-fill background removal by sampling 4 corners
- upscale_image   — upscayl_pipeline (multi-pass upscayl + blur + Lanczos downscale)
- list_images     — query available images in mcp_outputs (with metadata)
- get_image       — retrieve an image by path/filename from mcp_outputs

All tools accept/return base64-encoded images via MCP ImageContent, so they are
reachable from remote machines via any MCP client over Streamable HTTP
(host 0.0.0.0 by default).

Run:
    python -m pixel_alchemy.mcp.server --host 0.0.0.0 --port 1255
    uv run pixel-alchemy-mcp --host 0.0.0.0 --port 1255
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import TextContent
from pydantic import Field

# ---------------------------------------------------------------------------
# Defaults — Z-image (ZiT) profile
# ---------------------------------------------------------------------------
_DEFAULT_DIFFUSION = Path(
    os.getenv(
        "PIXEL_ALCHEMY_Z_DIFFUSION",
        "/Users/crn/.local/share/stable-diffusion.cpp/build/models/z_image_turbo-Q3_K.gguf",
    )
)
_DEFAULT_VAE = Path(
    os.getenv(
        "PIXEL_ALCHEMY_Z_VAE",
        "/Users/crn/.local/share/stable-diffusion.cpp/build/models/flux1_schnell_diffusion_pytorch_model.safetensors",
    )
)
_DEFAULT_LLM = Path(
    os.getenv(
        "PIXEL_ALCHEMY_Z_LLM",
        "/Users/crn/.local/share/stable-diffusion.cpp/build/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    )
)

# Persistent output dir inside repo — surives temp cleanup so remote chat can
# refer to files by path (e.g. for follow-up remove_background/upscale).
_REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(os.getenv("PIXEL_ALCHEMY_MCP_OUTPUT_DIR", str(_REPO_ROOT / "mcp_outputs")))

# Max HTTP body size — must allow >100MB b64 payloads (100MB b64 ~75MB binary + JSON overhead).
# Default 250 MiB; override via PIXEL_ALCHEMY_MCP_MAX_BODY_SIZE or --max-body-size.
# MCP SDK default is 4 MiB, which would reject large images with 413.
try:
    _DEFAULT_MAX_BODY_SIZE = int(os.getenv("PIXEL_ALCHEMY_MCP_MAX_BODY_SIZE", str(250 * 1024 * 1024)))
except ValueError:
    _DEFAULT_MAX_BODY_SIZE = 250 * 1024 * 1024


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _save_persistent(data: bytes, prefix: str, suffix: str = ".png") -> Path:
    d = _ensure_output_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    # prefix already describes tool, e.g. generate, foreground, upscaled
    p = d / f"{prefix}_{ts}_{uid}{suffix}"
    p.write_bytes(data)
    return p


def _cleanup_outputs() -> None:
    # No-op: outputs are preserved per user request (do not auto-delete mcp_outputs).
    return


@asynccontextmanager
async def _lifespan(server: MCPServer):  # type: ignore[no-untyped-def]
    _ensure_output_dir()
    yield


mcp = MCPServer(
    name="pixel-alchemy",
    instructions=(
        "Pixel Alchemy image tools: generate images with Z-image (sd-cli), "
        "remove backgrounds with BiRefNet or corner flood-fill (4-corner sampling), "
        "and upscale via upscayl-pipeline. "
        "Use list_images to see available outputs in mcp_outputs, and get_image to retrieve one by path/filename."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w/\-+]+);base64,(?P<data>.+)$", re.DOTALL)


def _strip_data_uri(b64: str) -> tuple[str, str | None]:
    """Return (raw_base64, mime_or_None). Handles plain base64 and data URIs."""
    b64 = b64.strip()
    m = _DATA_URI_RE.match(b64)
    if m:
        return m.group("data").strip(), m.group("mime")
    return b64, None


def _mime_to_suffix(mime: str | None) -> str:
    if mime is None:
        return ".png"
    m = mime.lower()
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "png" in m:
        return ".png"
    if "webp" in m:
        return ".webp"
    return ".png"


def _decode_base64_to_file(image_base64: str, tmp_dir: str) -> Path:
    raw, mime = _strip_data_uri(image_base64)
    # Remove whitespace/newlines (MCP clients may wrap base64) — needed before validate/size check.
    # For >100MB payloads this also avoids validate failing on line breaks.
    if len(raw) > 1024 and any(c in raw for c in ("\n", "\r", " ", "\t")):
        raw = "".join(raw.split())

    # For large payloads (>8 MiB encoded), decode in streaming chunks to avoid
    # peak memory of holding both the full b64 string and the full decoded bytes.
    # Chunk size must be a multiple of 4 to keep base64 quantum intact.
    LARGE_THRESHOLD = 8 * 1024 * 1024
    suffix = _mime_to_suffix(mime)
    if len(raw) > LARGE_THRESHOLD:
        # Validate quickly on a small prefix before streaming; fall back to full error if invalid.
        try:
            # Probe first 4KB with validate=True to surface malformed base64 early
            base64.b64decode(raw[:4096].strip(), validate=True)
        except Exception as e:
            # Full validate will raise with precise error; try full decode to report
            try:
                base64.b64decode(raw, validate=True)
            except Exception as e2:
                raise ValueError(f"Invalid base64 image data: {e2}") from e2
            raise ValueError(f"Invalid base64 image data: {e}") from e

        CHUNK_ENCODED = 4 * 1024 * 1024  # 4 MiB encoded -> ~3 MiB decoded per chunk
        # Ensure chunk size is multiple of 4
        assert CHUNK_ENCODED % 4 == 0

        # Probe magic from first chunk for suffix sniffing when mime not provided
        sniff_suffix = suffix
        if mime is None:
            try:
                first_bytes = base64.b64decode(raw[: min(len(raw), 32)], validate=False)
                if first_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                    sniff_suffix = ".png"
                elif first_bytes[:2] == b"\xff\xd8":
                    sniff_suffix = ".jpg"
            except Exception:  # noqa: BLE001, S110
                pass

        path = Path(tmp_dir) / f"input{sniff_suffix}"
        # Stream decode in chunks
        with open(path, "wb") as f:
            total_written = 0
            for i in range(0, len(raw), CHUNK_ENCODED):
                chunk = raw[i : i + CHUNK_ENCODED]
                # Pad last chunk if needed (base64 length must be %4==0)
                if len(chunk) % 4 != 0:
                    chunk += "=" * (4 - len(chunk) % 4)
                try:
                    # Use validate=False for chunked to allow padding variations; we've pre-validated
                    decoded = base64.b64decode(chunk, validate=False)
                except Exception as e:
                    raise ValueError(f"Invalid base64 image data at chunk {i // CHUNK_ENCODED}: {e}") from e
                f.write(decoded)
                total_written += len(decoded)
            if total_written == 0:
                raise ValueError("Decoded image is empty")
            # If sniff failed earlier due to short prefix, re-sniff from written file header
            if mime is None and sniff_suffix == ".png":
                # Already sniffed; if still default .png but file is actually jpg, the earlier sniff would have set it
                pass
        return path

    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e
    if not data:
        raise ValueError("Decoded image is empty")
    # sniff PNG/JPEG magic if no mime
    if mime is None:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            suffix = ".png"
        elif data[:2] == b"\xff\xd8":
            suffix = ".jpg"
    path = Path(tmp_dir) / f"input{suffix}"
    path.write_bytes(data)
    return path


def _coerce_to_str_input(value: object) -> str:
    """Coerce MCP-standard alternative input forms to a string for _load_image_input.

    Supports:
    - plain str (base64, data URI, file path, http(s) / file:// URL)
    - dict MCP ImageContent / EmbeddedResource style: {"data": "<b64>", "mimeType": "image/png"}
      or {"data": "<b64>", "mime_type": "..."} or {"content": {"data": ...}}
    - list containing a single such dict (some clients wrap content blocks in arrays)
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, dict):
        # MCP ImageContent: {type: "image", data: "<b64>", mimeType: "image/png"}
        # or BlobResourceContents: {uri: ..., blob: "<b64>", mimeType: ...}
        for key in ("data", "blob", "content"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                # if dict also has mimeType, prepend as data URI so _decode preserves suffix
                mime = value.get("mimeType") or value.get("mime_type") or value.get("mime_type") or ""
                if mime and key in ("data", "blob") and not v.strip().startswith("data:"):
                    # only wrap if v looks like base64 (not a path/URL)
                    try:
                        base64.b64decode(v.strip(), validate=True)
                        return f"data:{mime};base64,{v.strip()}"
                    except Exception:  # noqa: BLE001, S110
                        pass
                return v
        # nested content: {content: {data: ...}} or {image: {data: ...}}
        for v in value.values():
            if isinstance(v, dict):
                nested = _coerce_to_str_input(v)
                if nested.strip():
                    return nested
            if isinstance(v, str) and v.strip().startswith(("data:", "http://", "https://", "file://")):
                return v
        # fallback: if dict has uri pointing to file/http
        uri = value.get("uri") or value.get("url") or value.get("path")
        if isinstance(uri, str) and uri.strip():
            return uri.strip()
        # stringify dict as JSON fallback (will error downstream with clear message)
        import json

        return json.dumps(value)
    if isinstance(value, list) and value:
        # take first image-like entry
        for item in value:
            s = _coerce_to_str_input(item)
            if s.strip():
                return s
    return str(value)


def _load_image_input(image_input: str | object, tmp_dir: str) -> Path:
    """Accept base64, data URI, file:// URI, local file path, http(s) URL, or MCP ImageContent dict.

    MCP-standard alternatives handled:
    - Raw base64 string (with or without data URI prefix) -> decoded
    - data:image/...;base64,... URI
    - file:// URI -> local file
    - http(s):// URL -> fetched
    - Local file path (absolute, relative, or bare filename like mcp_outputs/...png)
    - Dict form MCP ImageContent / BlobResourceContents: {"data": "<b64>", "mimeType": "image/png"}
    """
    # Coerce non-string MCP content blocks
    if not isinstance(image_input, str):
        image_input = _coerce_to_str_input(image_input)
    s = image_input.strip()
    if not s:
        raise ValueError("image must be non-empty")

    # handle file:// URI explicitly (MCP standard resource URI)
    if s.startswith("file://"):
        # strip file:// and keep leading / for absolute paths
        file_part = s[len("file://") :]
        # handle file:///tmp/x.png -> /tmp/x.png
        if file_part.startswith("///"):
            file_part = file_part[2:]
        p = Path(file_part).expanduser()
        if p.exists():
            suffix = p.suffix.lower() or ".png"
            dest = Path(tmp_dir) / f"input{suffix}"
            import shutil

            shutil.copy(p, dest)
            return dest
        raise FileNotFoundError(f"File not found for file URI: {s} -> {p}")

    # 1. data URI
    if s.startswith("data:"):
        return _decode_base64_to_file(s, tmp_dir)

    # 2. http(s) URL — fetch
    if s.startswith(("http://", "https://")):
        dest = Path(tmp_dir) / "input.png"
        try:
            import httpx

            resp = httpx.get(s, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            # fix suffix from content-type or sniff
            ct = resp.headers.get("content-type", "")
            if "jpeg" in ct or "jpg" in ct:
                dest = dest.with_suffix(".jpg")
                Path(tmp_dir, "input.png").rename(dest) if Path(tmp_dir, "input.png").exists() else None
            return dest if dest.exists() else Path(tmp_dir) / "input.png"
        except ImportError:
            import urllib.request

            with urllib.request.urlopen(s, timeout=30) as r:
                data = r.read()
            dest.write_bytes(data)
            return dest
        except Exception as e:
            raise ValueError(f"Failed to fetch image URL {s}: {e}") from e

    # 3. local file path (handles bare filename like image-1788902666771.png)
    p = Path(s).expanduser()
    if p.exists():
        suffix = p.suffix.lower() or ".png"
        dest = Path(tmp_dir) / f"input{suffix}"
        import shutil

        shutil.copy(p, dest)
        return dest
    # also try relative to cwd / common search if bare filename
    if not p.is_absolute() and p.suffix:
        for base in [Path.cwd(), Path.cwd() / "images", Path("/tmp"), Path(tmp_dir)]:
            cand = base / p.name
            if cand.exists():
                dest = Path(tmp_dir) / f"input{cand.suffix.lower()}"
                import shutil

                shutil.copy(cand, dest)
                return dest
        # hint for LM Studio case: file was from previous generate_image
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            raise FileNotFoundError(
                f"File not found: {s} (tried {p.resolve()} and cwd/tmp). "
                "If this was a generated image, re-upload it or pass its base64/data URI instead."
            )

    # 4. fallback — treat as raw base64
    return _decode_base64_to_file(s, tmp_dir)


def _file_to_image(path: Path) -> Image:
    suffix = path.suffix.lower()
    fmt = {"png": "png", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(suffix, "png")
    return Image(path=str(path), format=fmt)


def _resolve_model_path(value: str | Path | None, default: Path) -> Path:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return Path(value)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def generate_image(
    prompt: Annotated[
        str,
        Field(
            description="Text prompt for generation. Be descriptive, e.g. 'a cinematic portrait of a cat in studio lighting'.",
            min_length=1,
        ),
    ],
    width: Annotated[int, Field(description="Image width in pixels.", ge=64, le=4096)] = 1024,
    height: Annotated[int, Field(description="Image height in pixels.", ge=64, le=4096)] = 1024,
    steps: Annotated[
        int, Field(description="Number of diffusion steps. Z-image is turbo so 8 is usually enough.", ge=1, le=100)
    ] = 8,
    cfg_scale: Annotated[
        float, Field(description="Classifier-free guidance scale. Keep 1.0 for sd-cli.", ge=0, le=20)
    ] = 1.0,
    sampling_method: Annotated[
        str, Field(description="Sampling method, e.g. 'euler', 'dpm++2m', 'euler_a'.")
    ] = "euler",
    diffusion_model: Annotated[
        str | None,
        Field(
            description="Override path to diffusion .gguf. Defaults to Z-image model (/Users/crn/.local/share/.../z_image_turbo-Q3_K.gguf or PIXEL_ALCHEMY_Z_DIFFUSION env)."
        ),
    ] = None,
    vae: Annotated[
        str | None,
        Field(description="Override path to VAE .safetensors. Defaults to flux1_schnell VAE (PIXEL_ALCHEMY_Z_VAE)."),
    ] = None,
    llm: Annotated[
        str | None,
        Field(description="Override path to LLM .gguf. Defaults to Qwen3-4B-Instruct (PIXEL_ALCHEMY_Z_LLM)."),
    ] = None,
) -> list[Image | TextContent]:  # type: ignore[valid-type]
    """Generate an image with sd-cli using the Z-image (ZiT) model by default.

    Args:
        prompt: Text prompt for generation. Be descriptive, e.g. "a cinematic portrait...".
        width: Image width in pixels (default 1024).
        height: Image height in pixels (default 1024).
        steps: Number of diffusion steps (default 8, Z-image is turbo).
        cfg_scale: Classifier-free guidance scale (keep 1.0 for sd-cli, default 1.0).
        sampling_method: Sampling method e.g. "euler", "dpm++2m" (default "euler").
        diffusion_model: Override path to diffusion .gguf. Defaults to Z-image model
            (/Users/crn/.local/share/.../z_image_turbo-Q3_K.gguf or PIXEL_ALCHEMY_Z_DIFFUSION env).
        vae: Override path to VAE .safetensors. Defaults to flux1_schnell VAE.
        llm: Override path to LLM .gguf. Defaults to Qwen3-4B-Instruct.

    Returns:
        PNG image + local path (mcp_outputs/...) so follow-up tools can use image_path.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be non-empty")

    dm = _resolve_model_path(diffusion_model, _DEFAULT_DIFFUSION)
    vae_path = _resolve_model_path(vae, _DEFAULT_VAE)
    llm_path = _resolve_model_path(llm, _DEFAULT_LLM)

    for p, label in [(dm, "diffusion_model"), (vae_path, "vae"), (llm_path, "llm")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p} (set env PIXEL_ALCHEMY_Z_* or pass explicit path)")

    # late import so server can start without sd-cli binary present (tools fail at call time)
    from pixel_alchemy.generation.sd_cli import generate

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "generated.png"
        generate(
            prompt=prompt,
            diffusion_model=dm,
            vae=vae_path,
            llm=llm_path,
            output=out,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            sampling_method=sampling_method,
        )
        data = out.read_bytes()
    persist = _save_persistent(data, "generate")
    rel = persist.relative_to(_REPO_ROOT) if persist.is_relative_to(_REPO_ROOT) else persist
    return [
        Image(data=data, format="png"),
        TextContent(
            type="text",
            text=f'Saved locally to {persist} (relative: {rel}) — for follow-up use image_path="{persist}" or image_base64 with this file.',
        ),
    ]


@mcp.tool()
def remove_background(
    image_base64: Annotated[
        str | dict,
        Field(
            description="Image input. Accepts: raw base64 string (no prefix, contentEncoding: base64, contentMediaType: image/png|image/jpeg|image/webp), data URI 'data:image/png;base64,...', local file path ('/tmp/out.png' or 'mcp_outputs/generate_....png'), file:// URI, http(s) URL, or MCP-standard ImageContent dict {'type':'image','data':'<base64>','mimeType':'image/png'} / BlobResourceContents {'blob':'<base64>','mimeType':'image/png'} / EmbeddedResource. If passing a path from a prior tool, you can also use image_path."
        ),
    ] = "",
    foreground: Annotated[
        bool,
        Field(
            description="If True (default), return RGBA cutout with transparent background. If False, return grayscale mask (white=foreground)."
        ),
    ] = True,
    fp16: Annotated[
        bool, Field(description="Use FP16 ONNX model (smaller/faster, default True). False for FP32.")
    ] = True,
    image_path: Annotated[
        str | None,
        Field(
            description="Alternative to image_base64 — explicit local file path or file:// URI. E.g. 'mcp_outputs/generate_...png' or '/tmp/image.png'."
        ),
    ] = None,
    image_url: Annotated[
        str | None,
        Field(description="Alternative to image_base64 — http(s) URL to fetch. E.g. 'https://example.com/image.jpg'."),
    ] = None,
) -> list[Image | TextContent]:  # type: ignore[valid-type]
    """Remove background from an image using BiRefNet.

    Args:
        image_base64: Base64 string, data URI (data:image/png;base64,...),
            local file path (e.g. image-1788902666771.png or /tmp/out.png or mcp_outputs/generate_....png),
            file:// URI, http(s) URL, or MCP ImageContent dict {"data": "<base64>", "mimeType": "image/png"}.
            LM Studio / chat-generated files can be passed as their file path directly — no manual base64 needed.
            Also accepts BlobResourceContents {"blob": "<base64>"} and raw base64 with contentEncoding base64.
        foreground: If True (default), return RGBA cutout with transparent background.
            If False, return grayscale mask (white=foreground).
        fp16: Use FP16 ONNX model (smaller/faster, default True). False for FP32.
        image_path: Alternative to image_base64 — explicit local file path or file:// URI.
        image_url: Alternative to image_base64 — http(s) URL to fetch.

    Returns:
        PNG image + local path (mcp_outputs/...) so it can be referenced in conversation.
    """
    # MCP-standard: accept str, dict ImageContent {data, mimeType}, BlobResource {blob}, or file:// URI
    raw_candidate = (
        image_path if image_path not in (None, "") else image_url if image_url not in (None, "") else image_base64
    )
    raw_input = _coerce_to_str_input(raw_candidate).strip()
    if not raw_input:
        raise ValueError(
            "image_base64 (or image_path/image_url) must be non-empty — expected base64 string, data URI, file path, file:// URI, http(s) URL, or MCP ImageContent dict {data, mimeType}"
        )

    from pixel_alchemy.background_removal.birefnet import birefnet

    with tempfile.TemporaryDirectory() as tmp:
        inp = _load_image_input(raw_input, tmp)
        out = Path(tmp) / ("foreground.png" if foreground else "mask.png")
        birefnet(inp, out, fp16=fp16, foreground=foreground)
        data = out.read_bytes()
    prefix = "foreground" if foreground else "mask"
    persist = _save_persistent(data, prefix)
    rel = persist.relative_to(_REPO_ROOT) if persist.is_relative_to(_REPO_ROOT) else persist
    return [
        Image(data=data, format="png"),
        TextContent(
            type="text",
            text=f'Saved locally to {persist} (relative: {rel}) — use image_path="{persist}" for next step.',
        ),
    ]


@mcp.tool()
def remove_background_flood(
    image_base64: Annotated[
        str | dict,
        Field(
            description="Image input. Accepts: raw base64 string (no prefix, contentEncoding: base64, contentMediaType: image/png|image/jpeg|image/webp), data URI 'data:image/png;base64,...', local file path ('/tmp/out.png' or 'mcp_outputs/generate_....png'), file:// URI, http(s) URL, or MCP-standard ImageContent dict {'type':'image','data':'<base64>','mimeType':'image/png'} / BlobResourceContents {'blob':'<base64>','mimeType':'image/png'} / EmbeddedResource. Same flexibility as remove_background."
        ),
    ] = "",
    tolerance: Annotated[
        int,
        Field(
            description="Max RGB Euclidean distance (0-441) for a pixel to match the sampled corner color (default 30). Higher = more tolerant. Ignored when force_match is True.",
            ge=0,
            le=442,
        ),
    ] = 30,
    force_match: Annotated[
        bool,
        Field(
            description="When True, auto-tune tolerance via binary search so all four corners flood-fill as the same region."
        ),
    ] = False,
    feather: Annotated[
        int,
        Field(description="Gaussian blur radius for soft mask edges. 0 = hard edges, 1-10 = feathered.", ge=0, le=100),
    ] = 0,
    foreground: Annotated[
        bool,
        Field(
            description="If True (default), return RGBA cutout with transparent background. If False, return background mask (white=background, black=foreground; invert for foreground matte)."
        ),
    ] = True,
    image_path: Annotated[
        str | None, Field(description="Alternative to image_base64 — explicit local file path or file:// URI.")
    ] = None,
    image_url: Annotated[str | None, Field(description="Alternative to image_base64 — http(s) URL to fetch.")] = None,
) -> list[Image | TextContent]:  # type: ignore[valid-type]
    """Remove background by sampling the 4 corners and flood-filling.

    Samples a small patch at each corner, picks the dominant corner color,
    then flood-fills from all four corners to build a background mask.
    Best for solid / near-solid backgrounds (e.g. white product shots).
    Uses OpenCV floodFill with mask-only mode, combined across 4 corners.

    Args:
        image_base64: Base64 string, data URI (data:image/png;base64,...),
            local file path (e.g. image-1788902666771.png or /tmp/out.png or mcp_outputs/generate_....png),
            file:// URI, http(s) URL, or MCP ImageContent dict {"data": "<base64>", "mimeType": "image/png"}.
            Same input flexibility as remove_background (supports MCP ImageContent/ContentBlock alternatives).
        tolerance: Max RGB Euclidean distance (0-441) for a pixel to match the corner color
            (default 30). Ignored when force_match is True.
        force_match: When True, auto-tune tolerance so all four corners match (binary search).
        feather: Gaussian blur radius for soft mask edges. 0 = no feathering.
        foreground: If True (default), return RGBA cutout with transparent background.
            If False, return background mask (white=background, black=foreground; invert for foreground matte).
        image_path: Alternative to image_base64 — explicit local file path or file:// URI.
        image_url: Alternative to image_base64 — http(s) URL to fetch.

    Returns:
        PNG image + local path (mcp_outputs/...) so it can be referenced in conversation.
    """
    raw_candidate = (
        image_path if image_path not in (None, "") else image_url if image_url not in (None, "") else image_base64
    )
    raw_input = _coerce_to_str_input(raw_candidate).strip()
    if not raw_input:
        raise ValueError(
            "image_base64 (or image_path/image_url) must be non-empty — expected base64 string, data URI, file path, file:// URI, http(s) URL, or MCP ImageContent dict {data, mimeType}"
        )
    if tolerance < 0 or tolerance > 442:
        raise ValueError("tolerance must be between 0 and 442")
    if feather < 0:
        raise ValueError("feather must be >= 0")

    from pixel_alchemy.background_removal.corner_flood import corner_flood

    with tempfile.TemporaryDirectory() as tmp:
        inp = _load_image_input(raw_input, tmp)
        out = Path(tmp) / ("foreground.png" if foreground else "mask.png")
        corner_flood(inp, out, tolerance=tolerance, force_match=force_match, feather=feather, foreground=foreground)
        data = out.read_bytes()
    prefix = "foreground_flood" if foreground else "mask_flood"
    persist = _save_persistent(data, prefix)
    rel = persist.relative_to(_REPO_ROOT) if persist.is_relative_to(_REPO_ROOT) else persist
    return [
        Image(data=data, format="png"),
        TextContent(
            type="text",
            text=f'Saved locally to {persist} (relative: {rel}) — use image_path="{persist}" for next step.',
        ),
    ]


@mcp.tool()
def upscale_image(
    image_base64: Annotated[
        str | dict,
        Field(
            description="Image input. Accepts: raw base64 string (no prefix, contentEncoding: base64, contentMediaType: image/png|image/jpeg|image/webp), data URI 'data:image/png;base64,...', local file path ('/tmp/out.png' or 'mcp_outputs/generate_....png'), file:// URI, http(s) URL, or MCP-standard ImageContent dict {'type':'image','data':'<base64>','mimeType':'image/png'} / BlobResourceContents {'blob':'<base64>','mimeType':'image/png'} / EmbeddedResource. Same as remove_background."
        ),
    ] = "",
    target_width: Annotated[
        int, Field(description="Exact final width in pixels (aspect preserved).", ge=1, le=10000)
    ] = 2048,
    blur_multipliers: Annotated[
        list[float] | None,
        Field(
            description="Per-pass blur radii multipliers (default [5, 3, 1]). Each pass does: upscayl → GaussianBlur(m * log(scale)) → Lanczos to target_width. Fewer values = fewer passes (e.g. [3, 1] = 2 passes)."
        ),
    ] = None,
    model: Annotated[
        str,
        Field(
            description="Upscayl model name. Common: 'high-fidelity-4x', 'upscayl-standard-4x', 'ultrasharp-4x', 'upscayl-lite-4x', 'digital-art-4x'."
        ),
    ] = "upscayl-standard-4x",
    scale: Annotated[int, Field(description="Upscayl scale factor.", ge=2, le=4)] = 4,
    image_path: Annotated[
        str | None, Field(description="Alternative to image_base64 — explicit local file path or file:// URI.")
    ] = None,
    image_url: Annotated[str | None, Field(description="Alternative to image_base64 — http(s) URL to fetch.")] = None,
) -> list[Image | TextContent]:  # type: ignore[valid-type]
    """Upscale an image via the upscayl_pipeline (multi-pass upscayl + blur + Lanczos downscale).

    Args:
        image_base64: Base64 string, data URI, local file path, file:// URI, http(s) URL, or MCP ImageContent dict
            (same as remove_background — file paths like image-1788902666771.png or mcp_outputs/... work, plus MCP ImageContent).
        target_width: Exact final width in pixels (aspect preserved, default 2048).
        blur_multipliers: Per-pass blur radii multipliers (default [5, 3, 1]).
            Each pass does: upscayl → GaussianBlur(m * log(scale)) → Lanczos to target_width.
            Fewer values = fewer passes (e.g. [3, 1] = 2 passes).
        model: Upscayl model name (default "upscayl-standard-4x").
            Common: "high-fidelity-4x", "ultrasharp-4x", "upscayl-lite-4x", "digital-art-4x".
        scale: Upscayl scale factor 2, 3, or 4 (default 4).
        image_path: Alternative to image_base64 — explicit local file path or file:// URI.
        image_url: Alternative to image_base64 — http(s) URL to fetch.

    Returns:
        PNG image + local path (mcp_outputs/...) so it can be referenced in conversation.
    """
    raw_candidate = (
        image_path if image_path not in (None, "") else image_url if image_url not in (None, "") else image_base64
    )
    raw_input = _coerce_to_str_input(raw_candidate).strip()
    if not raw_input:
        raise ValueError(
            "image_base64 (or image_path/image_url) must be non-empty — expected base64 string, data URI, file path, file:// URI, http(s) URL, or MCP ImageContent dict {data, mimeType}"
        )
    if target_width <= 0:
        raise ValueError("target_width must be > 0")
    if scale not in (2, 3, 4):
        raise ValueError("scale must be 2, 3, or 4")

    multipliers = blur_multipliers if blur_multipliers is not None else [5, 3, 1]

    from pixel_alchemy.pipeline.upscayl_pipeline import upscayl_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        inp = _load_image_input(raw_input, tmp)
        out = Path(tmp) / "upscaled.png"
        upscayl_pipeline(
            inp,
            out,
            target_width=target_width,
            blur_multipliers=multipliers,
            model=model,
            scale=scale,
        )
        data = out.read_bytes()
    persist = _save_persistent(data, "upscaled")
    rel = persist.relative_to(_REPO_ROOT) if persist.is_relative_to(_REPO_ROOT) else persist
    return [
        Image(data=data, format="png"),
        TextContent(
            type="text",
            text=f'Saved locally to {persist} (relative: {rel}) — use image_path="{persist}" for next step.',
        ),
    ]


# ---------------------------------------------------------------------------
# Image inventory — list / retrieve from mcp_outputs
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}


def _image_info(p: Path) -> dict:
    try:
        stat = p.stat()
        size = stat.st_size
        mtime = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))
    except Exception:
        size = 0
        mtime = ""
    rel = p.relative_to(_REPO_ROOT) if p.is_relative_to(_REPO_ROOT) else p
    info: dict = {
        "filename": p.name,
        "path": str(p),
        "relative_path": str(rel),
        "size_bytes": size,
        "modified": mtime,
    }
    # quick dimensions without heavy deps
    try:
        from PIL import Image as PILImage

        with PILImage.open(p) as im:
            info["width"] = im.width
            info["height"] = im.height
            info["mode"] = im.mode
            info["format"] = im.format
    except Exception:  # noqa: BLE001, S110
        pass
    return info


def _resolve_image_request(name: str) -> Path:
    s = name.strip()
    if not s:
        raise ValueError("image_path must be non-empty")
    if s.startswith("file://"):
        s = s[len("file://") :]
        if s.startswith("///"):
            s = s[2:]
    p = Path(s).expanduser()
    out_resolved = OUTPUT_DIR.resolve()

    # Absolute path — must be inside OUTPUT_DIR
    if p.is_absolute():
        try:
            resolved = p.resolve()
            if resolved.is_relative_to(out_resolved) and resolved.exists() and resolved.is_file():
                return resolved
        except Exception:
            pass
        raise FileNotFoundError(
            f"Image not found or outside mcp_outputs: {name} (allowed dir: {OUTPUT_DIR})"
        )

    # Try direct relative candidates inside OUTPUT_DIR
    candidates = [
        OUTPUT_DIR / s,
        OUTPUT_DIR / Path(s).name,
        _REPO_ROOT / s,
    ]
    for cand in candidates:
        try:
            if cand.exists() and cand.is_file() and cand.resolve().is_relative_to(out_resolved):
                return cand.resolve()
        except Exception:  # noqa: BLE001, S110
            continue

    # Fallback: search by exact filename inside OUTPUT_DIR
    target = Path(s).name
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.iterdir():
            if f.name == target and f.is_file():
                return f.resolve()

    raise FileNotFoundError(
        f"Image not found: {name} (searched in {OUTPUT_DIR}). Use list_images to see available files."
    )


@mcp.tool()
def list_images(
    prefix: Annotated[
        str | None,
        Field(description="Optional filename prefix filter, e.g. 'generate', 'foreground', 'upscaled'. Case-sensitive substring match."),
    ] = None,
    limit: Annotated[int, Field(description="Max number of images to return (newest first).", ge=1, le=500)] = 50,
    offset: Annotated[int, Field(description="Offset for pagination (newest first).", ge=0)] = 0,
) -> list[TextContent]:  # type: ignore[valid-type]
    """List available images in mcp_outputs.

    Args:
        prefix: Optional substring filter on filename (e.g. 'generate' or 'foreground').
        limit: Max number to return (default 50, newest first).
        offset: Pagination offset.

    Returns:
        JSON list of image metadata (filename, path, relative_path, size_bytes, modified, width/height) + human-readable summary.
    """
    _ensure_output_dir()
    if not OUTPUT_DIR.exists():
        return [TextContent(type="text", text=json.dumps([], indent=2))]

    files = [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    if prefix and prefix.strip():
        needle = prefix.strip()
        files = [p for p in files if needle in p.name]

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(files)
    sliced = files[offset : offset + limit]

    infos = [_image_info(p) for p in sliced]
    summary = f"Showing {len(infos)}/{total} images in {OUTPUT_DIR} (offset {offset}, limit {limit})"
    if prefix:
        summary += f" filtered by prefix '{prefix}'"
    body = summary + "\n" + json.dumps(infos, indent=2)
    # Also include relative paths hint
    if infos:
        body += "\n\nTo retrieve: call get_image with image_path set to the 'path' or 'filename' above."
    else:
        body += "\n\nNo images match. Generate one with generate_image first."
    return [TextContent(type="text", text=body)]


@mcp.tool()
def get_image(
    image_path: Annotated[
        str,
        Field(description="Path or filename of image to retrieve. Accepts bare filename (e.g. 'generate_...png'), relative path 'mcp_outputs/...png', absolute path inside mcp_outputs, or file:// URI. Use list_images to discover names.", min_length=1),
    ],
) -> list[Image | TextContent]:  # type: ignore[valid-type]
    """Retrieve an image from mcp_outputs by path or filename.

    Args:
        image_path: Filename or path inside mcp_outputs (e.g. 'generate_20260304_...png' or 'mcp_outputs/generate_...png' or absolute path). Use list_images first to discover available images.

    Returns:
        PNG/JPEG image + local path confirmation. Works for any image previously saved by generate_image, remove_background, upscale_image, etc.
    """
    p = _resolve_image_request(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path} -> {p}")
    if p.suffix.lower() not in _IMAGE_EXTS:
        raise ValueError(f"Not an image file: {p} (allowed: {sorted(_IMAGE_EXTS)})")
    data = p.read_bytes()
    # keep .jpg as jpeg for MCP
    suffix = p.suffix.lower()
    fmt = {"png": "png", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(suffix, "png")
    # Provide base64 via Image(data=...) — MCP will send as content block
    rel = p.relative_to(_REPO_ROOT) if p.is_relative_to(_REPO_ROOT) else p
    return [
        Image(data=data, format=fmt),
        TextContent(type="text", text=f"Retrieved {p.name} from {p} (relative: {rel}, {len(data)} bytes, {fmt})"),
    ]


# ---------------------------------------------------------------------------
# CLI / entrypoint
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pixel Alchemy MCP server (Streamable HTTP)")
    p.add_argument("--host", default=os.getenv("PIXEL_ALCHEMY_MCP_HOST", "0.0.0.0"), help="Bind host (default 0.0.0.0)")
    p.add_argument(
        "--port", type=int, default=int(os.getenv("PIXEL_ALCHEMY_MCP_PORT", "1255")), help="Bind port (default 1255)"
    )
    p.add_argument(
        "--path", default=os.getenv("PIXEL_ALCHEMY_MCP_PATH", "/mcp"), help="Streamable HTTP path (default /mcp)"
    )
    p.add_argument("--stateless", action="store_true", help="Enable stateless HTTP mode")
    p.add_argument(
        "--max-body-size",
        type=int,
        default=_DEFAULT_MAX_BODY_SIZE,
        help="Max HTTP request body size in bytes (default 250 MiB, ~262M; use 0 for uvicorn default; must be >100MB for large b64 payloads, env PIXEL_ALCHEMY_MCP_MAX_BODY_SIZE)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # mcp_outputs is intentionally preserved after shutdown (no cleanup).
    # max_request_body_size must be >100MB to accept large b64 payloads (default 250 MiB).
    mcp.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        streamable_http_path=args.path,
        stateless_http=args.stateless,
        max_request_body_size=args.max_body_size,
    )


if __name__ == "__main__":
    main()
