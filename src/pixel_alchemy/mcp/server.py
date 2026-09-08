"""MCP server exposing pixel-alchemy image tools over Streamable HTTP.

Tools:
- generate_image  — sd_cli generate with Z-image defaults (image generation only)
- remove_background — BiRefNet background removal
- upscale_image   — upscayl_pipeline (multi-pass upscayl + blur + Lanczos downscale)

All tools accept/return base64-encoded images via MCP ImageContent, so they are
reachable from remote machines via any MCP client over Streamable HTTP
(host 0.0.0.0 by default).

Run:
    python -m pixel_alchemy.mcp.server --host 0.0.0.0 --port 1255
    uv run pixel-alchemy-mcp --host 0.0.0.0 --port 1255
"""

from __future__ import annotations

import argparse
import atexit
import base64
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import TextContent

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
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


atexit.register(_cleanup_outputs)


@asynccontextmanager
async def _lifespan(server: MCPServer):  # type: ignore[no-untyped-def]
    _ensure_output_dir()
    try:
        yield
    finally:
        _cleanup_outputs()


mcp = MCPServer(
    name="pixel-alchemy",
    instructions=(
        "Pixel Alchemy image tools: generate images with Z-image (sd-cli), "
        "remove backgrounds with BiRefNet, and upscale via upscayl-pipeline."
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
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e
    if not data:
        raise ValueError("Decoded image is empty")
    suffix = _mime_to_suffix(mime)
    # sniff PNG/JPEG magic if no mime
    if mime is None:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            suffix = ".png"
        elif data[:2] == b"\xff\xd8":
            suffix = ".jpg"
    path = Path(tmp_dir) / f"input{suffix}"
    path.write_bytes(data)
    return path


def _load_image_input(image_input: str, tmp_dir: str) -> Path:
    """Accept base64, data URI, local file path, or http(s) URL and materialise to tmp_dir/input.*."""
    s = image_input.strip()
    if not s:
        raise ValueError("image must be non-empty")

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
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampling_method: str = "euler",
    diffusion_model: str | None = None,
    vae: str | None = None,
    llm: str | None = None,
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
            text=f"Saved locally to {persist} (relative: {rel}) — for follow-up use image_path=\"{persist}\" or image_base64 with this file.",
        ),
    ]


@mcp.tool()
def remove_background(
    image_base64: str = "",
    foreground: bool = True,
    fp16: bool = True,
    image_path: str | None = None,
    image_url: str | None = None,
) -> list[Image | TextContent]:  # type: ignore[valid-type]
    """Remove background from an image using BiRefNet.

    Args:
        image_base64: Base64 string, data URI (data:image/png;base64,...),
            local file path (e.g. image-1788902666771.png or /tmp/out.png or mcp_outputs/generate_....png),
            or http(s) URL. LM Studio / chat-generated files can be passed
            as their file path directly — no manual base64 needed.
        foreground: If True (default), return RGBA cutout with transparent background.
            If False, return grayscale mask (white=foreground).
        fp16: Use FP16 ONNX model (smaller/faster, default True). False for FP32.
        image_path: Alternative to image_base64 — explicit local file path.
        image_url: Alternative to image_base64 — http(s) URL to fetch.

    Returns:
        PNG image + local path (mcp_outputs/...) so it can be referenced in conversation.
    """
    raw_input = (image_path or image_url or image_base64 or "").strip()
    if not raw_input:
        raise ValueError("image_base64 (or image_path/image_url) must be non-empty")

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
        TextContent(type="text", text=f"Saved locally to {persist} (relative: {rel}) — use image_path=\"{persist}\" for next step."),
    ]


@mcp.tool()
def upscale_image(
    image_base64: str = "",
    target_width: int = 2048,
    blur_multipliers: list[float] | None = None,
    model: str = "upscayl-standard-4x",
    scale: int = 4,
    image_path: str | None = None,
    image_url: str | None = None,
) -> list[Image | TextContent]:  # type: ignore[valid-type]
    """Upscale an image via the upscayl_pipeline (multi-pass upscayl + blur + Lanczos downscale).

    Args:
        image_base64: Base64 string, data URI, local file path, or http(s) URL
            (same as remove_background — file paths like image-1788902666771.png or mcp_outputs/... work).
        target_width: Exact final width in pixels (aspect preserved, default 2048).
        blur_multipliers: Per-pass blur radii multipliers (default [5, 3, 1]).
            Each pass does: upscayl → GaussianBlur(m * log(scale)) → Lanczos to target_width.
            Fewer values = fewer passes (e.g. [3, 1] = 2 passes).
        model: Upscayl model name (default "upscayl-standard-4x").
            Common: "high-fidelity-4x", "ultrasharp-4x", "upscayl-lite-4x", "digital-art-4x".
        scale: Upscayl scale factor 2, 3, or 4 (default 4).
        image_path: Alternative to image_base64 — explicit local file path.
        image_url: Alternative to image_base64 — http(s) URL to fetch.

    Returns:
        PNG image + local path (mcp_outputs/...) so it can be referenced in conversation.
    """
    raw_input = (image_path or image_url or image_base64 or "").strip()
    if not raw_input:
        raise ValueError("image_base64 (or image_path/image_url) must be non-empty")
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
        TextContent(type="text", text=f"Saved locally to {persist} (relative: {rel}) — use image_path=\"{persist}\" for next step."),
    ]


# ---------------------------------------------------------------------------
# CLI / entrypoint
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pixel Alchemy MCP server (Streamable HTTP)")
    p.add_argument("--host", default=os.getenv("PIXEL_ALCHEMY_MCP_HOST", "0.0.0.0"), help="Bind host (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=int(os.getenv("PIXEL_ALCHEMY_MCP_PORT", "1255")), help="Bind port (default 1255)")
    p.add_argument("--path", default=os.getenv("PIXEL_ALCHEMY_MCP_PATH", "/mcp"), help="Streamable HTTP path (default /mcp)")
    p.add_argument("--stateless", action="store_true", help="Enable stateless HTTP mode")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        # MCPServer.run handles anyio + uvicorn internally; lifespan cleans OUTPUT_DIR,
        # but also clean here for abrupt SIGINT/CancelledError where lifespan may abort.
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            streamable_http_path=args.path,
            stateless_http=args.stateless,
        )
    finally:
        _cleanup_outputs()


if __name__ == "__main__":
    main()
