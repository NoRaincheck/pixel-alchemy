"""Stylize close to a reference, then palette-align, with blend control.

Recipe:
  1. Stylize content toward a style image (AdaAttN torch, ONNX via cv2.dnn,
     or Reinhard Lab fallback).
  2. Blend stylized result with original at `alpha` (0 = original, 1 = full style).
  3. Map blended image toward a target palette (OKLab/Lab, Catppuccin etc).

Usage:
  from pixel_alchemy.pipeline.stylize_palette import stylize_palette
  stylize_palette("wallpaper/wallpaper_enhanced.jpg", "wallpaper/fox.png",
                  palette="catppuccin-mocha", alpha=0.7)

CLI:
  uv run python -m pixel_alchemy.pipeline.stylize_palette \\
      wallpaper/wallpaper_enhanced.jpg wallpaper/fox.png \\
      -o out.jpg --alpha 0.7 --palette catppuccin-mocha --strength 0.28
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# reuse palette helpers without circular import: try wallpaper.palette_map first,
# fall back to inline minimal set.
try:
    from wallpaper.palette_map import PALETTES, map_image  # type: ignore[import-not-found]
except ImportError:
    # fallback when wallpaper not on path (e.g. installed package)
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from wallpaper.palette_map import PALETTES, map_image  # type: ignore[no-redef]

from pixel_alchemy.style_transfer.adaattn import AdaAttNModel


def _reinhard_bgr(content_bgr: np.ndarray, style_bgr: np.ndarray) -> np.ndarray:
    c_lab = cv2.cvtColor(content_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    s_lab = cv2.cvtColor(style_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    cm, cs = c_lab.mean(axis=(0, 1)), c_lab.std(axis=(0, 1))
    sm, ss = s_lab.mean(axis=(0, 1)), s_lab.std(axis=(0, 1))
    cs = np.where(cs < 1e-6, 1, cs)
    out = (c_lab - cm) / cs * ss + sm
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


def _stylize_bgr(
    content_bgr: np.ndarray,
    style_bgr: np.ndarray,
    size: int | None = 512,
    device: str | None = None,
    onnx_path: Path | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ONNX via cv2.dnn if provided (not used here - caller handles)
    # Try torch AdaAttN
    try:
        from pathlib import Path as _P

        vgg = _P(__file__).parents[3] / "models" / "AdaAttN_model" / "vgg_normalised.pth"
        ckpt = _P(__file__).parents[3] / "models" / "AdaAttN_model" / "AdaAttN"
        if onnx_path and Path(onnx_path).exists():
            raise RuntimeError("onnx requested")
        if vgg.exists() and ckpt.exists():
            model = AdaAttNModel()
            model.load_vgg(vgg, device)
            model.load_pretrained(ckpt, device)
            from pixel_alchemy.style_transfer.adaattn import stylize

            return stylize(content_bgr, style_bgr, model, device=device, size=size, alpha=1.0, seed=seed)
    except Exception:
        pass
    # fallback: Reinhard
    # resize style to content size for Lab stats
    h, w = content_bgr.shape[:2]
    style_rs = cv2.resize(style_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return _reinhard_bgr(content_bgr, style_rs)


def _dnn_onnx_bgr(content_bgr: np.ndarray, onnx_path: Path, alpha: float = 1.0) -> np.ndarray:
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    h, w = content_bgr.shape[:2]
    # AdaAttN onnx expects content+style inputs; we use cv2.dnn only for
    # single-input fast-style models here. For AdaAttN dual-input, use torch.
    # So this path is for single-input fast-style onnx.
    max_side = 1280
    scale = min(1.0, max_side / max(h, w))
    iw, ih = int(w * scale), int(h * scale)
    small = cv2.resize(content_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(small, 1.0, (iw, ih), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    if out.ndim == 4:
        out = out[0].transpose(1, 2, 0) if out.shape[1] == 3 else out[0]
    out = np.clip(out, 0, 255).astype(np.uint8)
    if out.shape[0] != ih or out.shape[1] != iw:
        out = cv2.resize(out, (iw, ih), interpolation=cv2.INTER_LINEAR)
    stylized = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    if alpha < 1.0:
        stylized = cv2.addWeighted(stylized, alpha, content_bgr, 1 - alpha, 0)
    return stylized


def _blend_bgr(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a (stylized) over b (original) at alpha."""
    if alpha >= 1.0:
        return a
    if alpha <= 0.0:
        return b
    return cv2.addWeighted(a, alpha, b, 1 - alpha, 0)


def stylize_palette(
    content: str | Path | Image.Image | np.ndarray,
    style: str | Path | Image.Image | np.ndarray | None = None,
    *,
    palette: str = "catppuccin-mocha",
    palette_hex: list[str] | None = None,
    alpha: float = 0.75,
    strength: float | None = None,
    k: int | None = None,
    preserve_luma: float | None = None,
    colorspace: str = "oklab",
    size: int | None = 512,
    device: str | None = None,
    onnx_path: str | Path | None = None,
    seed: int | None = None,
    output: str | Path | None = None,
    quality: int = 95,
) -> Image.Image:
    """Stylize then palette-map with blend control.

    Args:
        content: content image path / PIL / BGR ndarray.
        style: style reference path / PIL / BGR. If None, skips stylize.
        palette: palette name in PALETTES or JSON path or comma-hex list.
        palette_hex: explicit hex list (overrides palette name).
        alpha: stylized blend 0..1 (0 = original, 1 = full stylized).
        strength/k/preserve_luma: palette-map params (defaults per palette).
        colorspace: "oklab" or "lab".
        size: AdaAttN inference size (512, 1024, or None for full-res).
        device: torch device (auto if None).
        onnx_path: fast-style ONNX for cv2.dnn single-input mode.
        seed: deterministic seed for AdaAttN sampling.
        output: if set, save to path.
        quality: JPEG quality.

    Returns:
        PIL RGB image (also saved if output given).
    """
    # -- load content as BGR + PIL --
    if isinstance(content, np.ndarray):
        content_bgr = content
        content_pil = Image.fromarray(cv2.cvtColor(content_bgr, cv2.COLOR_BGR2RGB))
    elif isinstance(content, Image.Image):
        content_pil = content.convert("RGB")
        content_bgr = cv2.cvtColor(np.array(content_pil), cv2.COLOR_RGB2BGR)
    else:
        p = Path(content)
        if not p.exists():
            raise FileNotFoundError(p)
        content_pil = Image.open(p).convert("RGB")
        content_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if content_bgr is None:
            raise FileNotFoundError(p)

    # -- stylize --
    if style is not None:
        if isinstance(style, np.ndarray):
            style_bgr = style
        elif isinstance(style, Image.Image):
            style_bgr = cv2.cvtColor(np.array(style.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            sp = Path(style)
            if not sp.exists():
                raise FileNotFoundError(sp)
            style_bgr = cv2.imread(str(sp), cv2.IMREAD_COLOR)
            if style_bgr is None:
                raise FileNotFoundError(sp)

        if onnx_path and Path(onnx_path).exists():
            stylized_bgr = _dnn_onnx_bgr(content_bgr, Path(onnx_path), alpha=1.0)
        else:
            stylized_bgr = _stylize_bgr(content_bgr, style_bgr, size=size, device=device, seed=seed)
        blended_bgr = _blend_bgr(stylized_bgr, content_bgr, alpha)
        blended_pil = Image.fromarray(cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB))
    else:
        blended_pil = content_pil

    # -- palette-map --
    if palette_hex is not None:
        hexs = palette_hex
    elif palette in PALETTES:
        hexs = PALETTES[palette]
    else:
        # try resolve via palette_map helper
        from wallpaper.palette_map import resolve_palette  # type: ignore

        _, hexs = resolve_palette(palette)

    # defaults per palette
    from wallpaper.palette_map import DEFAULTS  # type: ignore

    d = DEFAULTS.get(palette, {})
    s = strength if strength is not None else d.get("strength", 0.28)
    kk = k if k is not None else d.get("k", 3)
    pl = preserve_luma if preserve_luma is not None else d.get("preserve_luma", 0.12)

    out = map_image(blended_pil, hexs, strength=s, k=kk, preserve_luma=pl, colorspace=colorspace)

    if output is not None:
        o = Path(output)
        o.parent.mkdir(parents=True, exist_ok=True)
        kw: dict = {"quality": quality}
        if o.suffix.lower() in (".jpg", ".jpeg"):
            kw["subsampling"] = 0
            kw["optimize"] = True
        out.save(o, **kw)

    return out


def _parse_args():
    ap = argparse.ArgumentParser(description="Stylize → blend → palette-map recipe")
    ap.add_argument("content", type=Path, help="Content image")
    ap.add_argument("style", type=Path, nargs="?", default=None, help="Style reference image (optional)")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--palette", default="catppuccin-mocha", help="Palette name or JSON path or comma hex list")
    ap.add_argument("--alpha", type=float, default=0.75, help="Stylized blend 0..1 (0=original,1=full style)")
    ap.add_argument("--strength", type=float, default=None, help="Palette-map strength 0..1")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--preserve-luma", type=float, default=None)
    ap.add_argument("--colorspace", choices=["oklab", "lab"], default="oklab")
    ap.add_argument("--size", type=int, default=512, help="AdaAttN size (512/1024) or 0 for full-res")
    ap.add_argument("--device", default=None)
    ap.add_argument("--onnx", type=Path, default=None, help="Single-input fast-style ONNX (cv2.dnn)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quality", type=int, default=95)
    return ap.parse_args()


def main():
    args = _parse_args()
    size = None if args.size == 0 else args.size
    out = args.output
    if out is None:
        tag = f"stylized_a{args.alpha:g}_{args.palette}_{args.colorspace}"
        if args.strength is not None:
            tag += f"_s{args.strength:g}"
        out = args.content.with_name(f"{args.content.stem}_{tag}{args.content.suffix}")
    stylize_palette(
        args.content,
        args.style,
        palette=args.palette,
        alpha=args.alpha,
        strength=args.strength,
        k=args.k,
        preserve_luma=args.preserve_luma,
        colorspace=args.colorspace,
        size=size,
        device=args.device,
        onnx_path=args.onnx,
        seed=args.seed,
        output=out,
        quality=args.quality,
    )
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
