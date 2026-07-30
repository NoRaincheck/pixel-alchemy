# BiRefNet + Vtracer Iterative Expansion

Uses BiRefNet for foreground detection, then expands the mask outward via iterative dilation and vtracer spline smoothing until it reaches the image edge. Outputs an RGBA image with transparent background.

See `remove_bg_loose_example.py` for runnable code.

---

## Problem

BiRefNet produces a tight foreground mask that clips right at the subject boundary. For illustration-style images (e.g. baby/animal PNGs from NotebookLM), this can look harsh — the mask misses the natural "shape" of the immediate background elements around the subject.

A fixed dilation would expand uniformly in all directions, producing an unnatural bloated boundary. What's needed is an expansion that follows the natural contours of the subject.

## Solution

Iterate three steps:

1. **Dilate** the binary mask with an elliptical kernel (20px default) — pushes the boundary outward uniformly
2. **Vectorize** with vtracer (spline mode) — converts the pixelated dilated boundary into smooth curves
3. **Render back** to raster via ImageMagick — the spline approximation naturally smooths the boundary

Repeat until the foreground touches within 1px of any image edge. The vtracer smoothing at each step ensures the expanded boundary follows natural contours rather than pixel artifacts.

---

## How It Works

### Step 1: BiRefNet tight mask

Run BiRefNet to get a grayscale mask, threshold at 128 to get a binary foreground:

```python
tight_mask = get_birefnet_mask(img)
binary = (tight_mask > 128).astype(np.uint8) * 255
```

### Step 2: Downsample to working resolution

Resize to width 500 (keeping aspect ratio) for fast vtracer processing:

```python
scale = target_width / w
small_h = int(h * scale)
small = cv2.resize(binary, (target_width, small_h), interpolation=cv2.INTER_NEAREST)
```

### Step 3: Iterative expansion loop

Each iteration:
1. **Dilate** with a 20px elliptical kernel
2. **Vectorize** with vtracer (default settings, spline mode)
3. **Render** SVG back to PNG via ImageMagick
4. **Check** if mask touches the edge (within 1px of any border)

```python
for i in range(max_iters):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_step, dilation_step))
    small = cv2.dilate(small, kernel, iterations=1)
    small = _vtracer_smooth(small, tmp_dir)
    if _mask_touches_edge(small > 0):
        break
```

### Step 4: Upscale and apply

Resize the final mask back to original resolution, apply as alpha channel:

```python
final = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST) > 128
img_rgba.putalpha(Image.fromarray((final * 255).astype(np.uint8), mode="L"))
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_width` | 500 | Working width for vtracer (smaller = faster, less precise) |
| `dilation_step` | 20 | Pixels of dilation per iteration (larger = fewer iterations, coarser expansion) |
| `max_iters` | 50 | Maximum dilation iterations before stopping |

### Tuning tips

- **`target_width=300`** — faster processing, slightly less precise boundaries
- **`target_width=800`** — slower, more accurate boundary tracing
- **`dilation_step=10`** — finer expansion, more iterations needed
- **`dilation_step=40`** — coarser expansion, reaches edge faster

---

## Dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| `vtracer` (Python) | Vectorize binary masks to SVG | `pip install vtracer` |
| `magick` (ImageMagick) | Render SVG back to PNG | `brew install imagemagick` |
| `birefnet` (pixel_alchemy) | AI foreground detection | bundled in this project |

---

## Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| Subject already touches edge | Iteration 0 completes immediately |
| Very large images | Downsampled to `target_width` for speed, upscaled at the end |
| No foreground detected | BiRefNet mask is empty, loop runs but expansion is from nothing |

---

## When to Use This

- Illustration-style images with simple backgrounds
- When BiRefNet's tight mask looks too harsh at boundaries
- When you want a "looser" mask that captures the natural shape around the subject
- Batch processing where manual mask painting isn't practical

## When to Use BiRefNet Alone

- When you need a precise alpha matte (e.g. hair/fur detail)
- Complex backgrounds where flood fill or dilation would include unwanted areas
- When edge accuracy matters more than natural-looking expansion

---

## Performance

- BiRefNet inference: ~1–2s per image (ONNX, CPU)
- vtracer vectorization: ~0.1s per iteration (at 500px width)
- ImageMagick render: ~0.05s per iteration
- Typical images reach edge in 3–5 iterations
- **Total: ~2–4s per image** (dominated by BiRefNet)
