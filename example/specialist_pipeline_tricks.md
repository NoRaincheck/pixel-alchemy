# Specialist Pipeline Tricks

A cookbook of reusable image-processing patterns for batch workflows.

See `specialist_pipeline_tricks.py` for runnable code examples of each trick.

---

## Pipeline Patterns

### Pattern 1: Large JPEGs → High-res output

For source images with heavy compression artifacts (e.g. 20000px+ JPEGs):

```
Original (large)
  → Downscale to 1920px (Lanczos)     ← destroys JPEG 8×8 blocks
  → Gaussian blur radius=1.5           ← smooths noise before AI
  → upscayl digital-art-4x scale=4    → 7680px (AI detail at high res)
  → Downscale to 3840px (Lanczos)      ← detail "bakes in" naturally
  → Double unsharp mask                ← broad + fine sharpening
Output: 3840px (4K)
```

### Pattern 2: Mid-size images → Upscaled output

For source images that are already decent quality (500–2000px wide):

```
Original
  → upscayl high-fidelity-4x scale=2-4  ← faithful structure
  → upscayl ultrasharp-4x scale=2       ← sharpening pass
  → Lanczos resize to target
Output: target resolution
```

---

## Key Tricks

### 1. Pre-downscale kills JPEG artifacts

JPEG creates 8×8 block boundaries and ringing noise. AI upscalers amplify these. Downsampling to ~1920px first with Lanczos smears the blocks into smooth gradients, giving the AI a clean canvas.

### 2. Blur before upscale prevents hallucination

AI upscalers can mistake noise for detail. A light Gaussian blur (radius 1.0–2.0) removes high-frequency grain while preserving structure. The AI then generates clean detail instead of amplifying noise.

### 3. Upscale high → downscale for better detail

If you need 4K output, upscale to 8K first then downscale. The AI produces its best detail at the higher resolution. The Lanczos downscaler subsamples that detail, producing a sharper 4K than a direct 4K upscale.

### 4. Two-model sharpening (high-fidelity → ultrasharp)

Different models have different strengths. `high-fidelity-4x` preserves the original look faithfully. `ultrasharp-4x` enhances edges. Running both in sequence gives you faithful structure + crisp detail.

### 5. Adaptive scale selection

Don't always use 4x. If the input is already 1000px and you need 2000px, only 2x is needed. Choosing the smallest adequate scale saves VRAM and time.

```python
scale = min(max(round(target_width / input_width), 2), 4)
```

### 6. Double-pass sharpening

One sharpening pass handles one scale of detail. Two passes — broad (radius 0.8) then fine (radius 0.4) — sharpen both edges and texture.

### 7. Inpainting prompt recipe

For fixing missing/wrong parts, the prompt must describe:
1. **What** should be in the region
2. **Style** matching the surrounding image
3. **Context** details (colors, textures, etc.)

Example: `"a subject sitting, with details visible, soft watercolor style, consistent with the rest of the illustration, accessories"`

### 8. Inpainting parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `cfg_scale` | 0.5–1.0 | Higher values cause artifacts with sd-cli |
| `steps` | 9 | Sweet spot for inpainting quality |
| `model` | flux2-dev-Q4_K_S | Good balance of quality and speed |

### 9. Multi-pass inpainting for shading

When inpainting changes the lighting/brightness of a region:
- Pass 1: fill the region (broad mask)
- Pass 2: fix shading on a narrower sub-region

This prevents the second pass from undoing the first.

### 10. Batch skip logic

Always check if output already exists before processing:

```python
output = input.with_name(input.stem + "_enhanced.jpg")
if output.exists():
    continue
```

### 11. Async with semaphore

Bound concurrency to prevent OOM:

```python
sem = asyncio.Semaphore(3)  # max 3 images at once
async with sem:
    await process_image(img)
```

### 12. Pipeline reports

Track before/after stats for auditing:

```json
{
  "file": "image.png",
  "original": "500x750",
  "original_size_kb": 120,
  "final": "2000x3000",
  "final_size_kb": 850,
  "width_scale": 4.0
}
```

### 13. Resize by shorter side

When you care about the smaller dimension (thumbnails, mobile, or pre-downscale before AI), resize so the shorter side hits a target:

```python
img = resize_lanczos_to_shorter(img, shorter_target=480)
# w, h scales proportionally so min(w, h) == 480
```

### 14. Compute target from min W and H constraints

When you need output that satisfies both minimum width AND minimum height while preserving aspect ratio:

```python
target_w, target_h = compute_target(orig_w, orig_h, min_width=6600, min_height=3300)
# Scales up just enough to meet both constraints
```

For a 22936×12800 image: `compute_target(22936, 12800, 6600, 3300)` → `(6600, 3683)`

### 15. Full 5-pass pipeline for large JPEGs

Ready-to-use pipeline for very large JPEGs with compression artifacts:

```python
from example.specialist_pipeline_tricks import large_jpeg_pipeline

large_jpeg_pipeline(
    input_path=Path("big-photo.jpg"),
    output_path=Path("big-photo_enhanced.jpg"),
    min_width=6600,
    min_height=3300,
)
```

Pipeline:
```
Original (22936×12800 JPEG)
  → Pass 1: Downscale shorter side to ceil(target/4) via Lanczos
  → Pass 2: Gaussian blur radius=1.5
  → Pass 3: upscayl digital-art-4x scale=4
  → Pass 4: Downscale to exact target (6600×3683) via Lanczos
  → Pass 5: Double unsharp mask (0.8, 0.4)
Output: 6600×3683 JPEG (95% quality)
```

---

## Model Selection Cheat Sheet

| Use case | Model | Scale | Notes |
|----------|-------|-------|-------|
| Large JPEGs with artifacts | digital-art-4x | 4 | Best for illustrated/digital art |
| Faithful upscale | high-fidelity-4x | 2–4 | Preserves original look |
| Sharpening pass | ultrasharp-4x | 2 | Enhances edges and detail |
| General purpose | upscayl-standard-4x | 4 | Default balanced model |
| Inpainting | flux2-dev | — | Use with sd-cli |
