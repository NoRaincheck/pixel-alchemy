# Corner Flood + BiRefNet Foreground Protection

Combines the fast corner-sampling flood fill with BiRefNet foreground detection to remove backgrounds without eating into subject matter.

See `corner_flood_birefnet_example.py` for runnable code.

---

## Problem

The corner flood fill is fast and simple: sample the four corners, find the dominant background color, flood fill outward. But it has no concept of "what is the subject." If the subject's edge colors are close to the background, the flood fill leaks in and removes parts of the subject.

BiRefNet identifies foreground accurately but is slower (ONNX inference at 1024×1024). It doesn't know about corner connectivity, so it can't do flood fill.

## Solution

Use both:

1. **BiRefNet** → boolean foreground mask (what is the subject?)
2. **Corner flood fill** → boolean background mask (what is connected to the corners?)
3. **Combine** → a pixel is only removed if it is both background-connected AND not flagged as foreground

The flood fill tolerance is not fixed. It is **binary searched per image** for the highest value that keeps foreground loss at or below a configurable threshold (default 10%).

---

## How It Works

### Step 1: Pre-compute the distance map

Sample the four corners, pick the most common color (majority vote), then compute the Euclidean distance from every pixel to that color:

```python
diff = np.sqrt((r - bg_color[0]) ** 2 + (g - bg_color[1]) ** 2 + (b - bg_color[2]) ** 2)
```

This is the expensive part (per-pixel sqrt), so it is done **once** per image.

### Step 2: Binary search for optimal tolerance

For a given tolerance `t`, threshold the distance map (`diff <= t`) and flood fill from the corners. The result is a boolean background mask. Check how much of the BiRefNet foreground it overlaps:

```python
loss = (flood_bg & fg_mask).sum() / fg_mask.sum()
```

Binary search tolerance from 1 to 50 for the highest value where `loss ≤ 0.10`. At most ~6 flood fill calls (log₂50).

### Step 3: Apply the final mask

```python
alpha = np.where(is_bg, 0, 255)  # 0 = transparent, 255 = opaque
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 128 | BiRefNet mask threshold (0–255). Higher = less foreground detected. |
| `max_fg_loss` | 0.10 | Maximum fraction of BiRefNet foreground that the flood fill may remove. |
| `tolerance range` | 1–50 | Euclidean distance in RGB space. 50 covers a wide color range. |

### Tuning tips

- **`max_fg_loss=0.05`** — stricter, preserves more foreground but may leave more background noise near edges.
- **`max_fg_loss=0.20`** — more aggressive, removes more background but risks eating into subject.
- **`threshold=160`** — if BiRefNet over-segments (flags too much as foreground), raise this.
- **`threshold=100`** — if BiRefNet under-segments (misses parts of the subject), lower this.

---

## Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| `fg_mask=None` | Falls back to fixed tolerance 3 (original behaviour) |
| No foreground detected (`fg_mask.sum() == 0`) | Uses max tolerance 50 |
| Even tolerance 1 exceeds `max_fg_loss` | Uses tolerance 1 (most conservative) |

---

## When to Use This

- Images with solid/uniform backgrounds (white, black, single color)
- Subjects that have edge colors similar to the background
- Batch processing where per-image manual tuning isn't practical
- When you need the speed of flood fill but the accuracy of AI segmentation

## When to Use BiRefNet Alone

- Complex or textured backgrounds (not uniform)
- When you want a precise alpha matte, not just transparent/opaque
- When edge quality matters more than speed

---

## Performance

- BiRefNet inference: ~1–2s per image (ONNX, CPU)
- Distance map computation: ~10ms (numpy vectorised)
- Flood fill per tolerance level: ~1ms (OpenCV C implementation)
- Binary search: at most 6 flood fill calls
- **Total overhead of dynamic tolerance vs fixed: ~6ms** (negligible)
