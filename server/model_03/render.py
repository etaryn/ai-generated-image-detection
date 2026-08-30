"""The four user-facing outputs: overlay, outlines, score, explanation.

Rendered with PIL and numpy alone -- no matplotlib. The renderer runs on every
upload in the Streamlit client, and pulling a plotting stack (and its global
figure state) into a per-request path is a cost with no return when the whole
job is "tint an array and draw some rectangles".

Two deliberate choices about how the map is drawn:

* **The uncertain band is rendered as uncertain**, not as a mid-range colour on
  the same ramp as everything else. Confident-low, uncertain and confident-high
  are three different statements, and a continuous blue-to-red ramp silently
  turns the middle one into "a bit suspicious", which is exactly the
  over-reading this pipeline is built to avoid. Below the low threshold the
  overlay is transparent, in the uncertain band it is a desaturated grey-blue,
  and only above the high threshold does it go warm.
* **Alpha follows confidence**, so a weakly-supported area is visibly faint
  rather than a solid patch of colour that reads as certainty.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from mapper.heatmap import AILikelihoodMap

# Anchors for the "confident AI" part of the ramp: amber -> orange -> red.
_WARM = np.array(
    [
        [255, 214, 102],
        [255, 149, 5],
        [214, 45, 32],
    ],
    dtype=np.float64,
)
_UNCERTAIN = np.array([120, 132, 148], dtype=np.float64)


def colorize(heat: np.ndarray, thresholds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Map a heatmap to (rgb uint8 (H, W, 3), alpha float (H, W) in [0, 1]).

    Alpha is 0 below the low threshold, a flat low value through the uncertain
    band, and rises with the score above the high threshold.
    """
    lo, hi = thresholds
    h = np.nan_to_num(np.asarray(heat, dtype=np.float64), nan=lo)

    rgb = np.zeros(h.shape + (3,), dtype=np.float64)
    alpha = np.zeros(h.shape, dtype=np.float64)

    uncertain = (h > lo) & (h < hi)
    rgb[uncertain] = _UNCERTAIN
    # Fade in across the band so the boundary at `lo` is not a hard edge.
    alpha[uncertain] = 0.10 + 0.20 * ((h[uncertain] - lo) / max(hi - lo, 1e-6))

    warm = h >= hi
    if warm.any():
        t = (h[warm] - hi) / max(1.0 - hi, 1e-6)
        idx = np.clip(t * (len(_WARM) - 1), 0, len(_WARM) - 1)
        low_i = np.floor(idx).astype(int)
        high_i = np.clip(low_i + 1, 0, len(_WARM) - 1)
        frac = (idx - low_i)[:, None]
        rgb[warm] = _WARM[low_i] * (1 - frac) + _WARM[high_i] * frac
        alpha[warm] = 0.40 + 0.35 * t

    return rgb.astype(np.uint8), alpha


def overlay_heatmap(
    image: Image.Image,
    amap: AILikelihoodMap,
    max_alpha: float = 0.75,
) -> Image.Image:
    """Tint `image` by the likelihood map. Returns a new RGB image."""
    base = np.asarray(image.convert("RGB"), dtype=np.float64)
    heat = amap.heat
    if heat.shape != base.shape[:2]:
        heat = np.asarray(
            Image.fromarray(np.nan_to_num(heat, nan=amap.thresholds[0]).astype(np.float32), mode="F")
            .resize(image.size, Image.BILINEAR),
            dtype=np.float64,
        )

    rgb, alpha = colorize(heat, amap.thresholds)
    # Confidence-following alpha: thin window support renders faint.
    support = amap.support
    if support.shape != base.shape[:2]:
        support = np.asarray(
            Image.fromarray(support.astype(np.float32), mode="F").resize(image.size, Image.BILINEAR),
            dtype=np.float64,
        )
    alpha = np.clip(alpha * (0.5 + 0.5 * support), 0.0, max_alpha)[..., None]

    blended = base * (1.0 - alpha) + rgb.astype(np.float64) * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


_LABEL_COLORS = {
    "generated_content": (214, 45, 32),
    "generative_edit": (255, 149, 5),
    "face_region_edit_evidence": (188, 80, 220),
    "conventional_manipulation": (48, 140, 214),
    "uncharacterised_suspicion": (120, 132, 148),
}


def draw_regions(image: Image.Image, findings, width: int = 3) -> Image.Image:
    """Outline each finding and caption it with its label, score and specialist."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    for finding in findings:
        x0, y0, x1, y1 = finding.region.bbox
        color = _LABEL_COLORS.get(finding.label, (255, 149, 5))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)

        caption = (
            f"#{finding.region.region_id} {finding.label} "
            f"{finding.probability:.2f} ({finding.route.primary})"
        )
        text_box = draw.textbbox((0, 0), caption)
        tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
        ty = max(0, y0 - th - 4)
        draw.rectangle([x0, ty, x0 + tw + 6, ty + th + 4], fill=color)
        draw.text((x0 + 3, ty + 2), caption, fill=(255, 255, 255))

    return out


def render_panel(image: Image.Image, amap: AILikelihoodMap, findings) -> Image.Image:
    """Original | heatmap overlay | outlined regions, side by side."""
    base = amap.working_image.convert("RGB")
    panels = [base, overlay_heatmap(base, amap), draw_regions(overlay_heatmap(base, amap), findings)]

    width = sum(p.width for p in panels) + 2 * 8
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, height), (18, 20, 24))

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + 8
    return canvas


def heatmap_image(amap: AILikelihoodMap) -> Image.Image:
    """The bare heatmap, for saving or for a side panel in the UI."""
    rgb, alpha = colorize(amap.heat, amap.thresholds)
    dark = np.zeros_like(rgb, dtype=np.float64)
    blended = dark * (1.0 - alpha[..., None]) + rgb.astype(np.float64) * alpha[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
