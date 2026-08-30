"""Edge-preserving smoothing for the heatmap.

Patch scores are noisy: neighbouring windows see almost the same pixels and
still disagree by a few points, and that jitter turns into speckle regions in
Layer 2. Smoothing fixes it -- but a plain Gaussian blur also smears the one
thing the pipeline is trying to localise, the *boundary* of a tampered area. A
blurred boundary is a boundary whose sharpness descriptor is meaningless, and
boundary sharpness is what separates a pasted object from a diffuse artefact.

So the default is a **guided filter** (He, Sun & Tang, ECCV 2010) with the
image's own luminance as the guide: the heatmap is smoothed *within* regions of
consistent image content and left alone across content edges. Tampered-region
borders usually coincide with image edges, so this preserves exactly the
transitions that matter while flattening the window jitter.

Both filters are separable-box / integral-image implementations in numpy: no
scipy, no cv2, and O(N) in the pixel count rather than O(N * r^2), so the radius
can scale with the image without the smoothing step dominating runtime.
"""
from __future__ import annotations

import numpy as np


def box_filter(img: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edge-correct (divides by the real count)."""
    if radius <= 0:
        return img.astype(np.float64, copy=True)

    arr = np.asarray(img, dtype=np.float64)
    ones = np.ones(arr.shape[:2], dtype=np.float64)
    return _box_sum(arr, radius) / _box_sum(ones, radius)


def _box_sum(arr: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2r+1)^2 window via a padded integral image."""
    h, w = arr.shape[:2]
    cum = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    cum = np.pad(cum, ((1, 0), (1, 0)), mode="constant")

    y0 = np.clip(np.arange(h) - radius, 0, h)
    y1 = np.clip(np.arange(h) + radius + 1, 0, h)
    x0 = np.clip(np.arange(w) - radius, 0, w)
    x1 = np.clip(np.arange(w) + radius + 1, 0, w)

    return (
        cum[np.ix_(y1, x1)]
        - cum[np.ix_(y0, x1)]
        - cum[np.ix_(y1, x0)]
        + cum[np.ix_(y0, x0)]
    )


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian. Used for the residual in the forensic features."""
    if sigma <= 0:
        return np.asarray(img, dtype=np.float64)

    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()

    arr = np.asarray(img, dtype=np.float64)
    pad = ((radius, radius), (0, 0)) if arr.ndim == 2 else ((radius, radius), (0, 0), (0, 0))
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, np.pad(arr, pad, mode="reflect"))
    pad = ((0, 0), (radius, radius)) if arr.ndim == 2 else ((0, 0), (radius, radius), (0, 0))
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, np.pad(tmp, pad, mode="reflect"))


def guided_filter(
    target: np.ndarray,
    guide: np.ndarray,
    radius: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Smooth `target` while keeping `guide`'s edges.

    `guide` is expected in [0, 1] (luminance); `eps` sets how large a guide
    variance still counts as "flat". Larger eps -> closer to a box blur.
    """
    if radius <= 0:
        return np.asarray(target, dtype=np.float64)

    p = np.asarray(target, dtype=np.float64)
    g = np.asarray(guide, dtype=np.float64)
    if g.shape != p.shape:
        raise ValueError(f"guide shape {g.shape} != target shape {p.shape}")

    mean_g = box_filter(g, radius)
    mean_p = box_filter(p, radius)
    corr_gg = box_filter(g * g, radius)
    corr_gp = box_filter(g * p, radius)

    var_g = np.maximum(corr_gg - mean_g * mean_g, 0.0)
    cov_gp = corr_gp - mean_g * mean_p

    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g

    return box_filter(a, radius) * g + box_filter(b, radius)


def smooth_heatmap(
    heat: np.ndarray,
    guide: np.ndarray | None,
    method: str = "guided",
    radius: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """Smooth a heatmap, tolerating NaN (uncovered) pixels.

    NaNs are filled with the map's own mean before filtering and restored
    afterwards, so an uncovered pixel neither poisons its neighbourhood nor
    silently acquires a score.
    """
    arr = np.asarray(heat, dtype=np.float64)
    nan = np.isnan(arr)
    if nan.all():
        return arr.astype(np.float32)
    filled = np.where(nan, np.nanmean(arr), arr)

    if method == "none" or radius <= 0:
        out = filled
    elif method == "box":
        out = box_filter(filled, radius)
    elif method == "gaussian":
        out = gaussian_blur(filled, max(1.0, radius / 3.0))
    elif method == "guided":
        if guide is None:
            out = box_filter(filled, radius)
        else:
            out = guided_filter(filled, np.asarray(guide, dtype=np.float64), radius, eps)
    else:
        raise ValueError(f"unknown smoothing method {method!r}")

    out = np.clip(out, 0.0, 1.0)
    out[nan] = np.nan
    return out.astype(np.float32)
