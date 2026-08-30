"""Hand-derived forensic measurements shared by the specialists.

Every function here answers a question of the form "does this region's *physics*
match its surroundings?" -- not "does this look generated?", which is the
detector's job. The two kinds of evidence are deliberately separate: the
detector and these statistics fail in different situations, so a region that
lights up both is much stronger evidence than one that lights up either.

The organising idea is **local** comparison. A pasted or inpainted area differs
from the pixels immediately around it, even in an image that is globally
heterogeneous; comparing a region against the whole image instead would call
every dark corner suspicious. So each measurement is computed twice -- inside a
region mask and in a narrow ring just outside it -- and reported as a signed
contrast in [-1, 1]:

    contrast = (inside - outside) / (|inside| + |outside|)

which is scale-free, symmetric, and 0 when the two match.

What each measurement targets:

* **residual noise level** -- a camera lays down a roughly uniform sensor-noise
  floor. A decoder does not: generated content is typically *smoother* than its
  surroundings (negative contrast), while a badly composited paste from another
  photo carries a foreign, often stronger, noise floor.
* **high-frequency energy ratio** -- diffusion decoders manufacture high
  frequencies by upsampling, leaving less true fine detail than optics do.
* **JPEG block-grid energy** -- an image compressed once has a coherent 8x8
  grid everywhere. Paste in content that was decoded, edited and re-encoded and
  its grid phase no longer matches (or its grid is gone entirely, having been
  resampled). This one is loud on a first-generation JPEG and silent on a PNG.
* **ELA (error level analysis)** -- recompress and difference. Areas that have
  been through fewer compression generations than their surroundings give up
  more error on the next round. Classic, and classically over-read: ELA on a
  re-saved image says nothing at all, which is why it is one input among
  several and never a verdict.
* **edge straightness** -- how much of a region's rim is axis-aligned. Machine
  selections (rectangular crops, box masks, generative-fill selections) leave
  straight rims; objects do not.

None of these are learned, none have thresholds tuned on a benchmark, and each
is individually beatable. They are here to *corroborate or contradict* the
map -- see fusion.py, where a specialist's confidence is what decides how much
its probability moves the verdict.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from mapper.smooth import box_filter, gaussian_blur

EPS = 1e-8
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def to_luma(image: Image.Image | np.ndarray) -> np.ndarray:
    """Luminance in [0, 1] as float64 (H, W)."""
    arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.float64)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = arr[..., :3] @ LUMA_WEIGHTS
    return gray / 255.0 if gray.max() > 1.5 else gray


def residual(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """High-pass residual: the image minus its blurred self -- the noise floor."""
    return np.asarray(gray, dtype=np.float64) - gaussian_blur(gray, sigma)


def contrast(inside: float, outside: float) -> float:
    """Signed, scale-free difference in [-1, 1]. 0 when the two agree."""
    denom = abs(inside) + abs(outside)
    if denom < EPS:
        return 0.0
    return float(np.clip((inside - outside) / denom, -1.0, 1.0))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    sel = values[mask]
    return float(sel.mean()) if sel.size else 0.0


def noise_level(gray: np.ndarray, mask: np.ndarray, sigma: float = 1.0) -> float:
    """Robust noise-floor estimate over `mask` (MAD of the residual, MAD-scaled)."""
    res = residual(gray, sigma)[mask]
    if res.size < 8:
        return 0.0
    return float(1.4826 * np.median(np.abs(res - np.median(res))))


def high_freq_ratio(gray: np.ndarray, mask: np.ndarray) -> float:
    """Energy above ~1 cycle/2px relative to total, inside `mask`.

    Computed as a ratio of residual energy to total local variance, so a flat
    sky and a textured wall are comparable -- what is measured is how much of
    the local structure is *fine*, not how much structure there is.
    """
    g = np.asarray(gray, dtype=np.float64)
    fine = residual(g, 0.8)
    coarse = g - gaussian_blur(g, 3.0)
    fine_e = _masked_mean(fine ** 2, mask)
    coarse_e = _masked_mean(coarse ** 2, mask)
    return float(fine_e / (coarse_e + EPS))


def jpeg_grid_energy(gray: np.ndarray, mask: np.ndarray) -> float:
    """Blockiness at the JPEG 8x8 lattice, inside `mask`.

    Compares the mean absolute step across columns/rows that sit *on* the 8px
    lattice against those that do not. ~1.0 means no grid; larger means a
    visible JPEG grid in that area.
    """
    g = np.asarray(gray, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if m.sum() < 64:
        return 1.0

    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    mx = m[:, :-1] & m[:, 1:]
    my = m[:-1, :] & m[1:, :]

    cols = np.arange(dx.shape[1])
    rows = np.arange(dy.shape[0])
    on_x = ((cols + 1) % 8 == 0)[None, :] & mx
    off_x = (~((cols + 1) % 8 == 0))[None, :] & mx
    on_y = ((rows + 1) % 8 == 0)[:, None] & my
    off_y = (~((rows + 1) % 8 == 0))[:, None] & my

    on = np.concatenate([dx[on_x], dy[on_y]]) if (on_x.any() or on_y.any()) else np.array([])
    off = np.concatenate([dx[off_x], dy[off_y]]) if (off_x.any() or off_y.any()) else np.array([])
    if on.size < 8 or off.size < 8:
        return 1.0
    return float((on.mean() + EPS) / (off.mean() + EPS))


def ela_map(image: Image.Image, quality: int = 90) -> np.ndarray:
    """Error level analysis: |image - JPEG(image, quality)| as float64 luma in [0, 1]."""
    rgb = image.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(rgb, dtype=np.float64) - np.asarray(recompressed, dtype=np.float64))
    return (diff @ LUMA_WEIGHTS) / 255.0


def channel_residual_correlation(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Mean pairwise correlation of the per-channel residuals inside `mask`.

    Demosaicing ties a camera's channel noise together in a characteristic way;
    a decoder's does not have to obey it. Reported as a single mean so it can be
    contrasted inside-vs-outside like everything else here.
    """
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0
    arr = arr / 255.0 if arr.max() > 1.5 else arr

    res = [residual(arr[..., c], 1.0)[mask] for c in range(3)]
    if res[0].size < 32:
        return 0.0
    corrs = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = res[i], res[j]
        denom = a.std() * b.std()
        corrs.append(float(((a - a.mean()) * (b - b.mean())).mean() / (denom + EPS)) if denom > EPS else 0.0)
    return float(np.mean(corrs))


def edge_straightness(mask: np.ndarray) -> float:
    """Fraction of the region's rim that lies on a long axis-aligned run.

    Straight rims come from machine selections -- rectangular pastes, box masks,
    generative-fill selections, outpainting frames. Objects have curved,
    irregular rims. Runs of 5+ boundary pixels in a single row or column count
    as straight.
    """
    from regions.components import boundary  # local import: components imports nothing from here

    rim = boundary(np.asarray(mask, dtype=bool))
    total = int(rim.sum())
    if total == 0:
        return 0.0

    straight = 0
    for axis_view in (rim, rim.T):
        for line in axis_view:
            idx = np.flatnonzero(line)
            if idx.size < 5:
                continue
            # Count pixels belonging to runs of >= 5 consecutive rim pixels.
            splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
            straight += sum(run.size for run in splits if run.size >= 5)
    return float(min(1.0, straight / (2.0 * total)))


def local_variance(gray: np.ndarray, radius: int = 3) -> np.ndarray:
    """Per-pixel local variance -- used to render noise-inconsistency maps."""
    g = np.asarray(gray, dtype=np.float64)
    mean = box_filter(g, radius)
    return np.maximum(box_filter(g * g, radius) - mean * mean, 0.0)


def region_report(
    image: Image.Image,
    mask: np.ndarray,
    ring_mask: np.ndarray,
) -> dict:
    """Every measurement above, inside-vs-ring, as one dict of contrasts.

    The specialists all start from this and differ only in which entries they
    weigh and how they narrate them. Computing it once per region keeps a
    multi-specialist route from paying for the same statistics twice.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    gray = to_luma(image)
    m = np.asarray(mask, dtype=bool)
    r = np.asarray(ring_mask, dtype=bool)

    if m.sum() < 32 or r.sum() < 32:
        return {"valid": False, "reason": "region or surrounding ring too small to measure"}

    ela = ela_map(image, quality=90)
    hf_in, hf_out = high_freq_ratio(gray, m), high_freq_ratio(gray, r)
    noise_in, noise_out = noise_level(gray, m), noise_level(gray, r)
    grid_in, grid_out = jpeg_grid_energy(gray, m), jpeg_grid_energy(gray, r)
    ela_in, ela_out = _masked_mean(ela, m), _masked_mean(ela, r)
    corr_in, corr_out = channel_residual_correlation(rgb, m), channel_residual_correlation(rgb, r)

    return {
        "valid": True,
        "noise_inside": noise_in,
        "noise_outside": noise_out,
        "noise_contrast": contrast(noise_in, noise_out),
        "high_freq_inside": hf_in,
        "high_freq_outside": hf_out,
        "high_freq_contrast": contrast(hf_in, hf_out),
        "jpeg_grid_inside": grid_in,
        "jpeg_grid_outside": grid_out,
        "jpeg_grid_contrast": contrast(grid_in, grid_out),
        "ela_inside": ela_in,
        "ela_outside": ela_out,
        "ela_contrast": contrast(ela_in, ela_out),
        "channel_corr_inside": corr_in,
        "channel_corr_outside": corr_out,
        "channel_corr_contrast": contrast(corr_in, corr_out),
        "edge_straightness": edge_straightness(m),
    }
