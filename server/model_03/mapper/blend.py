"""Projecting patch scores back onto the image, without square artefacts.

The naive stitch -- paint each patch's score flat into its box and average the
overlaps -- produces a heatmap made of visible rectangles. Region extraction
then finds those rectangles rather than the tampered object, and every
downstream descriptor (boundary sharpness, compactness) reads the grid instead
of the evidence. The blocking is not cosmetic; it corrupts the pipeline.

The fix is to weight each patch's contribution by a kernel that peaks at the
patch centre and falls to (near) zero at its border, so a pixel is described
mostly by the patches that see it centrally:

    score(x, y) = sum_p w_p(x, y) * s_p / sum_p w_p(x, y)

with `w` a separable raised cosine (Hann). The kernel is floored at a small
epsilon rather than truly zero: a pixel touched by exactly one window (possible
at a corner when a scale was clamped) must still get that window's score
instead of 0/0.

Also emitted per scale is the accumulated weight itself, normalised into a
`support` map. Low support means "few windows saw this pixel centrally" -- a
genuine uncertainty signal that the label map uses at the image border, where
support is structurally lower than in the middle.
"""
from __future__ import annotations

import numpy as np

from mapper.windows import Window

EPS = 1e-6

# The kernel is floored well above EPS rather than at it: at the very corner of
# the frame a pixel may be seen by exactly one window, at that window's own
# corner, and a floor of EPS would make its accumulated weight ~1e-12 -- close
# enough to zero to be numerically indistinguishable from "never covered".
KERNEL_FLOOR = 1e-3


def hann2d(height: int, width: int) -> np.ndarray:
    """Separable raised-cosine window: peak 1.0 at the centre, KERNEL_FLOOR at the border."""
    def axis(n: int) -> np.ndarray:
        if n == 1:
            return np.ones(1, dtype=np.float64)
        # np.hanning is zero at both ends; shift off zero so a lone window still counts.
        return np.hanning(n + 2)[1:-1] + KERNEL_FLOOR

    return np.outer(axis(height), axis(width))


class ScoreCanvas:
    """Weighted accumulator for one scale's patch scores."""

    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width = int(width)
        self._num = np.zeros((self.height, self.width), dtype=np.float64)
        self._den = np.zeros((self.height, self.width), dtype=np.float64)
        # Coverage is counted separately from weight: a pixel seen once, at a
        # window's corner, carries almost no weight but is still covered, and
        # only genuinely uncovered pixels may become NaN.
        self._count = np.zeros((self.height, self.width), dtype=np.int32)
        self._kernels: dict[tuple[int, int], np.ndarray] = {}

    def _kernel(self, h: int, w: int) -> np.ndarray:
        # Windows of one scale are near-identical in size, so this cache is
        # essentially "build the kernel once per scale".
        key = (h, w)
        if key not in self._kernels:
            self._kernels[key] = hann2d(h, w)
        return self._kernels[key]

    def add(self, window: Window, score: float) -> None:
        kernel = self._kernel(window.height, window.width)
        ys, xs = slice(window.y0, window.y1), slice(window.x0, window.x1)
        self._num[ys, xs] += kernel * float(score)
        self._den[ys, xs] += kernel
        self._count[ys, xs] += 1

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (heatmap, support), both float32 (H, W).

        `support` is the accumulated kernel weight normalised by its own maximum:
        1.0 where coverage is densest, lower towards the frame edge.
        """
        den = np.maximum(self._den, EPS)
        heat = (self._num / den).astype(np.float32)
        peak = float(self._den.max()) if self._den.size else 0.0
        support = (self._den / peak).astype(np.float32) if peak > 0 else np.zeros_like(heat)
        # Pixels no window touched at all (shouldn't happen -- windows.py clamps
        # to the edges -- but a caller can hand in a hand-built plan).
        heat[self._count == 0] = np.nan
        return heat, support


def blend_scale(
    height: int,
    width: int,
    windows: list[Window],
    scores: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Blend one scale's windows into (heatmap, support)."""
    if len(windows) != len(scores):
        raise ValueError(f"{len(windows)} windows but {len(scores)} scores")
    canvas = ScoreCanvas(height, width)
    for window, score in zip(windows, scores):
        canvas.add(window, score)
    return canvas.result()


def combine_scales(
    per_scale: dict[int, np.ndarray],
    weights: dict[int, float] | None = None,
    method: str = "max",
) -> np.ndarray:
    """Combine the per-scale heatmaps into one map.

    **Why the default is `max` and not `mean`.** The two scales are not two
    independent opinions about the same question. A window only ever reports the
    *average* of everything it covers, so a coarse window containing a 200px
    edit inside a 224px field of authentic pixels reports something close to
    "authentic" -- by construction, not by disagreement. Averaging the scales
    treats that structurally-guaranteed low reading as evidence *against* the
    edit and halves the very signal the fine scale just found. Measured on the
    synthetic case in tests/test_pipeline.py: fine scale peaks at 0.86 inside a
    192px edit, coarse scale cannot exceed 0.48, and the mean lands at 0.63 --
    below any threshold worth setting, so the edit disappears.

    `max` is the right asymmetry: a high score at any scale is evidence, a low
    score at a coarser scale is not counter-evidence. What stops that from
    becoming a false-positive machine is that a single scale still has to clear
    the *high* threshold, and that the scales' disagreement is preserved
    separately as a descriptor -- the router reads it to tell a local edit
    (fine-scale-dominant) from whole-image synthesis (all scales agreeing).

    `mean` is kept for the case where all scales are genuinely commensurable
    (whole-image screening, where every window covers similar content) and for
    ablation.
    """
    if not per_scale:
        raise ValueError("no per-scale maps to combine")
    keys = sorted(per_scale)
    stack = np.stack([np.asarray(per_scale[k], dtype=np.float64) for k in keys])
    # NaN = "no window covered this pixel at this scale"; let the other scales carry it.
    mask = ~np.isnan(stack)

    if method == "max":
        out = np.where(mask.any(axis=0), np.nanmax(np.where(mask, stack, -np.inf), axis=0), np.nan)
        return np.where(np.isneginf(out), np.nan, out).astype(np.float32)

    if method != "mean":
        raise ValueError(f"unknown scale-combination method {method!r}; expected 'max' or 'mean'")

    w = np.array([float((weights or {}).get(k, 1.0)) for k in keys], dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("scale weights sum to zero")
    w = w / w.sum()

    filled = np.where(mask, stack, 0.0)
    wcol = w[:, None, None]
    den = (mask * wcol).sum(axis=0)
    num = (filled * wcol).sum(axis=0)
    out = np.where(den > 0, num / np.maximum(den, EPS), np.nan)
    return out.astype(np.float32)
