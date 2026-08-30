"""Multi-scale sliding-window geometry.

Two rules drive everything here:

1. **Every pixel must be covered.** A pixel no window touches has no evidence,
   and an uncovered strip along the right/bottom edge is exactly where a
   generative-expansion edit tends to live. Naive `range(0, W - patch, stride)`
   leaves such a strip whenever `(W - patch) % stride != 0`, so the last window
   in each axis is *clamped* to the edge instead of dropped. Clamping makes the
   final overlap larger than the nominal stride, which is harmless -- the
   blender normalises by accumulated weight.

2. **Scale is evidence, not just a hyperparameter.** A 128px window and a 224px
   window disagreeing about the same pixel is informative: a small pasted object
   lights up the fine scale and gets diluted at the coarse one, while a fully
   synthetic image lights up both. So scales are kept separate all the way
   through blending, and the router reads the per-scale profile
   (see regions/proposals.py).

Images smaller than a requested scale are not skipped -- the scale is clamped to
the short side, and duplicate scales that result are dropped, so a 96px thumbnail
still gets a (single-scale) map rather than an empty one.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """One patch location. Coordinates are pixels, x1/y1 exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int
    scale: int  # the nominal scale this window belongs to, pre-clamping

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def box(self) -> tuple[int, int, int, int]:
        """PIL-style crop box."""
        return (self.x0, self.y0, self.x1, self.y1)


def _starts(length: int, patch: int, stride: int) -> list[int]:
    """Window start offsets along one axis, with the last one clamped to the edge."""
    if patch >= length:
        return [0]
    out = list(range(0, length - patch + 1, stride))
    last = length - patch
    if out[-1] != last:
        out.append(last)
    return out


def effective_scales(width: int, height: int, scales: list[int]) -> list[int]:
    """Clamp requested scales to the image and drop duplicates, order preserved.

    A 224px scale on a 160px image becomes a 160px scale; if 128 was also
    requested it survives separately, so the two-scale story holds until the
    image is genuinely too small for it.
    """
    short_side = min(width, height)
    seen: set[int] = set()
    out: list[int] = []
    for scale in scales:
        clamped = max(8, min(int(scale), short_side))
        if clamped not in seen:
            seen.add(clamped)
            out.append(clamped)
    return out


def plan_windows(
    width: int,
    height: int,
    scales: list[int],
    overlap: float = 0.5,
) -> dict[int, list[Window]]:
    """Plan every window for every scale.

    `overlap` is the fraction of a patch shared with its neighbour, so 0.5 means
    a stride of half the patch. Returns {nominal_scale: [Window, ...]}, keyed by
    the *requested* scale so callers can talk about "the 224px map" even when
    the windows were clamped smaller.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate image size {width}x{height}")

    plan: dict[int, list[Window]] = {}
    for scale in effective_scales(width, height, scales):
        patch = min(scale, width, height)
        stride = max(1, int(round(patch * (1.0 - overlap))))
        windows = [
            Window(x0, y0, x0 + patch, y0 + patch, scale)
            for y0 in _starts(height, patch, stride)
            for x0 in _starts(width, patch, stride)
        ]
        plan[scale] = windows
    return plan


def count_windows(plan: dict[int, list[Window]]) -> int:
    return sum(len(v) for v in plan.values())
