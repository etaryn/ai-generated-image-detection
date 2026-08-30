"""Layer 1: the calibrated AI-likelihood map.

    image
      -> multi-scale windows            (windows.py)
      -> batched patch scoring          (backends.py)
      -> centre-weighted blending       (blend.py)
      -> calibration                    (calibration.py)
      -> edge-preserving smoothing      (smooth.py)
      -> three-way label map            (here)

The output is deliberately three-valued, not binary. "Likely AI" and "likely
non-AI" are the two confident answers; everything between the thresholds is
`uncertain`, and uncertainty is carried forward rather than rounded away --
Layer 2 spends no specialist compute on confident non-AI area, and fusion is
forbidden from turning an uncertain region into a definitive verdict on one
specialist's say-so.

Uncertainty has three sources here, and they are kept distinct because they mean
different things:

* **score uncertainty** -- the calibrated probability sits between the
  thresholds. The detector genuinely has no opinion.
* **scale disagreement** -- the per-scale maps disagree about a pixel. Real
  evidence about *what kind* of tampering it might be, so it is recorded as a
  descriptor rather than treated as noise.
* **low support** -- few windows covered the pixel centrally (structurally true
  near the frame edge). Not the detector's fault, but not confidence either.

Work happens at a bounded working resolution: a 4000px upload at 128px/50%
overlap is ~4000 patches, which no demo survives. The image is downscaled so its
long side is at most `max_side`, and both sizes are recorded so callers can map
coordinates back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image

from mapper.blend import blend_scale, combine_scales
from mapper.calibration import Calibrator
from mapper.smooth import smooth_heatmap
from mapper.windows import Window, count_windows, plan_windows

LABEL_NON_AI = 0
LABEL_UNCERTAIN = 1
LABEL_AI = 2

LABEL_NAMES = {LABEL_NON_AI: "likely_non_ai", LABEL_UNCERTAIN: "uncertain", LABEL_AI: "likely_ai"}


@dataclass
class AILikelihoodMap:
    """Everything Layer 1 knows, at working resolution."""

    heat: np.ndarray                       # (H, W) float32, calibrated P(AI), NaN where uncovered
    labels: np.ndarray                     # (H, W) uint8, one of LABEL_*
    support: np.ndarray                    # (H, W) float32 in [0, 1], window coverage density
    per_scale: dict[int, np.ndarray]       # nominal scale -> its own blended heatmap
    working_size: tuple[int, int]          # (W, H) the map is computed at
    original_size: tuple[int, int]         # (W, H) of the uploaded image
    working_image: Image.Image             # the downscaled RGB image the map describes
    thresholds: tuple[float, float]        # (lo, hi)
    calibrated: bool
    meta: dict = field(default_factory=dict)

    @property
    def scale_factor(self) -> float:
        """Multiply a working-resolution coordinate by this to get an original one."""
        return self.original_size[0] / max(1, self.working_size[0])

    def fraction(self, label: int) -> float:
        return float((self.labels == label).mean())

    def scale_disagreement(self) -> np.ndarray:
        """Per-pixel spread across scales -- 0 when the scales agree."""
        if len(self.per_scale) < 2:
            return np.zeros(self.heat.shape, dtype=np.float32)
        stack = np.stack([np.nan_to_num(v, nan=0.5) for v in self.per_scale.values()])
        return (stack.max(axis=0) - stack.min(axis=0)).astype(np.float32)

    def summary(self) -> dict:
        heat = self.heat[~np.isnan(self.heat)]
        return {
            "mean_score": float(heat.mean()) if heat.size else float("nan"),
            "max_score": float(heat.max()) if heat.size else float("nan"),
            "p95_score": float(np.percentile(heat, 95)) if heat.size else float("nan"),
            "frac_likely_ai": self.fraction(LABEL_AI),
            "frac_uncertain": self.fraction(LABEL_UNCERTAIN),
            "frac_likely_non_ai": self.fraction(LABEL_NON_AI),
            "mean_scale_disagreement": float(self.scale_disagreement().mean()),
            "calibrated": self.calibrated,
            "scales": sorted(self.per_scale),
            "working_size": list(self.working_size),
            "original_size": list(self.original_size),
            **self.meta,
        }


class AILikelihoodMapper:
    """Turns one image into an `AILikelihoodMap`."""

    def __init__(
        self,
        scorer,
        scales: Sequence[int] = (64, 128, 224),
        overlap: float = 0.5,
        threshold_hi: float = 0.75,
        threshold_lo: float = 0.45,
        max_side: int = 1024,
        smoothing: str = "guided",
        smooth_radius: int = 8,
        smooth_eps: float = 1e-3,
        min_support: float = 0.15,
        calibrator: Calibrator | None = None,
        scale_weights: dict[int, float] | None = None,
        scale_combine: str = "max",
    ):
        if not 0.0 <= threshold_lo <= threshold_hi <= 1.0:
            raise ValueError(
                f"thresholds must satisfy 0 <= lo ({threshold_lo}) <= hi ({threshold_hi}) <= 1"
            )
        self.scorer = scorer
        self.scales = [int(s) for s in scales]
        self.overlap = float(overlap)
        self.threshold_hi = float(threshold_hi)
        self.threshold_lo = float(threshold_lo)
        self.max_side = int(max_side)
        self.smoothing = smoothing
        self.smooth_radius = int(smooth_radius)
        self.smooth_eps = float(smooth_eps)
        self.min_support = float(min_support)
        self.calibrator = calibrator or Calibrator.identity()
        self.scale_weights = scale_weights
        self.scale_combine = scale_combine

    def _to_working(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        long_side = max(rgb.size)
        if long_side <= self.max_side:
            return rgb
        factor = self.max_side / long_side
        size = (max(1, int(round(rgb.width * factor))), max(1, int(round(rgb.height * factor))))
        return rgb.resize(size, Image.BICUBIC)

    def _calibrate(self, scores: np.ndarray, scale: int) -> np.ndarray:
        """Apply the calibrator, per scale when it knows about scales."""
        try:
            return self.calibrator.apply(scores, scale=scale)
        except TypeError:
            return self.calibrator.apply(scores)

    def _score_windows(self, image: Image.Image, windows: list[Window]) -> list[float]:
        patches = [image.crop(w.box) for w in windows]
        return self.scorer.score_patches(patches)

    def run(self, image: Image.Image) -> AILikelihoodMap:
        work = self._to_working(image)
        width, height = work.size

        plan = plan_windows(width, height, self.scales, self.overlap)
        if not plan:
            raise ValueError(f"no windows planned for a {width}x{height} image")

        per_scale: dict[int, np.ndarray] = {}
        supports: list[np.ndarray] = []
        for scale, windows in plan.items():
            scores = self._score_windows(work, windows)
            # Calibration is per patch, not per pixel: the fit was measured on
            # patch scores, so it has to be applied before the blend averages
            # them (calibrate-then-average and average-then-calibrate differ
            # whenever the map is non-linear, which it is).
            #
            # It is also per *scale*. A 64px crop and a 224px crop are different
            # questions to a detector trained on whole images, and measurably so
            # -- see ScaleCalibrators. Passing the scale lets each one be
            # corrected by its own fit; a plain Calibrator ignores the argument.
            scores = self._calibrate(np.asarray(scores), scale).tolist()
            heat, support = blend_scale(height, width, windows, scores)
            per_scale[scale] = heat
            supports.append(support)

        combined = combine_scales(per_scale, self.scale_weights, self.scale_combine)
        support = np.mean(np.stack(supports), axis=0).astype(np.float32)

        guide = np.asarray(work.convert("L"), dtype=np.float64) / 255.0
        smoothed = smooth_heatmap(
            combined, guide, self.smoothing, self.smooth_radius, self.smooth_eps
        )

        labels = self._label(smoothed, support)

        return AILikelihoodMap(
            heat=smoothed,
            labels=labels,
            support=support,
            per_scale=per_scale,
            working_size=(width, height),
            original_size=image.size,
            working_image=work,
            thresholds=(self.threshold_lo, self.threshold_hi),
            calibrated=bool(self.calibrator.fitted),
            meta={
                "backend": getattr(self.scorer, "name", "unknown"),
                "windows_scored": count_windows(plan),
                "overlap": self.overlap,
                "smoothing": self.smoothing,
            },
        )

    def _label(self, heat: np.ndarray, support: np.ndarray) -> np.ndarray:
        labels = np.full(heat.shape, LABEL_UNCERTAIN, dtype=np.uint8)
        known = ~np.isnan(heat)
        labels[known & (heat >= self.threshold_hi)] = LABEL_AI
        labels[known & (heat <= self.threshold_lo)] = LABEL_NON_AI

        # Thinly-covered pixels get demoted to uncertain rather than trusted:
        # the frame edge is where support is structurally lowest and where a
        # confident label would be least earned.
        labels[support < self.min_support] = LABEL_UNCERTAIN
        labels[~known] = LABEL_UNCERTAIN
        return labels
