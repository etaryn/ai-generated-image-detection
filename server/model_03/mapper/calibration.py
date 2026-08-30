"""Making the map's numbers mean what they say.

The report calls Layer 1 a *calibrated* AI-likelihood map, and that word is
load-bearing: the whole design routes on thresholds ("likely AI" above hi,
"uncertain" between lo and hi), and a threshold on an uncalibrated score is an
arbitrary cut. Two specific distortions make raw patch scores unsafe to
threshold:

* **Scale shift.** Both sibling detectors were trained on whole images. Asked
  about a 128px crop they answer a different question, and their scores drift --
  typically towards the middle, sometimes towards a corner, depending on the
  backend. The mapping from "what the detector says about a fragment" to "how
  likely that fragment is generated" has to be measured.
* **Saturation.** Both siblings saturate hard (client/app.py already prints the
  unrounded score because "100.0%" hides everything). A field of 0.9999s and a
  field of 0.93s look identical after thresholding but are not the same
  evidence.

So: fit a monotone map from patch score to patch-level P(AI) on held-out data,
store it beside the config, and apply it before anything reads a threshold.
Monotone matters -- calibration must not reorder patches, only rescale them.

Two fits, both dependency-free (no sklearn):

* `platt` -- a two-parameter logistic on the logit. Smooth, extrapolates
  sanely, needs little data. The default.
* `isotonic` -- pool-adjacent-violators, stored as knots and linearly
  interpolated. Fits any monotone distortion, and will happily overfit a few
  hundred patches, so it wants thousands.

`identity` is what you get when nothing has been fitted. It is a legitimate
starting point, but `Calibrator.identity()` reports `fitted=False` and the
mapper says so in its metadata, so an uncalibrated map is never quietly passed
off as a calibrated one.

Fit one with scripts/calibrate_mapper.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass
class Calibrator:
    """A monotone score -> probability map, serialisable to JSON."""

    kind: str = "identity"
    params: dict = field(default_factory=dict)
    fitted: bool = False
    meta: dict = field(default_factory=dict)

    @classmethod
    def identity(cls) -> "Calibrator":
        return cls(kind="identity", fitted=False)

    def apply(self, scores) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float64)
        nan = np.isnan(arr)
        safe = np.where(nan, 0.5, arr)

        if self.kind == "identity":
            out = safe
        elif self.kind == "platt":
            a = float(self.params["a"])
            b = float(self.params["b"])
            out = _sigmoid(a * _logit(safe) + b)
        elif self.kind == "isotonic":
            x = np.asarray(self.params["x"], dtype=np.float64)
            y = np.asarray(self.params["y"], dtype=np.float64)
            out = np.interp(safe, x, y)
        else:
            raise ValueError(f"unknown calibrator kind {self.kind!r}")

        out = np.clip(out, 0.0, 1.0)
        return np.where(nan, np.nan, out).astype(np.float32)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.params, "fitted": self.fitted, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibrator":
        if not d:
            return cls.identity()
        return cls(
            kind=d.get("kind", "identity"),
            params=d.get("params", {}) or {},
            fitted=bool(d.get("fitted", False)),
            meta=d.get("meta", {}) or {},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibrator":
        if path is None:
            return cls.identity()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"No calibration file at {p}. Fit one with scripts/calibrate_mapper.py, "
                f"or set mapper.calibration.path to null to run uncalibrated."
            )
        return cls.from_dict(json.loads(p.read_text()))


def fit_platt(scores, labels, iters: int = 200, lr: float = 0.5) -> Calibrator:
    """Fit P(AI) = sigmoid(a * logit(s) + b) by Newton steps on the log-loss.

    Two parameters on a 1-D problem, so plain Newton converges in a handful of
    iterations; `lr` damps the step for the pathological cases (perfectly
    separable input, where `a` would otherwise run away).
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if s.size != y.size:
        raise ValueError(f"{s.size} scores but {y.size} labels")
    if s.size < 4:
        raise ValueError("need at least 4 patches to fit a calibrator")
    if len(np.unique(y)) < 2:
        raise ValueError("calibration needs both classes present")

    z = _logit(s)
    X = np.stack([z, np.ones_like(z)], axis=1)
    w = np.array([1.0, 0.0])

    for _ in range(iters):
        p = _sigmoid(X @ w)
        grad = X.T @ (p - y)
        W = np.clip(p * (1.0 - p), 1e-8, None)
        hess = (X * W[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w_new = w - lr * step
        if not np.all(np.isfinite(w_new)):
            break
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new

    # Calibration must not reorder patches. A negative slope would do exactly
    # that, and always means the fit data was mislabelled rather than that the
    # detector is anti-correlated.
    if w[0] <= 0:
        raise ValueError(
            f"Platt fit produced a non-monotone slope (a={w[0]:.4g}), which would "
            f"reorder patches. Check that label 1 means AI-generated."
        )

    return Calibrator(
        kind="platt",
        params={"a": float(w[0]), "b": float(w[1])},
        fitted=True,
        meta={"n": int(s.size), "positives": int(y.sum())},
    )


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: nearest non-decreasing fit to y under weights w."""
    values = list(y.astype(np.float64))
    weights = list(w.astype(np.float64))
    counts = [1] * len(values)

    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        total_w = weights[i] + weights[i + 1]
        values[i] = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total_w
        weights[i] = total_w
        counts[i] += counts[i + 1]
        del values[i + 1], weights[i + 1], counts[i + 1]
        if i > 0:
            i -= 1

    out = np.empty(int(sum(counts)), dtype=np.float64)
    pos = 0
    for value, count in zip(values, counts):
        out[pos : pos + count] = value
        pos += count
    return out


def fit_isotonic(scores, labels, max_knots: int = 64) -> Calibrator:
    """Fit a monotone step function by PAVA, stored as `max_knots` knots.

    Thinning to knots keeps the JSON small and the interpolation smooth; the
    full PAVA solution is a staircase with as many steps as distinct scores,
    which would just memorise the fit set.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if s.size != y.size:
        raise ValueError(f"{s.size} scores but {y.size} labels")
    if s.size < max_knots:
        raise ValueError(f"need at least {max_knots} patches for an isotonic fit; got {s.size}")
    if len(np.unique(y)) < 2:
        raise ValueError("calibration needs both classes present")

    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    fitted = _pava(y_sorted, np.ones_like(y_sorted))

    idx = np.unique(np.linspace(0, s.size - 1, max_knots).round().astype(int))
    x_knots = s_sorted[idx]
    y_knots = fitted[idx]

    # np.interp needs strictly increasing x; collapse ties to their mean y.
    x_unique, inverse = np.unique(x_knots, return_inverse=True)
    y_unique = np.zeros_like(x_unique)
    np.add.at(y_unique, inverse, y_knots)
    counts = np.bincount(inverse, minlength=x_unique.size)
    y_unique = y_unique / np.maximum(counts, 1)
    y_unique = np.maximum.accumulate(y_unique)  # guard monotonicity after averaging

    return Calibrator(
        kind="isotonic",
        params={"x": x_unique.tolist(), "y": y_unique.tolist()},
        fitted=True,
        meta={"n": int(s.size), "positives": int(y.sum()), "knots": int(x_unique.size)},
    )


class ScaleCalibrators:
    """One calibrator per window scale, because the distortion *is* scale-dependent.

    Measured on SID-Set with the default backend, the fraction of patches from
    **authentic photographs** scoring above the 0.75 "likely AI" threshold:

        64px    36.6%
        128px   16.3%
        224px   10.4%

    while patches of fully-synthetic images scored ~0.71 at every scale. So the
    fine scale is not more sensitive to generated content -- it is more prone to
    calling *authentic* content generated, because a 64px crop upscaled to the
    detector's 224px input looks smooth and textureless, which is what the model
    was trained to read as "generated".

    A single calibrator cannot fix that. It applies one monotone map to every
    patch, so pulling the 64px scale's false positives down would drag the
    coarse scales' true positives down with them. Three separate maps, each
    fitted on its own scale's score distribution, is the correction the data
    actually calls for -- and it preserves what the multi-scale design is for,
    since after calibration the scales become directly comparable and `max`
    over them stops being dominated by whichever scale is most miscalibrated.

    Falls back to the shared calibrator for any scale that was not fitted, so a
    config that adds a fourth scale degrades rather than breaks.
    """

    def __init__(self, per_scale: dict[int, Calibrator] | None = None, shared: Calibrator | None = None):
        self.per_scale = {int(k): v for k, v in (per_scale or {}).items()}
        self.shared = shared or Calibrator.identity()

    @property
    def fitted(self) -> bool:
        return bool(self.per_scale) or self.shared.fitted

    def for_scale(self, scale: int) -> Calibrator:
        return self.per_scale.get(int(scale), self.shared)

    def apply(self, scores, scale: int | None = None) -> np.ndarray:
        if scale is None:
            return self.shared.apply(scores)
        return self.for_scale(scale).apply(scores)

    def to_dict(self) -> dict:
        return {
            "kind": "per_scale",
            "shared": self.shared.to_dict(),
            "scales": {str(k): v.to_dict() for k, v in sorted(self.per_scale.items())},
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "ScaleCalibrators":
        if not d:
            return cls()
        if d.get("kind") != "per_scale":
            # A single-calibrator file stays valid: it becomes the shared map.
            return cls(shared=Calibrator.from_dict(d))
        return cls(
            per_scale={int(k): Calibrator.from_dict(v) for k, v in (d.get("scales") or {}).items()},
            shared=Calibrator.from_dict(d.get("shared")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path | None) -> "ScaleCalibrators":
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"No calibration file at {p}. Fit one with scripts/calibrate_mapper.py, "
                f"or set mapper.calibration_path to null to run uncalibrated."
            )
        return cls.from_dict(json.loads(p.read_text()))

    def describe(self) -> dict:
        return {
            "fitted": self.fitted,
            "scales_fitted": sorted(self.per_scale),
            "shared_fitted": self.shared.fitted,
            "meta": {str(k): v.meta for k, v in sorted(self.per_scale.items())},
        }


def expected_calibration_error(probs, labels, bins: int = 10) -> float:
    """Standard ECE, for reporting how much a fit actually bought."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)
