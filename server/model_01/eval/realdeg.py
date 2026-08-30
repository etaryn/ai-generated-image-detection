"""RealDeg-style degradation protocol for out-of-distribution robustness eval.

Implements the benchmark protocol described in GlobalForge (arXiv:2607.14684) and
summarised in aigc-detector-improvement-notes.md, because our existing
`SEVERITY_LEVELS` matrix cannot answer the question we actually have:

  * SEVERITY_LEVELS applies ONE operator at a fixed strength (plus two hand-written
    compounds). Real redistribution chains several operators at random strengths --
    an image gets resized by one platform, re-compressed by the next, filtered by a
    third. A fixed single-operator sweep cannot characterise decay along that chain.
  * Its two compounds are fixed recipes, so they measure two points, not a curve.

Two protocols, matching the paper:

  single    -- one operator, strength drawn uniformly from that operator's set.
  compound  -- N in {1..5} operators drawn uniformly WITH REPLACEMENT from the pool,
               each strength drawn independently. With-replacement is deliberate:
               it is what models an image being re-compressed twice by two platforms.

Everything is seeded and every draw is recorded, so a run is reproducible from its
manifest alone. This matters because the primitives in `data/transforms.py` are NOT
reproducible -- `gaussian_noise` calls the global `np.random` and `color_jitter`
calls the global `random`, so two runs of the same "severity" give different images.
The operators below re-implement those two with explicit seeding.

Metric is Balanced Accuracy, not raw accuracy: our degraded eval sets are not
guaranteed class-balanced, and raw accuracy on an unbalanced set rewards a model
that collapses to the majority class -- exactly the failure mode we are hunting.

Usage:
    from eval.realdeg import RealDegSampler, apply_chain, balanced_accuracy

    sampler = RealDegSampler(seed=0)
    chain = sampler.compound()                  # e.g. [("jpeg", 40), ("resize", 0.3)]
    degraded = apply_chain(img, chain)
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# --------------------------------------------------------------------------- #
# Operators. Strength sets are taken verbatim from the RealDeg-Bench table.
# --------------------------------------------------------------------------- #

def _jpeg(img: Image.Image, q: float, rng: np.random.Generator) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _resize(img: Image.Image, scale: float, rng: np.random.Generator) -> Image.Image:
    """Downscale then restore original size -- the pixelation failure mode."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _noise(img: Image.Image, variance: float, rng: np.random.Generator) -> Image.Image:
    """Additive Gaussian noise. Strength is VARIANCE (per the bench table), so the
    std passed to the normal is sqrt(variance)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    noise = rng.normal(0.0, float(np.sqrt(variance)), arr.shape).astype(np.float32)
    return Image.fromarray((np.clip(arr + noise, 0, 1) * 255).astype(np.uint8), "RGB")


def _brightness(img: Image.Image, shift: float, rng: np.random.Generator) -> Image.Image:
    """Strength is a SHIFT; PIL enhancers take a factor where 1.0 is identity."""
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(1.0 + shift)


def _contrast(img: Image.Image, shift: float, rng: np.random.Generator) -> Image.Image:
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(1.0 + shift)


def _saturation(img: Image.Image, factor: float, rng: np.random.Generator) -> Image.Image:
    """Strength is already a factor here, not a shift -- matches the bench table."""
    return ImageEnhance.Color(img.convert("RGB")).enhance(factor)


OPERATORS: dict[str, Callable[[Image.Image, float, np.random.Generator], Image.Image]] = {
    "jpeg": _jpeg,
    "blur": _blur,
    "resize": _resize,
    "noise": _noise,
    "brightness": _brightness,
    "contrast": _contrast,
    "saturation": _saturation,
}

STRENGTHS: dict[str, tuple[float, ...]] = {
    "jpeg":       (90, 80, 70, 60, 40),
    "blur":       (0.5, 1, 2, 3, 5),
    "resize":     (0.9, 0.7, 0.5, 0.3, 0.2),   # closest proxy for the 1024^2 pixelation failure
    "noise":      (0.0005, 0.001, 0.002, 0.005, 0.01),
    "brightness": (-0.2, -0.1, 0.1, 0.2),
    "contrast":   (-0.3, -0.2, 0.1, 0.2),
    "saturation": (0.6, 0.8, 1.3, 1.5),
}

MAX_COMPOUND_STEPS = 5


@dataclass(frozen=True)
class Step:
    op: str
    strength: float


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

class RealDegSampler:
    """Draws degradation chains reproducibly from a single seed."""

    def __init__(self, seed: int = 0, operators: Sequence[str] | None = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.operators = tuple(operators or OPERATORS.keys())
        unknown = set(self.operators) - set(OPERATORS)
        if unknown:
            raise KeyError(f"unknown operators: {sorted(unknown)}")

    def _draw(self, op: str) -> Step:
        return Step(op, float(self.rng.choice(STRENGTHS[op])))

    def single(self, op: str | None = None) -> list[Step]:
        """One operator at one strength."""
        chosen = op if op is not None else str(self.rng.choice(self.operators))
        return [self._draw(chosen)]

    def compound(self, n_steps: int | None = None) -> list[Step]:
        """N operators drawn WITH REPLACEMENT, each with an independent strength."""
        n = n_steps if n_steps is not None else int(self.rng.integers(1, MAX_COMPOUND_STEPS + 1))
        return [self._draw(str(self.rng.choice(self.operators))) for _ in range(n)]


def apply_chain(img: Image.Image, chain: Iterable[Step], seed: int = 0) -> Image.Image:
    """Apply a recorded chain. Seeded, so replaying a manifest reproduces the image."""
    rng = np.random.default_rng(seed)
    out = img.convert("RGB")
    for step in chain:
        out = OPERATORS[step.op](out, step.strength, rng)
    return out


# --------------------------------------------------------------------------- #
# Manifest -- the reproducibility contract
# --------------------------------------------------------------------------- #

def build_manifest(paths: Sequence[str], labels: Sequence[int], seed: int = 0) -> dict:
    """One condition set: clean + 7 single-operator + 5 compound depths.

    Mirrors the bench's 13 conditions. Returns a JSON-serialisable dict recording
    every drawn operator and strength, so `apply_chain` can regenerate byte-identical
    inputs later.
    """
    if len(paths) != len(labels):
        raise ValueError(f"paths/labels length mismatch: {len(paths)} vs {len(labels)}")
    sampler = RealDegSampler(seed=seed)
    conditions: dict[str, list[list[dict]]] = {"clean": [[] for _ in paths]}
    for op in sampler.operators:
        conditions[f"single_{op}"] = [[asdict(s) for s in sampler.single(op)] for _ in paths]
    for n in range(1, MAX_COMPOUND_STEPS + 1):
        conditions[f"compound_{n}step"] = [[asdict(s) for s in sampler.compound(n)] for _ in paths]
    return {"seed": seed, "paths": list(paths), "labels": [int(x) for x in labels],
            "conditions": conditions}


def save_manifest(manifest: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2))


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def chain_from_manifest(entry: Sequence[dict]) -> list[Step]:
    return [Step(d["op"], d["strength"]) for d in entry]


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #

def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Mean of per-class recall. Unlike raw accuracy this does not reward a model
    that collapses to the majority class -- which is exactly what a detector does
    once degradation destroys its cue."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    recalls = []
    for cls in (0, 1):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")
