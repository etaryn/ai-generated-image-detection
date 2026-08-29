"""Common interface for the three feature extractors.

Every extractor takes the same input -- a canonical batch, (B, 3, S, S) float in
[0, 1], no normalization (see data_io.py) -- and returns a fixed-width block of
numbers, (B, dim) float32 numpy. Nothing in an extractor is trained; they are all
frozen, so a whole dataset can be turned into a feature matrix exactly once and
then reused across every classifier experiment.

`feature_names()` is required rather than optional: with three heterogeneous
blocks concatenated into one vector, a debuggable pipeline needs to be able to
say which column is which (used by eval/ablation.py and by XGBoost's importance
output).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F


class FeatureExtractor(ABC):
    """Frozen image -> fixed-width feature block."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def __call__(self, canonical: torch.Tensor) -> np.ndarray:
        """canonical: (B, 3, S, S) float in [0, 1]. Returns (B, dim) float32."""

    def feature_names(self) -> list[str]:
        """Per-column names; defaults to `<name>_<i>` for the embedding blocks,
        whose individual dimensions have no human-readable meaning anyway."""
        return [f"{self.name}_{i}" for i in range(self.dim)]

    def signature(self) -> dict:
        """Config echo stored in the feature cache and in the checkpoint, so
        infer.py can rebuild byte-identical extractors and so a stale cache is
        detectable rather than silently mixed with fresh features."""
        return {"name": self.name, "dim": self.dim}


def resize_and_normalize(
    canonical: torch.Tensor, size: int, mean: list[float], std: list[float]
) -> torch.Tensor:
    """Canonical [0,1] batch -> a backbone's expected resolution + normalization.

    `antialias=True` matters here: without it, downsampling introduces its own
    aliasing pattern, which is exactly the kind of high-frequency artifact this
    detector is supposed to be reading off the *generator*, not off our own
    preprocessing.
    """
    if canonical.shape[-1] != size or canonical.shape[-2] != size:
        canonical = F.interpolate(
            canonical, size=(size, size), mode="bilinear", align_corners=False, antialias=True
        )
    mean_t = torch.tensor(mean, dtype=canonical.dtype, device=canonical.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=canonical.dtype, device=canonical.device).view(1, 3, 1, 1)
    return (canonical - mean_t) / std_t
