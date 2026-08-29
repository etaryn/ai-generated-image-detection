"""Step 2: the small classifier that turns the feature vector into P(AI-generated).

Two interchangeable options, selected by `classifier.type` in the config:

    "mlp"     -- a 2-hidden-layer torch MLP. Handles the dense, correlated
                 embedding blocks well and is what to reach for when DINOv2/CLIP
                 features dominate the vector.
    "xgboost" -- gradient-boosted trees. Usually the stronger choice when the FFT
                 block is carrying the signal, since those are heterogeneous
                 hand-built statistics where axis-aligned splits and monotone
                 thresholds are exactly the right inductive bias -- and it gives
                 per-feature importances for free.

Both are trained on the cached feature matrix, so switching between them is a
config change and a few minutes, not a re-extraction.

`load_predictor` gives infer.py and the eval scripts one calling convention
(`predict(X) -> P(fake)`) regardless of which was trained.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def load_predictor(bundle: dict) -> Callable[[np.ndarray], np.ndarray]:
    """Rebuild a `predict(X) -> probabilities` callable from a saved checkpoint."""
    classifier_type = bundle["classifier_type"]
    if classifier_type == "mlp":
        from classifiers.mlp import load_mlp_predictor

        return load_mlp_predictor(bundle)
    if classifier_type == "xgboost":
        from classifiers.xgb import load_xgb_predictor

        return load_xgb_predictor(bundle)
    raise ValueError(f"Unknown classifier_type {classifier_type!r} in checkpoint")
