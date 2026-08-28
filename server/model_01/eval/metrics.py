"""Metric functions shared by robustness_eval.py and error_analysis.py.

`fpr_at_threshold` is called out separately because false positives (real content
flagged as AI-generated) are the costlier error for a moderation use case — the
challenge brief explicitly names false positives as a trade-off to discuss.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_all_metrics(labels: np.ndarray, preds: np.ndarray, threshold: float = 0.5) -> dict:
    """labels: 0/1 ground truth (1 = AI-generated). preds: predicted P(fake) in [0,1]."""
    pred_labels = (preds >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, pred_labels),
        "balanced_accuracy": balanced_accuracy_score(labels, pred_labels),
        "f1": f1_score(labels, pred_labels, zero_division=0),
    }
    # AUC requires both classes present in this slice.
    if len(np.unique(labels)) > 1:
        metrics["auc"] = roc_auc_score(labels, preds)
    else:
        metrics["auc"] = float("nan")
    metrics["fpr_at_threshold"] = fpr_at_threshold(labels, preds, threshold)
    return metrics


def fpr_at_threshold(labels: np.ndarray, preds: np.ndarray, threshold: float = 0.5) -> float:
    """False-positive rate: fraction of real (label=0) images predicted as fake."""
    real_mask = labels == 0
    if real_mask.sum() == 0:
        return float("nan")
    false_positives = ((preds[real_mask] >= threshold).astype(int)).sum()
    return float(false_positives / real_mask.sum())


def threshold_for_target_fpr(labels: np.ndarray, preds: np.ndarray, target_fpr: float = 0.05) -> float:
    """Find the decision threshold that achieves (at most) `target_fpr` on this data.

    Useful for calibrating the operating point against a false-positive budget
    rather than using a fixed 0.5 cutoff (see README's "limitations" section).
    """
    fprs, tprs, thresholds = roc_curve(labels, preds)
    valid = fprs <= target_fpr
    if not valid.any():
        return float(thresholds[np.argmin(fprs)])
    # Among thresholds meeting the FPR budget, pick the one with highest TPR.
    best_idx = np.argmax(np.where(valid, tprs, -1))
    return float(thresholds[best_idx])
