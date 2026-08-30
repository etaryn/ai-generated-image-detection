"""The with/without-localisation ablation, on fixtures with known answers.

This script decides whether the project's central claim holds, so the one thing
it must not do is conclude "it works" regardless of input. These tests feed it
constructed results where the right answer is known -- localisation clearly
better, clearly worse, and indistinguishable -- and check it says so.

The paired bootstrap matters too: the two arms score the same images, and
treating them as independent samples would widen the interval enough to hide a
real effect.

Torch-free and network-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ablation import (  # noqa: E402
    analyse,
    baseline_only_capabilities,
    false_positives_at_matched_recall,
    paired_auc_delta,
)


def _rows(with_scores, without_scores, classes, verdicts=None, ious=None):
    rows = []
    for i, (w, wo, cls) in enumerate(zip(with_scores, without_scores, classes)):
        row = {
            "class": cls,
            "score": float(w),
            "whole_image_score": float(wo),
            "verdict": (verdicts or {}).get(i, "ai_edited"),
            "n_regions": 1,
        }
        if ious is not None and cls == "tampered":
            row["localisation"] = {
                "iou": float(ious[i]), "recall": float(ious[i]), "precision": float(ious[i]),
                "f1": float(ious[i]), "pred_area_frac": 0.1, "true_area_frac": 0.1,
            }
        rows.append(row)
    return rows


def _split(n=40):
    """n tampered then n real, as a class list."""
    return ["tampered"] * n + ["real"] * n


def test_detects_a_clear_improvement():
    n = 40
    rng = np.random.default_rng(0)
    # "with" separates the classes; "without" is noise.
    with_scores = np.concatenate([rng.normal(0.9, 0.05, n), rng.normal(0.1, 0.05, n)])
    without = np.concatenate([rng.normal(0.5, 0.2, n), rng.normal(0.5, 0.2, n)])
    result = paired_auc_delta(_rows(with_scores, without, _split(n)), ("tampered",), n_boot=400)

    assert result["auc_with"] > 0.95 and result["delta"] > 0.3
    assert result["significant"], "a large real effect must be called significant"
    assert result["ci95"][0] > 0


def test_detects_a_clear_regression():
    n = 40
    rng = np.random.default_rng(1)
    with_scores = np.concatenate([rng.normal(0.5, 0.2, n), rng.normal(0.5, 0.2, n)])
    without = np.concatenate([rng.normal(0.9, 0.05, n), rng.normal(0.1, 0.05, n)])
    result = paired_auc_delta(_rows(with_scores, without, _split(n)), ("tampered",), n_boot=400)

    assert result["delta"] < -0.3
    assert result["significant"] and result["ci95"][1] < 0


def test_calls_a_non_difference_insignificant():
    """Identical arms must not be reported as a win in either direction."""
    n = 60
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(0.7, 0.2, n), rng.normal(0.3, 0.2, n)])
    result = paired_auc_delta(_rows(scores, scores.copy(), _split(n)), ("tampered",), n_boot=400)

    assert abs(result["delta"]) < 1e-9
    assert not result["significant"], "identical arms cannot be a significant difference"


def test_matched_operating_point_compares_like_with_like():
    """The arms are on different scales; the comparison must survive that."""
    n = 50
    rng = np.random.default_rng(3)
    # Same ranking, but "with" is shifted an order of magnitude higher.
    base_ai = rng.uniform(0.4, 0.6, n)
    base_real = rng.uniform(0.0, 0.4, n)
    rows = _rows(
        np.concatenate([base_ai * 100, base_real * 100]),
        np.concatenate([base_ai, base_real]),
        ["tampered"] * n + ["real"] * n,
    )
    matched = false_positives_at_matched_recall(rows, target_recall=0.8)

    assert matched["with"]["recall_on_ai"] >= 0.79
    assert matched["without"]["recall_on_ai"] >= 0.79
    # Identical ranking under a monotone rescale => identical false-positive rate.
    assert abs(matched["fpr_delta"]) < 1e-9


def test_capabilities_are_reported_separately_from_detection():
    rows = _rows(
        [0.9, 0.9, 0.1], [0.9, 0.9, 0.1],
        ["tampered", "synthetic", "real"],
        verdicts={0: "ai_edited", 1: "ai_generated", 2: "likely_authentic"},
        ious=[0.6, 0.0, 0.0],
    )
    caps = baseline_only_capabilities(rows)
    assert caps["localisation"]["mean_iou"] == 0.6
    assert caps["kind_discrimination"]["tampered_called_ai_edited"] == 1.0
    assert caps["kind_discrimination"]["synthetic_called_ai_generated"] == 1.0
    assert "undefined" in caps["kind_discrimination"]["note"]


def test_overall_verdict_says_it_does_not_work_when_it_does_not():
    """No detection gain and poor localisation must read as a negative result."""
    n = 30
    rng = np.random.default_rng(4)
    scores = np.concatenate([rng.normal(0.5, 0.2, n), rng.normal(0.5, 0.2, n)])
    rows = _rows(scores, scores.copy(), _split(n), ious=[0.0] * (2 * n))
    payload = {"per_image": rows, "backend": {"backend": "stub"}, "n_images": len(rows)}

    conclusions = " ".join(analyse(payload)["conclusions"])
    assert "does NOT yet work" in conclusions


def test_overall_verdict_credits_localisation_without_detection_gain():
    """The realistic middle case: same score, but it finds where the edit is."""
    n = 30
    rng = np.random.default_rng(5)
    scores = np.concatenate([rng.normal(0.7, 0.2, n), rng.normal(0.3, 0.2, n)])
    rows = _rows(scores, scores.copy(), _split(n), ious=[0.5] * (2 * n))
    payload = {"per_image": rows, "backend": {"backend": "stub"}, "n_images": len(rows)}

    conclusions = " ".join(analyse(payload)["conclusions"])
    assert "earns its place on localisation" in conclusions
    assert "does NOT yet work" not in conclusions


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} ablation tests passed")


if __name__ == "__main__":
    run()
