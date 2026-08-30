"""The evaluation metrics themselves, checked against cases with known answers.

Every claim model_03 makes about its own accuracy is one of these two functions
applied to pipeline output. A quietly wrong AUC or IoU would not crash anything;
it would produce a benchmark table that reads plausibly and is false, which is
worse than having no table. Both are reimplemented here rather than taken from
sklearn (which is not a dependency of this project), so they need their own
checks.

The tie handling in `auc` is the part most likely to be wrong and most likely to
matter: these detectors saturate hard, so identical scores are common, and
counting ties as wins would flatter every result.

Torch-free and network-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.evaluate import auc, mask_metrics  # noqa: E402


def test_perfect_separation_is_one():
    assert auc(np.array([0.9, 0.8, 0.7]), np.array([0.3, 0.2, 0.1])) == 1.0


def test_perfect_inversion_is_zero():
    assert auc(np.array([0.1, 0.2, 0.3]), np.array([0.7, 0.8, 0.9])) == 0.0


def test_all_ties_is_one_half():
    """A saturated detector scoring everything 1.0 has no skill, and must read as none."""
    assert auc(np.full(20, 1.0), np.full(20, 1.0)) == 0.5


def test_partial_ties_are_counted_as_half():
    # One positive above, one tied with the single negative.
    assert auc(np.array([0.9, 0.5]), np.array([0.5])) == 0.75


def test_matches_a_brute_force_definition_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(5):
        pos = rng.normal(1.0, 1.0, 60)
        neg = rng.normal(0.0, 1.0, 40)
        wins = sum(
            1.0 if p > n else 0.5 if p == n else 0.0
            for p in pos
            for n in neg
        )
        assert abs(auc(pos, neg) - wins / (pos.size * neg.size)) < 1e-9


def test_auc_of_empty_input_is_nan():
    assert np.isnan(auc(np.array([]), np.array([1.0])))


def _box(shape, y0, y1, x0, x1):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_identical_masks_score_one():
    mask = _box((100, 100), 20, 60, 20, 60)
    m = mask_metrics(mask, mask)
    assert m["iou"] == 1.0 and m["f1"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0


def test_disjoint_masks_score_zero():
    m = mask_metrics(_box((100, 100), 0, 20, 0, 20), _box((100, 100), 60, 80, 60, 80))
    assert m["iou"] == 0.0 and m["f1"] == 0.0


def test_half_overlap_has_known_iou():
    # Two 40x40 boxes sharing a 20x40 strip: intersection 800, union 2400.
    pred = _box((100, 100), 20, 60, 20, 60)
    truth = _box((100, 100), 20, 60, 40, 80)
    m = mask_metrics(pred, truth)
    assert abs(m["iou"] - 800 / 2400) < 1e-9
    assert abs(m["precision"] - 0.5) < 1e-9
    assert abs(m["recall"] - 0.5) < 1e-9


def test_over_prediction_costs_precision_not_recall():
    """Flagging the whole frame finds every edit and is still a failure."""
    truth = _box((100, 100), 40, 60, 40, 60)
    m = mask_metrics(np.ones((100, 100), dtype=bool), truth)
    assert m["recall"] == 1.0
    assert m["precision"] == 0.04
    assert m["iou"] == 0.04


def test_empty_prediction_is_zero_not_a_crash():
    m = mask_metrics(np.zeros((50, 50), dtype=bool), _box((50, 50), 10, 20, 10, 20))
    assert m["iou"] == 0.0 and m["precision"] == 0.0 and m["recall"] == 0.0


def test_area_fractions_are_reported():
    m = mask_metrics(_box((100, 100), 0, 10, 0, 100), _box((100, 100), 0, 20, 0, 100))
    assert abs(m["pred_area_frac"] - 0.10) < 1e-9
    assert abs(m["true_area_frac"] - 0.20) < 1e-9


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} evaluation-metric tests passed")


if __name__ == "__main__":
    run()
