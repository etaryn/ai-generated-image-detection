"""Calibration: monotone, actually calibrating, and refusing bad fits.

The property that matters most is monotonicity -- calibration must rescale
patch scores without reordering them, or the map's ranking changes and every
threshold downstream means something different. The Platt fit refuses a negative
slope outright rather than silently inverting the map.

Torch-free.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapper.calibration import (  # noqa: E402
    Calibrator,
    ScaleCalibrators,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
)


def _overconfident_sample(n: int = 2000, seed: int = 0):
    """Scores that are ranked correctly but far too extreme -- the real failure mode."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    latent = rng.normal(loc=np.where(labels > 0, 0.8, -0.8), scale=1.0)
    scores = 1.0 / (1.0 + np.exp(-4.0 * latent))  # temperature 4 == wildly overconfident
    return scores, labels


def test_identity_is_a_no_op_and_says_it_is_unfitted():
    cal = Calibrator.identity()
    assert cal.fitted is False
    scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(cal.apply(scores), scores, atol=1e-6)


def test_platt_improves_calibration():
    scores, labels = _overconfident_sample()
    cal = fit_platt(scores[:1400], labels[:1400])
    before = expected_calibration_error(scores[1400:], labels[1400:])
    after = expected_calibration_error(cal.apply(scores[1400:]), labels[1400:])
    assert cal.fitted
    assert after < before, f"ECE got worse: {before:.4f} -> {after:.4f}"


def test_platt_is_monotone():
    scores, labels = _overconfident_sample()
    cal = fit_platt(scores, labels)
    grid = np.linspace(0.001, 0.999, 200)
    mapped = cal.apply(grid)
    assert np.all(np.diff(mapped) >= -1e-9), "calibration reordered patches"


def test_platt_rejects_inverted_labels():
    scores, labels = _overconfident_sample()
    try:
        fit_platt(scores, 1.0 - labels)  # deliberately mislabelled
    except ValueError as exc:
        assert "monotone" in str(exc)
        return
    raise AssertionError("an inverted fit should raise rather than flip the map")


def test_isotonic_is_monotone_and_fits():
    scores, labels = _overconfident_sample()
    cal = fit_isotonic(scores, labels)
    grid = np.linspace(0.001, 0.999, 200)
    mapped = cal.apply(grid)
    assert np.all(np.diff(mapped) >= -1e-9)
    assert expected_calibration_error(cal.apply(scores), labels) < expected_calibration_error(scores, labels)


def test_nan_passes_through():
    cal = Calibrator.identity()
    out = cal.apply(np.array([0.5, np.nan, 0.9]))
    assert np.isnan(out[1]) and not np.isnan(out[0])


def test_round_trips_through_json():
    scores, labels = _overconfident_sample()
    cal = fit_platt(scores, labels)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.json"
        cal.save(path)
        loaded = Calibrator.load(path)
    grid = np.linspace(0.01, 0.99, 50)
    assert np.allclose(cal.apply(grid), loaded.apply(grid), atol=1e-9)
    assert loaded.fitted


def test_missing_calibration_file_is_a_clear_error():
    try:
        Calibrator.load(Path("definitely") / "not" / "here.json")
    except FileNotFoundError as exc:
        assert "calibrate_mapper" in str(exc)
        return
    raise AssertionError("a missing calibration file should raise")


def test_too_few_samples_refuses_rather_than_fitting_noise():
    try:
        fit_platt([0.1, 0.9], [0.0, 1.0])
    except ValueError:
        return
    raise AssertionError("fitting on 2 points should raise")


def test_scale_calibrators_apply_the_right_fit_per_scale():
    """The point of the per-scale design: 64px and 224px get different maps.

    Measured on SID-Set, 36.6% of 64px patches from *authentic* photographs
    clear the 0.75 threshold against 10.4% at 224px, so one shared map cannot
    correct both. Here the fine scale is given a deliberately harsher fit and
    the same raw score must come out lower at 64px than at 224px.
    """
    harsh = Calibrator(kind="platt", params={"a": 1.0, "b": -2.0}, fitted=True)
    gentle = Calibrator(kind="platt", params={"a": 1.0, "b": 0.0}, fitted=True)
    cal = ScaleCalibrators({64: harsh, 224: gentle})

    raw = np.array([0.8])
    assert cal.apply(raw, scale=64)[0] < cal.apply(raw, scale=224)[0]
    assert np.isclose(cal.apply(raw, scale=224)[0], 0.8, atol=1e-6)


def test_unfitted_scale_falls_back_to_the_shared_map():
    shared = Calibrator(kind="platt", params={"a": 1.0, "b": -1.0}, fitted=True)
    cal = ScaleCalibrators({64: Calibrator.identity()}, shared=shared)
    raw = np.array([0.7])
    # 96 was never fitted, so it must get the shared map, not identity.
    assert np.isclose(cal.apply(raw, scale=96)[0], shared.apply(raw)[0], atol=1e-9)


def test_scale_calibrators_report_unfitted_state():
    assert ScaleCalibrators().fitted is False
    assert ScaleCalibrators({64: Calibrator(kind="platt", params={"a": 1, "b": 0}, fitted=True)}).fitted


def test_a_single_calibrator_file_still_loads():
    """Older single-map calibration files must not break -- they become the shared map."""
    single = fit_platt(*_overconfident_sample())
    cal = ScaleCalibrators.from_dict(single.to_dict())
    assert cal.fitted and not cal.per_scale
    grid = np.linspace(0.01, 0.99, 20)
    assert np.allclose(cal.apply(grid, scale=64), single.apply(grid), atol=1e-9)


def test_scale_calibrators_round_trip_through_json():
    scores, labels = _overconfident_sample()
    cal = ScaleCalibrators(
        {64: fit_platt(scores, labels), 224: fit_isotonic(scores, labels)},
        shared=fit_platt(scores, labels),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.json"
        cal.save(path)
        loaded = ScaleCalibrators.load(path)
    grid = np.linspace(0.01, 0.99, 50)
    for scale in (64, 224, 999):
        assert np.allclose(cal.apply(grid, scale=scale), loaded.apply(grid, scale=scale), atol=1e-9)
    assert sorted(loaded.per_scale) == [64, 224]


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} calibration tests passed")


if __name__ == "__main__":
    run()
