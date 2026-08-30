"""Blending: correct averages, no square artefacts, sane multi-scale combination.

The blocking test is the one that matters. Flat-painted patches produce a
heatmap made of rectangles, region extraction then finds the rectangles rather
than the tampered object, and every downstream descriptor reads the grid. The
test asserts that a step in patch scores becomes a *smooth* ramp in the map, and
that no single-pixel jumps survive at the patch stride.

Torch-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapper.blend import blend_scale, combine_scales, hann2d  # noqa: E402
from mapper.windows import plan_windows  # noqa: E402


def test_constant_scores_give_a_constant_map():
    plan = plan_windows(256, 256, [128], overlap=0.5)
    windows = plan[128]
    heat, support = blend_scale(256, 256, windows, [0.7] * len(windows))
    assert np.allclose(heat, 0.7, atol=1e-6), f"range {heat.min()}..{heat.max()}"
    assert support.max() <= 1.0 + 1e-6
    assert support.min() > 0.0


def test_kernel_peaks_at_the_centre():
    k = hann2d(64, 64)
    assert k[32, 32] == k.max()
    assert k[0, 0] < 0.05 * k.max()
    assert k.min() > 0.0, "a zero kernel would make a singly-covered pixel 0/0"


def test_no_blocking_at_the_patch_stride():
    # Left half of the image scored 0.1, right half 0.9. A flat-paint blend puts
    # a hard step at a patch edge; a centre-weighted blend puts a smooth ramp.
    plan = plan_windows(512, 256, [128], overlap=0.5)
    windows = plan[128]
    scores = [0.9 if (w.x0 + w.x1) / 2 > 256 else 0.1 for w in windows]
    heat, _ = blend_scale(256, 512, windows, scores)

    row = heat[128]
    jumps = np.abs(np.diff(row))
    assert jumps.max() < 0.10, f"hard step of {jumps.max():.3f} -- flat-painted patches"
    # The transition must still happen: this is a smoothness test, not a blur test.
    assert row[:64].mean() < 0.2 and row[-64:].mean() > 0.8


def test_blend_rejects_mismatched_inputs():
    plan = plan_windows(128, 128, [64], overlap=0.5)
    try:
        blend_scale(128, 128, plan[64], [0.5])
    except ValueError:
        return
    raise AssertionError("mismatched windows/scores should raise")


def test_combine_scales_defaults_to_max():
    """The default must not average a coarse scale's structurally-low reading in.

    A coarse window covering an edit plus authentic pixels reports their mean by
    construction, so averaging the scales erases evidence the fine scale found.
    See combine_scales' docstring for the measured version of this.
    """
    fine = np.full((8, 8), 0.86, dtype=np.float32)
    coarse = np.full((8, 8), 0.48, dtype=np.float32)
    combined = combine_scales({128: fine, 224: coarse})
    assert np.allclose(combined, 0.86, atol=1e-6), "the fine scale's evidence was diluted"


def test_combine_scales_mean_still_available_for_ablation():
    a = np.full((8, 8), 0.2, dtype=np.float32)
    b = np.full((8, 8), 0.8, dtype=np.float32)
    assert np.allclose(combine_scales({128: a, 224: b}, method="mean"), 0.5, atol=1e-6)


def test_combine_scales_tolerates_nan_under_both_methods():
    a = np.full((8, 8), 0.2, dtype=np.float32)
    b = np.full((8, 8), 0.8, dtype=np.float32)
    b_holed = b.copy()
    b_holed[0, 0] = np.nan  # uncovered at the coarse scale only

    for method in ("max", "mean"):
        combined = combine_scales({128: a, 224: b_holed}, method=method)
        assert np.isclose(combined[0, 0], 0.2, atol=1e-6), method
        assert not np.isnan(combined).any(), method

    # Uncovered at *every* scale stays NaN rather than inventing a score.
    a_holed = a.copy()
    a_holed[0, 0] = np.nan
    assert np.isnan(combine_scales({128: a_holed, 224: b_holed})[0, 0])


def test_combine_scales_respects_weights_in_mean_mode():
    a = np.full((4, 4), 0.0, dtype=np.float32)
    b = np.full((4, 4), 1.0, dtype=np.float32)
    combined = combine_scales({128: a, 224: b}, weights={128: 3.0, 224: 1.0}, method="mean")
    assert np.allclose(combined, 0.25, atol=1e-6)


def test_combine_scales_rejects_an_unknown_method():
    a = np.full((4, 4), 0.5, dtype=np.float32)
    try:
        combine_scales({128: a}, method="median")
    except ValueError:
        return
    raise AssertionError("an unknown combination method should raise")


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} blending tests passed")


if __name__ == "__main__":
    run()
