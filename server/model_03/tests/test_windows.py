"""Window geometry: full coverage, edge clamping, degenerate sizes.

Coverage is the property worth testing hardest. An uncovered strip along the
right or bottom edge is silent -- the map just reads low there -- and it is
exactly where a generative-expansion edit lives.

Torch-free; runs anywhere numpy is installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapper.windows import Window, count_windows, effective_scales, plan_windows  # noqa: E402


def _coverage(width: int, height: int, windows: list[Window]) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.int32)
    for w in windows:
        canvas[w.y0 : w.y1, w.x0 : w.x1] += 1
    return canvas


def test_every_pixel_is_covered_at_every_scale():
    for width, height in [(640, 480), (501, 333), (128, 128), (1024, 97)]:
        plan = plan_windows(width, height, [128, 224], overlap=0.5)
        assert plan, f"no windows for {width}x{height}"
        for scale, windows in plan.items():
            cov = _coverage(width, height, windows)
            assert cov.min() >= 1, (
                f"{width}x{height} scale {scale}: {(cov == 0).sum()} uncovered pixels"
            )


def test_windows_stay_inside_the_frame():
    plan = plan_windows(333, 217, [128, 224], overlap=0.5)
    for windows in plan.values():
        for w in windows:
            assert 0 <= w.x0 < w.x1 <= 333
            assert 0 <= w.y0 < w.y1 <= 217


def test_last_window_is_clamped_to_the_edge():
    # 300 wide, 128 patch, 64 stride: naive range() stops at 128, leaving 44px
    # of the right edge unseen. The clamped plan must reach x1 == 300.
    plan = plan_windows(300, 300, [128], overlap=0.5)
    assert max(w.x1 for w in plan[128]) == 300
    assert max(w.y1 for w in plan[128]) == 300


def test_scales_larger_than_the_image_are_clamped_not_dropped():
    # A 96px thumbnail still gets a map rather than an empty plan.
    plan = plan_windows(96, 96, [128, 224], overlap=0.5)
    assert len(plan) == 1, "128 and 224 both clamp to 96 and must dedupe to one scale"
    assert count_windows(plan) == 1
    assert effective_scales(96, 96, [128, 224]) == [96]


def test_overlap_controls_stride():
    dense = plan_windows(512, 512, [128], overlap=0.75)
    sparse = plan_windows(512, 512, [128], overlap=0.0)
    assert len(dense[128]) > len(sparse[128])
    # With zero overlap and an exact fit, tiling is 4x4 with no duplicates.
    assert len(sparse[128]) == 16


def test_rejects_nonsense_arguments():
    for bad in (1.0, -0.1, 1.5):
        try:
            plan_windows(100, 100, [32], overlap=bad)
        except ValueError:
            continue
        raise AssertionError(f"overlap={bad} should have raised")


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} window tests passed")


if __name__ == "__main__":
    run()
