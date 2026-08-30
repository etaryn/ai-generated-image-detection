"""Connected components, morphology, and region descriptors.

The union-find here replaces `scipy.ndimage.label`, so it is worth testing
against the cases that actually differ: diagonal bridges (8- vs 4-connectivity),
labels that merge only late in the scan (a U shape, where the two arms get
different provisional labels and are joined at the bottom), and dense
renumbering.

Torch-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regions.components import (  # noqa: E402
    _label_components_numpy,
    boundary,
    close_mask,
    dilate,
    erode,
    label_components,
    ring,
)


def test_labels_two_separate_blobs():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True
    mask[12:18, 12:18] = True
    labels, count = label_components(mask)
    assert count == 2
    assert set(np.unique(labels)) == {0, 1, 2}
    assert (labels[2:6, 2:6] == labels[2, 2]).all()


def test_diagonal_bridge_depends_on_connectivity():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    mask[4:6, 4:6] = True  # touches the first block only at a corner
    assert label_components(mask, connectivity=8)[1] == 1
    assert label_components(mask, connectivity=4)[1] == 2


def test_u_shape_merges_across_the_scan():
    # Two arms labelled separately on the way down, joined at the bottom row --
    # the case a single-pass labeller gets wrong.
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 2:4] = True
    mask[2:10, 8:10] = True
    mask[8:10, 2:10] = True
    labels, count = label_components(mask)
    assert count == 1, f"U shape split into {count} components"
    assert len(np.unique(labels[mask])) == 1


def test_empty_mask_has_no_components():
    labels, count = label_components(np.zeros((8, 8), dtype=bool))
    assert count == 0
    assert labels.max() == 0


def test_labels_are_densely_numbered():
    mask = np.zeros((30, 30), dtype=bool)
    for i in range(5):
        mask[i * 6 : i * 6 + 3, i * 6 : i * 6 + 3] = True
    labels, count = label_components(mask)
    assert sorted(np.unique(labels[mask]).tolist()) == list(range(1, count + 1))


def test_dilate_erode_are_duals():
    mask = np.zeros((16, 16), dtype=bool)
    mask[5:11, 5:11] = True
    assert dilate(mask, 1).sum() > mask.sum()
    assert erode(mask, 1).sum() < mask.sum()
    # Closing a solid block leaves it unchanged...
    assert (close_mask(mask, 1) == mask).all()
    # ...and fills a pinhole in it.
    holed = mask.copy()
    holed[8, 8] = False
    assert close_mask(holed, 1)[8, 8]


def test_scipy_and_numpy_labellers_agree():
    """The fast path and the dependency-free fallback must not diverge.

    Compared up to relabelling: both are dense 1..n, but nothing guarantees they
    number the components in the same order.
    """
    rng = np.random.default_rng(7)
    mask = rng.random((64, 64)) > 0.55
    for connectivity in (4, 8):
        fast_labels, fast_count = label_components(mask, connectivity)
        slow_labels, slow_count = _label_components_numpy(mask, connectivity)
        assert fast_count == slow_count, f"{fast_count} vs {slow_count} components"
        # Same partition: each fast label maps onto exactly one slow label.
        for value in np.unique(fast_labels[fast_labels > 0]):
            assert len(np.unique(slow_labels[fast_labels == value])) == 1


def test_boundary_and_ring_are_disjoint_and_adjacent():
    mask = np.zeros((20, 20), dtype=bool)
    mask[6:14, 6:14] = True
    rim = boundary(mask)
    band = ring(mask, 3)
    assert (rim & ~mask).sum() == 0, "boundary is the inner rim"
    assert (band & mask).sum() == 0, "ring is strictly outside"
    assert rim.sum() == 8 * 8 - 6 * 6
    assert band.sum() > 0


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} region tests passed")


if __name__ == "__main__":
    run()
