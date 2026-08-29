"""Tests for the feature stack's bookkeeping and the train/val split.

The failure modes these cover are quiet ones: a block spec that doesn't line up
with the columns it describes (so an ablation silently trains on the wrong
features), and a train/val split that lets augmented copies of the same image
land on both sides (so the val score is inflated by near-duplicate leakage and
nobody notices until the held-out demo set says otherwise).

Only the FFT extractor is exercised here -- DINOv2 and CLIP would need weight
downloads, and what's being tested is the plumbing, not the backbones.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from extract_features import build_rows
    from features.pipeline import FeatureStack
    from train import group_split, select_blocks

FFT_ONLY_CFG = {
    "dinov2": {"enabled": False},
    "clip": {"enabled": False},
    "fft": {"enabled": True, "work_size": 64, "n_radial_bins": 8, "n_angular_bins": 4, "n_threads": 1},
}


def test_stack_dim_blocks_and_names_agree():
    stack = FeatureStack.from_config(FFT_ONLY_CFG)
    assert len(stack.blocks) == 1
    block = stack.blocks[0]
    assert (block.name, block.start, block.stop) == ("fft", 0, stack.dim)
    assert len(stack.feature_names()) == stack.dim
    feats = stack(torch.rand(3, 3, 64, 64))
    assert feats.shape == (3, stack.dim)


def test_empty_stack_is_rejected():
    try:
        FeatureStack.from_config({"dinov2": {"enabled": False}, "clip": {"enabled": False}, "fft": {"enabled": False}})
    except ValueError:
        return
    raise AssertionError("a stack with every extractor disabled should raise, not produce 0 features")


def test_select_blocks_slices_the_right_columns():
    block_spec = [
        {"name": "dino", "dim": 4, "start": 0, "stop": 4},
        {"name": "clip", "dim": 3, "start": 4, "stop": 7},
        {"name": "fft", "dim": 2, "start": 7, "stop": 9},
    ]
    X = np.arange(2 * 9, dtype=np.float32).reshape(2, 9)

    X_sub, kept, cols = select_blocks(X, block_spec, ["fft"])
    assert X_sub.shape == (2, 2)
    assert np.array_equal(X_sub, X[:, 7:9])
    assert kept == [{"name": "fft", "dim": 2, "start": 0, "stop": 2}], "kept blocks must be renumbered"
    assert np.array_equal(cols, [7, 8])

    # Two blocks: columns concatenate in block-spec order, not argument order.
    X_sub, kept, cols = select_blocks(X, block_spec, ["fft", "dino"])
    assert np.array_equal(cols, [0, 1, 2, 3, 7, 8])
    assert [b["name"] for b in kept] == ["dino", "fft"]

    # No selection = everything, untouched.
    X_all, kept_all, cols_all = select_blocks(X, block_spec, None)
    assert X_all.shape == X.shape and kept_all == block_spec and len(cols_all) == 9


def test_select_blocks_rejects_unknown_name():
    block_spec = [{"name": "fft", "dim": 2, "start": 0, "stop": 2}]
    try:
        select_blocks(np.zeros((1, 2), dtype=np.float32), block_spec, ["dino"])
    except ValueError:
        return
    raise AssertionError("selecting a block the cache doesn't contain should raise")


def test_build_rows_assigns_one_group_per_source_image():
    samples = [(Path("a.jpg"), 0), (Path("b.jpg"), 1)]
    rows, groups, flags = build_rows(samples, aug_copies=2, seed=0)
    assert len(rows) == len(groups) == len(flags) == 6  # 2 images x (1 clean + 2 augmented)
    assert groups == [0, 0, 0, 1, 1, 1]
    assert flags == [False, True, True, False, True, True]
    assert [r[0].name for r in rows] == ["a.jpg"] * 3 + ["b.jpg"] * 3


def test_group_split_keeps_augmented_copies_on_one_side():
    """The leakage guard: every row sharing a group id must end up in the same split."""
    groups = np.repeat(np.arange(50), 3)  # 50 images x 3 copies
    train_idx, val_idx = group_split(groups, val_fraction=0.2, seed=42)

    assert len(train_idx) + len(val_idx) == len(groups)
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    assert not (train_groups & val_groups), "a source image appeared in both splits"
    assert len(val_groups) == 10, f"expected 20% of 50 groups in val, got {len(val_groups)}"


def test_group_split_is_deterministic():
    groups = np.repeat(np.arange(30), 2)
    a = group_split(groups, 0.2, seed=7)
    b = group_split(groups, 0.2, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


if __name__ == "__main__":
    import traceback

    if not HAS_TORCH:
        print("torch is not installed in this environment -- skipping pipeline tests.")
        sys.exit(0)

    tests = {n: o for n, o in list(globals().items()) if n.startswith("test_") and callable(o)}
    passed, failed = 0, 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
