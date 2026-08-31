"""Tests for native-resolution canonicalization and multi-cache merging.

Both cover quiet failures -- the kind that produce a *better* number rather than
an error, and so never get investigated:

  - `CanonicalCrop` resampling an image it didn't have to. Rescaling writes
    directly into the frequency band `features/fft.py` reads, so a silent resize
    turns the spectral columns into a description of our own interpolation. The
    tests below assert that a large-enough image reaches the extractors with its
    pixels bit-for-bit untouched, and that the crop keeps JPEG grid phase.
  - `load_caches` stacking group ids from different caches without offsetting
    them. Ids are only unique within one cache, so raw stacking fuses unrelated
    source images into a single group and `group_split` then splits a corrupted
    grouping -- reintroducing exactly the near-duplicate leakage it exists to
    prevent, while reporting a higher val score for it.

The crop tests need only Pillow/numpy, so they run even where torch is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from data_io import CanonicalCrop
else:
    # The class is pure Pillow; import it without dragging in data_io's torch
    # dependencies so these tests still run in a bare environment.
    import importlib.util

    _src = (Path(__file__).resolve().parents[1] / "data_io.py").read_text()
    _start = _src.index("class CanonicalCrop")
    _end = _src.index("def canonical_transform")
    _ns: dict = {"Image": Image}
    exec(compile(_src[_start:_end], "data_io.py", "exec"), _ns)
    CanonicalCrop = _ns["CanonicalCrop"]


def _noise_image(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_large_image_is_cropped_not_resampled():
    """A 1024x1024 source must reach the extractors with untouched pixels."""
    img = _noise_image(1024, 1024, seed=1)
    out = CanonicalCrop(256)(img)
    assert out.size == (256, 256), out.size

    left = ((1024 - 256) // 2) & ~7
    top = ((1024 - 256) // 2) & ~7
    expected = np.asarray(img)[top:top + 256, left:left + 256]
    # Bit-exact: any resampling at all would perturb these values.
    assert np.array_equal(np.asarray(out), expected), "crop was resampled"


def test_crop_offsets_preserve_jpeg_grid_phase():
    """Offsets must be multiples of 8 for every input shape.

    64 of the FFT block's 130 columns are a block-DCT profile keyed to the JPEG
    8x8 grid starting at pixel 0. An off-phase crop smears block boundaries
    across all 64 of them.
    """
    for w, h in [(1024, 1024), (500, 375), (1792, 1024), (257, 257), (263, 300)]:
        img = _noise_image(w, h, seed=2)
        out = CanonicalCrop(256)(img)
        assert out.size == (256, 256), (w, h, out.size)
        left = ((w - 256) // 2) & ~7
        top = ((h - 256) // 2) & ~7
        assert left % 8 == 0 and top % 8 == 0, (w, h, left, top)
        expected = np.asarray(img)[top:top + 256, left:left + 256]
        assert np.array_equal(np.asarray(out), expected), (w, h)


def test_small_image_is_upscaled_then_cropped():
    """Below canonical_size there is no lossless option; resize is the fallback.

    32x32 CIFAKE must still come out at the canonical size so existing caches and
    checkpoints keep working.
    """
    out = CanonicalCrop(256)(_noise_image(32, 32, seed=3))
    assert out.size == (256, 256), out.size

    # Non-square small input: short side is scaled up to reach the target, then
    # the long side is cropped rather than squashed.
    out = CanonicalCrop(256)(_noise_image(100, 60, seed=4))
    assert out.size == (256, 256), out.size


def test_exact_size_image_is_identity():
    """A 256x256 source -- what prepare_generators.py writes -- must be a no-op."""
    img = _noise_image(256, 256, seed=5)
    out = CanonicalCrop(256)(img)
    assert np.array_equal(np.asarray(out), np.asarray(img)), "no-op path resampled"


class Skipped(Exception):
    """Raised by a test that needs a dependency this environment lacks."""


def test_load_caches_offsets_group_ids():
    """Merged caches must not fuse distinct source images into one group."""
    if not HAS_TORCH:
        raise Skipped("needs torch (train.py imports it)")

    import json

    from train import group_split, load_caches

    tmp = Path(__file__).resolve().parent / "_tmp_caches"
    tmp.mkdir(exist_ok=True)
    meta = {"features": {"blocks": [{"name": "fft", "dim": 4, "start": 0, "stop": 4}]},
            "feature_names": [f"f{i}" for i in range(4)]}
    paths = []
    try:
        for k in range(2):
            p = tmp / f"c{k}.npz"
            # Both caches use group ids 0..4 -- the collision this guards against.
            np.savez(
                p,
                X=np.full((5, 4), k, dtype=np.float32),
                y=np.array([0, 1, 0, 1, 0], dtype=np.int64),
                groups=np.arange(5, dtype=np.int64),
                paths=np.array([f"{k}_{i}.jpg" for i in range(5)]),
                aug_flags=np.zeros(5, dtype=np.int64),
                meta=json.dumps(meta),
            )
            paths.append(p)

        merged = load_caches(paths)
        assert merged["X"].shape == (10, 4), merged["X"].shape
        assert len(np.unique(merged["groups"])) == 10, "group ids collided across caches"

        # And the split must respect the merged grouping.
        tr, va = group_split(merged["groups"], 0.5, seed=0)
        assert len(set(merged["groups"][tr]) & set(merged["groups"][va])) == 0
    finally:
        for p in paths:
            p.unlink(missing_ok=True)
        tmp.rmdir()


def test_load_caches_rejects_mismatched_widths():
    """Caches built with different feature configs must fail loudly, not silently."""
    if not HAS_TORCH:
        raise Skipped("needs torch (train.py imports it)")

    import json

    from train import load_caches

    tmp = Path(__file__).resolve().parent / "_tmp_caches_bad"
    tmp.mkdir(exist_ok=True)
    paths = []
    try:
        for k, dim in enumerate((4, 6)):
            p = tmp / f"c{k}.npz"
            meta = {"features": {"blocks": [{"name": "fft", "dim": dim, "start": 0, "stop": dim}]},
                    "feature_names": [f"f{i}" for i in range(dim)]}
            np.savez(
                p,
                X=np.zeros((3, dim), dtype=np.float32),
                y=np.zeros(3, dtype=np.int64),
                groups=np.arange(3, dtype=np.int64),
                paths=np.array([f"{k}_{i}.jpg" for i in range(3)]),
                aug_flags=np.zeros(3, dtype=np.int64),
                meta=json.dumps(meta),
            )
            paths.append(p)

        try:
            load_caches(paths)
        except ValueError as exc:
            assert "feature widths" in str(exc), exc
        else:
            raise AssertionError("mismatched cache widths were silently combined")
    finally:
        for p in paths:
            p.unlink(missing_ok=True)
        tmp.rmdir()


if __name__ == "__main__":
    import traceback

    tests = {n: o for n, o in list(globals().items()) if n.startswith("test_") and callable(o)}
    passed = failed = skipped = 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Skipped as exc:
            print(f"SKIP  {name} ({exc})")
            skipped += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
