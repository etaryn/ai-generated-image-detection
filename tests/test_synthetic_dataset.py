"""Tests for data/make_synthetic_dataset.py -- torch-free (PIL/numpy only), so it
runs anywhere, including environments without network access to install torch.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.make_synthetic_dataset import generate_dataset  # noqa: E402


def test_generate_dataset_creates_expected_layout():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "synthetic"
        generate_dataset(out_dir, n_per_class=4, size=64, seed=0)

        real_files = sorted((out_dir / "real").glob("*.png"))
        fake_files = sorted((out_dir / "fake").glob("*.png"))
        assert len(real_files) == 4, f"expected 4 real images, got {len(real_files)}"
        assert len(fake_files) == 4, f"expected 4 fake images, got {len(fake_files)}"


def test_generated_images_are_valid_rgb_with_variance():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "synthetic"
        generate_dataset(out_dir, n_per_class=2, size=64, seed=1)

        for label in ("real", "fake"):
            for path in (out_dir / label).glob("*.png"):
                img = Image.open(path)
                assert img.mode == "RGB"
                assert img.size == (64, 64)
                arr = np.asarray(img)
                assert arr.min() >= 0 and arr.max() <= 255
                assert arr.std() > 1.0, f"{path} looks flat/degenerate"


def test_real_and_fake_share_base_but_differ():
    """fake_i should be a perturbed version of real_i (same underlying base
    image), not an unrelated random image -- verifies the artifact is actually
    additive rather than the generator accidentally producing two independent
    scenes."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "synthetic"
        generate_dataset(out_dir, n_per_class=3, size=64, seed=2)

        for i in range(3):
            real = np.asarray(Image.open(out_dir / "real" / f"real_{i:04d}.png"), dtype=np.float32)
            fake = np.asarray(Image.open(out_dir / "fake" / f"fake_{i:04d}.png"), dtype=np.float32)
            diff = np.abs(real - fake)
            # Subtle but nonzero and bounded -- not identical, not a different image.
            assert 0.1 < diff.mean() < 20.0, f"unexpected real/fake difference magnitude: {diff.mean()}"


def test_is_reproducible_with_same_seed():
    with tempfile.TemporaryDirectory() as tmp:
        out_a = Path(tmp) / "a"
        out_b = Path(tmp) / "b"
        generate_dataset(out_a, n_per_class=2, size=32, seed=42)
        generate_dataset(out_b, n_per_class=2, size=32, seed=42)
        a = np.asarray(Image.open(out_a / "real" / "real_0000.png"))
        b = np.asarray(Image.open(out_b / "real" / "real_0000.png"))
        assert np.array_equal(a, b), "same seed should produce identical output"


if __name__ == "__main__":
    import traceback

    tests = {name: obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)}
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
