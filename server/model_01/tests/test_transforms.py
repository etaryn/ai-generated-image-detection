"""Tests for data/transforms.py -- the core of the "robust under transform" story.

Deliberately torch-free: these only need PIL/numpy, which are far more likely to
be available in any environment (including this sandbox) than a full torch
install, so they can catch pipeline bugs even before a training environment is
set up. Written pytest-style (`test_*` functions, plain `assert`) so `pytest
tests/` picks them up once pytest is available, but also runnable directly with
`python3 tests/test_transforms.py` (see the __main__ block) with no pytest
dependency at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from data.transforms import (  # noqa: E402
    SEVERITY_LEVELS,
    apply_named_transform,
    center_crop,
    color_jitter,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    resize_roundtrip,
)


def _make_test_image(size=(224, 224)) -> Image.Image:
    """A synthetic image with actual structure (gradient + noise + a hard edge),
    not a flat color -- flat images can hide bugs in blur/noise/JPEG (e.g. JPEG
    compresses a flat block to itself trivially, so it wouldn't catch a broken
    quality parameter)."""
    w, h = size
    x = np.linspace(0, 255, w, dtype=np.float32)
    y = np.linspace(0, 255, h, dtype=np.float32)
    gradient = np.outer(y, x) / 255.0  # (h, w) in [0, 255]
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 15, size=(h, w))
    channel = np.clip(gradient + noise, 0, 255).astype(np.uint8)
    # Hard edge: right half inverted, so crop/resize/blur all have something
    # structural to potentially disturb.
    channel[:, w // 2 :] = 255 - channel[:, w // 2 :]
    arr = np.stack([channel] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _assert_valid_rgb_image(img: Image.Image, expected_size=None):
    assert isinstance(img, Image.Image), f"expected a PIL Image, got {type(img)}"
    assert img.mode == "RGB", f"expected RGB mode, got {img.mode}"
    if expected_size is not None:
        assert img.size == expected_size, f"expected size {expected_size}, got {img.size}"
    arr = np.asarray(img)
    assert arr.min() >= 0 and arr.max() <= 255, "pixel values out of [0,255] range"
    # Not all-identical-pixel (i.e. the transform didn't collapse the image to a
    # flat color, which would indicate a broken transform).
    assert arr.std() > 1.0, "output image has ~zero variance -- transform likely broken"


def test_jpeg_compress_all_qualities():
    img = _make_test_image()
    for q in (90, 70, 50, 30):
        out = jpeg_compress(img, q)
        _assert_valid_rgb_image(out, expected_size=img.size)


def test_jpeg_compress_lower_quality_is_more_lossy():
    img = _make_test_image()
    q90 = np.asarray(jpeg_compress(img, 90), dtype=np.float32)
    q30 = np.asarray(jpeg_compress(img, 30), dtype=np.float32)
    orig = np.asarray(img, dtype=np.float32)
    err_90 = np.abs(q90 - orig).mean()
    err_30 = np.abs(q30 - orig).mean()
    assert err_30 > err_90, "quality=30 should introduce more distortion than quality=90"


def test_gaussian_blur_all_sigmas():
    img = _make_test_image()
    for sigma in (0.5, 1.0, 2.0):
        out = gaussian_blur(img, sigma)
        _assert_valid_rgb_image(out, expected_size=img.size)


def test_gaussian_blur_reduces_high_frequency_energy():
    img = _make_test_image()
    orig = np.asarray(img, dtype=np.float32)
    blurred = np.asarray(gaussian_blur(img, 2.0), dtype=np.float32)
    # Crude high-frequency proxy: mean absolute gradient between neighboring pixels.
    orig_grad = np.abs(np.diff(orig, axis=1)).mean()
    blurred_grad = np.abs(np.diff(blurred, axis=1)).mean()
    assert blurred_grad < orig_grad, "blur should reduce local pixel-to-pixel variation"


def test_resize_roundtrip_preserves_size():
    img = _make_test_image()
    for scale in (0.5, 0.25):
        out = resize_roundtrip(img, scale)
        _assert_valid_rgb_image(out, expected_size=img.size)


def test_gaussian_noise_all_sigmas():
    img = _make_test_image()
    for sigma in (0.02, 0.05, 0.10):
        out = gaussian_noise(img, sigma)
        _assert_valid_rgb_image(out, expected_size=img.size)


def test_gaussian_noise_scales_with_sigma():
    img = _make_test_image()
    orig = np.asarray(img, dtype=np.float32)
    low = np.asarray(gaussian_noise(img, 0.02), dtype=np.float32)
    high = np.asarray(gaussian_noise(img, 0.10), dtype=np.float32)
    err_low = np.abs(low - orig).mean()
    err_high = np.abs(high - orig).mean()
    assert err_high > err_low, "sigma=0.10 should perturb pixels more than sigma=0.02"


def test_color_jitter():
    img = _make_test_image()
    out = color_jitter(img, max_delta=0.20)
    _assert_valid_rgb_image(out, expected_size=img.size)


def test_center_crop_preserves_output_size_but_changes_content():
    img = _make_test_image()
    out = center_crop(img, crop_fraction=0.80)
    _assert_valid_rgb_image(out, expected_size=img.size)
    # Content should differ from the original (edges are cropped away then the
    # remaining 80% is resized back up) -- verifies this isn't a no-op.
    orig = np.asarray(img, dtype=np.float32)
    cropped = np.asarray(out, dtype=np.float32)
    assert not np.allclose(orig, cropped, atol=1.0), "center_crop appears to be a no-op"


def test_all_named_severity_levels_run_without_error():
    img = _make_test_image()
    for name in SEVERITY_LEVELS:
        out = apply_named_transform(img, name)
        _assert_valid_rgb_image(out, expected_size=img.size)


def test_compound_severities_are_more_destructive_than_single_transforms():
    img = _make_test_image()
    orig = np.asarray(img, dtype=np.float32)
    single = np.asarray(apply_named_transform(img, "jpeg_q70"), dtype=np.float32)
    compound = np.asarray(apply_named_transform(img, "compound_severe"), dtype=np.float32)
    err_single = np.abs(single - orig).mean()
    err_compound = np.abs(compound - orig).mean()
    assert err_compound > err_single, (
        "compound_severe (resize+jpeg+blur+noise) should distort the image more "
        "than a single jpeg_q70 transform"
    )


def test_unknown_severity_raises():
    img = _make_test_image()
    try:
        apply_named_transform(img, "not_a_real_severity")
    except KeyError:
        pass
    else:
        raise AssertionError("apply_named_transform should raise KeyError for an unknown severity name")


if __name__ == "__main__":
    # Pytest-free runner: discover and run every test_* function in this module.
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
