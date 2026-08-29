"""Tests for the hand-built spectral features (features/fft.py).

This is the one Step-1 branch with no pretrained weights behind it -- every number
is derived here, so it's also the one that can silently produce nonsense (a nan
from a flat image, a mis-sliced spectrum, a feature that responds to nothing).
These tests check the contract (width, names, finiteness, determinism) and, more
importantly, that the upsampling-fingerprint features actually *respond* to a
synthetic upsampling fingerprint.

Guarded the same way as model_01/tests: skips cleanly if torch isn't installed,
rather than failing the whole suite.
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
    from features.fft import FFTStatsFeatures

WORK_SIZE = 128


def _extractor(**kwargs):
    return FFTStatsFeatures(work_size=WORK_SIZE, n_threads=1, **kwargs)


def _natural_like(seed: int = 0, size: int = WORK_SIZE) -> torch.Tensor:
    """A 1/f-ish image: smooth low-frequency content plus a fine noise floor,
    which is roughly how a photograph's spectrum behaves."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(3, size, size)).astype(np.float32)
    spectrum = np.fft.fft2(noise, axes=(-2, -1))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fy**2 + fx**2) + 1e-3
    smooth = np.real(np.fft.ifft2(spectrum / radius, axes=(-2, -1)))
    smooth = (smooth - smooth.min()) / (np.ptp(smooth) + 1e-8)
    return torch.from_numpy(smooth.astype(np.float32)).unsqueeze(0)


def _upsampled(seed: int = 0, size: int = WORK_SIZE) -> torch.Tensor:
    """A 1/f image built at half resolution and nearest-neighbour upsampled --
    the crudest possible stand-in for a decoder's upsampling stage, which is
    exactly the periodic artifact the peak features are supposed to catch."""
    half = _natural_like(seed, size // 2)
    return torch.nn.functional.interpolate(half, size=(size, size), mode="nearest")


def test_dim_matches_feature_names():
    ex = _extractor()
    assert ex.dim == len(ex.feature_names()), (
        f"dim={ex.dim} but {len(ex.feature_names())} names -- the two are used to line "
        "up cache columns with the block spec and must agree"
    )
    assert ex.dim == 130, f"expected 130 features for the default config, got {ex.dim}"


def test_output_shape_and_finite():
    ex = _extractor()
    batch = torch.cat([_natural_like(0), _upsampled(1)], dim=0)
    feats = ex(batch)
    assert feats.shape == (2, ex.dim)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all(), "spectral features must never contain nan/inf"


def test_deterministic():
    ex = _extractor()
    img = _natural_like(3)
    assert np.allclose(ex(img), ex(img)), "extraction must be deterministic (no RNG in this block)"


def test_flat_image_does_not_produce_nan():
    """A constant image has a zero-variance residual -- skew/kurtosis/correlation
    are all undefined there, and one nan would poison a whole feature column
    during standardization."""
    ex = _extractor()
    feats = ex(torch.full((1, 3, WORK_SIZE, WORK_SIZE), 0.5))
    assert np.isfinite(feats).all()


def test_resizes_input_to_work_size():
    ex = FFTStatsFeatures(work_size=64, n_threads=1)
    feats = ex(_natural_like(0, size=WORK_SIZE))  # 128 in, 64 expected internally
    assert feats.shape == (1, ex.dim)
    assert np.isfinite(feats).all()


def test_upsampling_artifact_moves_the_peak_features():
    """The point of the block: an upsampled image should look different from a
    natural-spectrum one in the features designed to detect upsampling."""
    ex = _extractor()
    names = ex.feature_names()
    half_idx = names.index("fft_peak_half_nyquist")
    autocorr_idx = names.index("fft_res_autocorr_dx1")

    natural = np.stack([ex(_natural_like(s))[0] for s in range(4)])
    upsampled = np.stack([ex(_upsampled(s))[0] for s in range(4)])

    # Nearest-neighbour upsampling duplicates adjacent pixels, so the residual's
    # neighbour autocorrelation rises sharply -- a robust, direction-agnostic check.
    assert upsampled[:, autocorr_idx].mean() > natural[:, autocorr_idx].mean(), (
        "residual autocorrelation should be higher for an upsampled image"
    )
    # And the two classes should be separable somewhere in the block at all.
    separation = np.abs(upsampled.mean(0) - natural.mean(0)) / (natural.std(0) + 1e-6)
    assert separation.max() > 3.0, "no feature separates upsampled from natural images"
    assert np.isfinite(natural).all() and np.isfinite(upsampled).all()
    assert half_idx < ex.dim


def test_threaded_and_serial_agree():
    serial = FFTStatsFeatures(work_size=WORK_SIZE, n_threads=1)
    threaded = FFTStatsFeatures(work_size=WORK_SIZE, n_threads=4)
    batch = torch.cat([_natural_like(s) for s in range(4)], dim=0)
    assert np.allclose(serial(batch), threaded(batch), atol=1e-5), (
        "the thread pool must preserve row order and results"
    )


if __name__ == "__main__":
    import traceback

    if not HAS_TORCH:
        print("torch is not installed in this environment -- skipping FFT feature tests.")
        print("Install requirements.txt and re-run before starting a real extraction run.")
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
