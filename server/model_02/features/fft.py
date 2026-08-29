"""Step 1c: FFT / spectral statistics -- "numbers describing microscopic pixel noise".

DINOv2 and CLIP both look at an image the way a *viewer* does. This block looks at
it the way a *sensor forensics* tool does, and it is entirely hand-derived: no
network, no weights, nothing to train or overfit.

The signal it targets: a photograph's high-frequency content is sensor noise plus
optics -- close to isotropic, close to a 1/f^a power law, with channel noise
correlations set by the demosaicing filter. A generated image's high-frequency
content is instead manufactured by a decoder that repeatedly upsamples, which
leaves periodic structure -- energy concentrated at fractions of the Nyquist
frequency, directional (axis-aligned) bias, and unnaturally correlated channels.
Those differences live in the parts of the image a viewer never consciously reads.

Everything is computed on a high-pass *residual* (image minus its blurred self)
wherever possible, so the descriptor is about the noise floor rather than about
scene content -- otherwise the classifier would learn "photos of grass" instead
of "photos".

Feature blocks (default config -> 130 numbers):
    32  radial log-power profile, mean-centered    (isotropic spectrum shape)
     1  mean log power                             (the scale removed by centering)
     3  power-law fit: slope, intercept, RMSE      (natural images are ~1/f^a)
    16  azimuthal log-power profile, mean-centered (directional/grid bias)
     4  peak + high-frequency-band descriptors     (upsampling fingerprints)
     3  residual std / skew / kurtosis             (noise-floor shape)
     4  residual autocorrelation at lags 1 and 2   (upsampling correlates neighbors)
     3  cross-channel residual correlation         (demosaicing vs. decoder)
    64  8x8 block-DCT mean log-magnitude           (JPEG grid + generator fingerprint)

Caveat worth stating up front: these are exactly the statistics that JPEG
recompression, blurring and resizing attack -- which is why training runs on
augmented copies (see extract_features.py --aug-copies) and why this block's
contribution is reported separately (eval/ablation.py).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy import fft as sp_fft
from scipy import ndimage, stats

from features.base import FeatureExtractor

EPS = 1e-8
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Cache of (radius, angle) grids per spectrum shape -- these depend only on the
# working resolution, so building them once per run rather than per image is a
# meaningful fraction of this block's runtime.
_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _freq_grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """fftshifted (radius, angle) grids in cycles/pixel; radius 0.5 == Nyquist."""
    key = (h, w)
    if key not in _GRID_CACHE:
        fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
        fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
        radius = np.sqrt(fy**2 + fx**2)
        angle = np.mod(np.arctan2(fy, fx), np.pi)  # spectrum is symmetric -> [0, pi)
        _GRID_CACHE[key] = (radius.astype(np.float32), angle.astype(np.float32))
    return _GRID_CACHE[key]


def _binned_mean(values: np.ndarray, bin_index: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean of `values` per bin; empty bins fall back to the global mean."""
    sums = np.bincount(bin_index, weights=values, minlength=n_bins)[:n_bins]
    counts = np.bincount(bin_index, minlength=n_bins)[:n_bins]
    out = np.full(n_bins, float(values.mean()), dtype=np.float64)
    nonempty = counts > 0
    out[nonempty] = sums[nonempty] / counts[nonempty]
    return out


def _log_power_spectrum(gray: np.ndarray) -> np.ndarray:
    """Windowed log power spectrum, fftshifted.

    The Hann window is not cosmetic: without it the FFT treats the image as
    periodic, and the discontinuity at the wrap-around edge injects a bright
    cross along the axes that swamps the real directional signal being measured.
    """
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * window))
    return np.log(np.abs(spectrum) ** 2 + EPS).astype(np.float32)


def _residual(gray: np.ndarray, sigma: float) -> np.ndarray:
    """High-pass residual: the image minus a blurred copy of itself."""
    return gray - ndimage.gaussian_filter(gray, sigma=sigma, mode="reflect")


def _autocorr(res: np.ndarray, dy: int, dx: int) -> float:
    """Normalized autocorrelation of the residual at one integer lag."""
    h, w = res.shape
    a = res[: h - dy, : w - dx]
    b = res[dy:, dx:]
    denom = float(res.var()) + EPS
    return float((a * b).mean() / denom)


def _block_dct_profile(gray: np.ndarray) -> np.ndarray:
    """Mean log-magnitude of each of the 64 coefficients of an 8x8 block DCT.

    AC coefficients are reported relative to the DC term (which is kept as its own
    feature), so the profile describes the *shape* of the block spectrum rather
    than overall image brightness. The 8x8 grid is deliberately JPEG's grid: it
    picks up both compression history and the block-periodic texture some decoders
    leave behind.
    """
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    blocks = gray[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)
    coeffs = sp_fft.dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    mean_mag = np.log1p(np.abs(coeffs)).mean(axis=(0, 1)).reshape(64)
    dc = mean_mag[0]
    profile = mean_mag - dc
    profile[0] = dc
    return profile.astype(np.float64)


class FFTStatsFeatures(FeatureExtractor):
    name = "fft"

    def __init__(
        self,
        work_size: int = 256,
        n_radial_bins: int = 32,
        n_angular_bins: int = 16,
        blur_sigma: float = 1.0,
        n_threads: int = 4,
    ):
        self.work_size = work_size
        self.n_radial_bins = n_radial_bins
        self.n_angular_bins = n_angular_bins
        self.blur_sigma = blur_sigma
        self.n_threads = max(1, n_threads)
        self.dim = n_radial_bins + 4 + n_angular_bins + 4 + 3 + 4 + 3 + 64
        self._pool = ThreadPoolExecutor(max_workers=self.n_threads) if self.n_threads > 1 else None

    # ------------------------------------------------------------------ #
    # Per-image computation
    # ------------------------------------------------------------------ #
    def _spectral_features(self, gray: np.ndarray) -> np.ndarray:
        log_power = _log_power_spectrum(gray)
        h, w = gray.shape
        radius, angle = _freq_grids(h, w)

        # --- radial profile over [0, Nyquist] ---------------------------- #
        in_disk = radius <= 0.5
        r_bin = np.clip(
            (radius[in_disk] / 0.5 * self.n_radial_bins).astype(np.int64), 0, self.n_radial_bins - 1
        )
        radial = _binned_mean(log_power[in_disk], r_bin, self.n_radial_bins)
        radial_mean = float(radial.mean())
        radial_centered = radial - radial_mean

        # --- power-law fit: natural images sit close to a straight line --- #
        bin_centers = (np.arange(self.n_radial_bins) + 0.5) / self.n_radial_bins * 0.5
        log_r = np.log(bin_centers)
        slope, intercept = np.polyfit(log_r, radial, 1)
        fit_rmse = float(np.sqrt(np.mean((radial - (slope * log_r + intercept)) ** 2)))

        # --- azimuthal profile over the mid/high band -------------------- #
        band = (radius >= 0.15) & (radius <= 0.5)
        a_bin = np.clip(
            (angle[band] / np.pi * self.n_angular_bins).astype(np.int64), 0, self.n_angular_bins - 1
        )
        angular = _binned_mean(log_power[band], a_bin, self.n_angular_bins)
        angular_centered = angular - angular.mean()

        # --- upsampling-fingerprint descriptors --------------------------- #
        def annulus_mean(lo: float, hi: float) -> float:
            mask = (radius >= lo) & (radius < hi)
            return float(log_power[mask].mean()) if mask.any() else radial_mean

        # Transposed-convolution / pixel-shuffle upsampling concentrates energy at
        # half and quarter Nyquist; measuring each against its own local
        # neighborhood makes these peak detectors rather than energy detectors
        # (raw energy in a band is mostly scene content).
        peak_half = annulus_mean(0.24, 0.26) - 0.5 * (annulus_mean(0.20, 0.24) + annulus_mean(0.26, 0.30))
        peak_quarter = annulus_mean(0.115, 0.135) - 0.5 * (
            annulus_mean(0.09, 0.115) + annulus_mean(0.135, 0.16)
        )
        hf_ratio = annulus_mean(0.35, 0.5) - annulus_mean(0.05, 0.15)
        hf_vals = log_power[radius >= 0.15]
        peakiness = float((hf_vals.max() - np.median(hf_vals)) / (hf_vals.std() + EPS))

        return np.concatenate([
            radial_centered,
            [radial_mean, slope, intercept, fit_rmse],
            angular_centered,
            [peak_half, peak_quarter, hf_ratio, peakiness],
        ])

    def _residual_features(self, rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
        res = _residual(gray, self.blur_sigma)
        res_std = float(res.std())
        # skew/kurtosis are undefined for a constant residual (a flat synthetic
        # image, or one blurred into uniformity) -- report 0 rather than nan, since
        # one nan would poison standardization for that whole column.
        if res_std < EPS:
            shape_stats = [0.0, 0.0, 0.0]
            autocorrs = [0.0, 0.0, 0.0, 0.0]
        else:
            shape_stats = [
                res_std,
                float(stats.skew(res, axis=None)),
                float(stats.kurtosis(res, axis=None)),
            ]
            autocorrs = [
                _autocorr(res, 0, 1),
                _autocorr(res, 1, 0),
                _autocorr(res, 0, 2),
                _autocorr(res, 2, 0),
            ]

        # Cross-channel residual correlation: a camera's demosaicing ties the
        # channels' noise together in a specific way; a decoder's does not.
        chan_res = [_residual(rgb[c], self.blur_sigma).ravel() for c in range(3)]
        channel_corrs = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            a, b = chan_res[i], chan_res[j]
            if a.std() < EPS or b.std() < EPS:
                channel_corrs.append(0.0)
            else:
                channel_corrs.append(float(np.corrcoef(a, b)[0, 1]))

        return np.concatenate([shape_stats, autocorrs, channel_corrs])

    def _one_image(self, rgb: np.ndarray) -> np.ndarray:
        gray = np.tensordot(LUMA_WEIGHTS, rgb, axes=([0], [0])).astype(np.float32)
        feats = np.concatenate([
            self._spectral_features(gray),
            self._residual_features(rgb, gray),
            _block_dct_profile(gray),
        ])
        # A pathological image (fully flat, fully saturated) can still produce a
        # non-finite fit coefficient; clamp rather than let it reach the scaler.
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def __call__(self, canonical: torch.Tensor) -> np.ndarray:
        if canonical.shape[-1] != self.work_size or canonical.shape[-2] != self.work_size:
            canonical = torch.nn.functional.interpolate(
                canonical,
                size=(self.work_size, self.work_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        batch = canonical.detach().float().cpu().numpy()
        if self._pool is not None:
            # numpy/scipy release the GIL inside their FFT and filter kernels, so
            # threads genuinely help here -- this block is the CPU-bound part of
            # extraction while the two ViTs run on the GPU.
            rows = list(self._pool.map(self._one_image, batch))
        else:
            rows = [self._one_image(img) for img in batch]
        return np.stack(rows).astype(np.float32)

    def feature_names(self) -> list[str]:
        names = [f"fft_radial_{i}" for i in range(self.n_radial_bins)]
        names += ["fft_radial_mean", "fft_powerlaw_slope", "fft_powerlaw_intercept", "fft_powerlaw_rmse"]
        names += [f"fft_angular_{i}" for i in range(self.n_angular_bins)]
        names += ["fft_peak_half_nyquist", "fft_peak_quarter_nyquist", "fft_hf_band_ratio", "fft_peakiness"]
        names += ["fft_res_std", "fft_res_skew", "fft_res_kurtosis"]
        names += [
            "fft_res_autocorr_dx1",
            "fft_res_autocorr_dy1",
            "fft_res_autocorr_dx2",
            "fft_res_autocorr_dy2",
        ]
        names += ["fft_chan_corr_rg", "fft_chan_corr_rb", "fft_chan_corr_gb"]
        names += [f"fft_dct8x8_{i // 8}{i % 8}" for i in range(64)]
        return names

    def signature(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "work_size": self.work_size,
            "n_radial_bins": self.n_radial_bins,
            "n_angular_bins": self.n_angular_bins,
            "blur_sigma": self.blur_sigma,
        }
