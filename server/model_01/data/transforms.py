"""Robustness augmentation pipeline.

This module implements the exact transform family named in the challenge brief
(JPEG compression, Gaussian blur, resize round-trip, Gaussian noise, color jitter,
center crop), both as:

  1. Random per-sample augmentations applied during training (`RobustnessAugment`),
     so the model learns invariance to them instead of memorizing pristine-image
     statistics.
  2. Deterministic, labeled transforms at fixed severities (`SEVERITY_LEVELS` /
     `apply_named_transform`) used by `eval/robustness_eval.py` to build the
     clean-vs-transformed evaluation matrix.

All functions operate on PIL Images and return PIL Images, so they compose cleanly
with torchvision pipelines.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance


# --------------------------------------------------------------------------- #
# Individual transforms (each matches one row of the challenge's transform table)
# --------------------------------------------------------------------------- #

def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode through JPEG at the given quality (90/70/50/30 in the brief).

    Real-world analog: social-media re-encode, messaging apps.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur with the given sigma (0.5/1.0/2.0 in the brief).

    Real-world analog: out-of-focus capture.
    """
    if sigma <= 0:
        return img
    from PIL import ImageFilter

    # PIL's GaussianBlur radius ~ sigma for our purposes at these small magnitudes.
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    """Downscale by `scale` (0.5 / 0.25 in the brief) then upscale back to original size.

    Real-world analog: thumbnail generation.
    """
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Additive Gaussian noise with std `sigma` on a [0,1]-scaled image (0.02/0.05/0.10 in the brief).

    Real-world analog: low-light sensor noise.
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8), mode="RGB")


def color_jitter(img: Image.Image, max_delta: float = 0.20) -> Image.Image:
    """Randomly perturb brightness/contrast/saturation by up to +/- max_delta (0.20 in the brief).

    Real-world analog: filter apps, auto-enhance.
    """
    img = img.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-max_delta, max_delta)
        img = enhancer_cls(img).enhance(factor)
    return img


def center_crop(img: Image.Image, crop_fraction: float = 0.80) -> Image.Image:
    """Center-crop to `crop_fraction` of each dimension, then resize back to original size.

    Real-world analog: profile-picture cropping, framing.
    """
    w, h = img.size
    new_w, new_h = int(w * crop_fraction), int(h * crop_fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


# --------------------------------------------------------------------------- #
# Named severity levels for deterministic evaluation (used by robustness_eval.py)
# --------------------------------------------------------------------------- #

# Each entry maps a severity name -> list of (transform_name, callable) to apply in
# sequence. "clean" is the identity transform. Feel free to add "severe_compound"
# entries that stack multiple transforms, since real redistribution often does.
SEVERITY_LEVELS: dict[str, list[tuple[str, Callable[[Image.Image], Image.Image]]]] = {
    "clean": [],
    "jpeg_q90": [("jpeg", lambda im: jpeg_compress(im, 90))],
    "jpeg_q70": [("jpeg", lambda im: jpeg_compress(im, 70))],
    "jpeg_q50": [("jpeg", lambda im: jpeg_compress(im, 50))],
    "jpeg_q30": [("jpeg", lambda im: jpeg_compress(im, 30))],
    "blur_s0.5": [("blur", lambda im: gaussian_blur(im, 0.5))],
    "blur_s1.0": [("blur", lambda im: gaussian_blur(im, 1.0))],
    "blur_s2.0": [("blur", lambda im: gaussian_blur(im, 2.0))],
    "resize_0.5x": [("resize", lambda im: resize_roundtrip(im, 0.5))],
    "resize_0.25x": [("resize", lambda im: resize_roundtrip(im, 0.25))],
    "noise_s0.02": [("noise", lambda im: gaussian_noise(im, 0.02))],
    "noise_s0.05": [("noise", lambda im: gaussian_noise(im, 0.05))],
    "noise_s0.10": [("noise", lambda im: gaussian_noise(im, 0.10))],
    "color_jitter_20pct": [("color_jitter", lambda im: color_jitter(im, 0.20))],
    "center_crop_80pct": [("center_crop", lambda im: center_crop(im, 0.80))],
    # Compounded transform to approximate realistic multi-step redistribution
    # (e.g. resized thumbnail, then re-compressed, then lightly filtered).
    "compound_moderate": [
        ("resize", lambda im: resize_roundtrip(im, 0.5)),
        ("jpeg", lambda im: jpeg_compress(im, 70)),
        ("color_jitter", lambda im: color_jitter(im, 0.20)),
    ],
    "compound_severe": [
        ("resize", lambda im: resize_roundtrip(im, 0.25)),
        ("jpeg", lambda im: jpeg_compress(im, 30)),
        ("blur", lambda im: gaussian_blur(im, 1.0)),
        ("noise", lambda im: gaussian_noise(im, 0.05)),
    ],
}


def apply_named_transform(img: Image.Image, severity_name: str) -> Image.Image:
    """Apply a named, deterministic severity level (see SEVERITY_LEVELS) to `img`."""
    if severity_name not in SEVERITY_LEVELS:
        raise KeyError(
            f"Unknown severity '{severity_name}'. Available: {list(SEVERITY_LEVELS)}"
        )
    for _, fn in SEVERITY_LEVELS[severity_name]:
        img = fn(img)
    return img


# --------------------------------------------------------------------------- #
# Random training-time augmentation
# --------------------------------------------------------------------------- #

@dataclass
class RobustnessAugment:
    """Randomly applies a subset of the challenge's transform family per sample.

    Each transform is independently applied with its own probability, and multiple
    transforms can stack in one call — this matches how real redistribution usually
    compounds several operations (e.g. resize-for-thumbnail then re-compress).

    Config keys mirror `configs/default.yaml`'s `augmentation` section.
    """

    jpeg_prob: float = 0.5
    jpeg_qualities: list[int] = field(default_factory=lambda: [90, 70, 50, 30])

    blur_prob: float = 0.4
    blur_sigmas: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    resize_prob: float = 0.4
    resize_scales: list[float] = field(default_factory=lambda: [0.5, 0.25])

    noise_prob: float = 0.4
    noise_sigmas: list[float] = field(default_factory=lambda: [0.02, 0.05, 0.10])

    color_jitter_prob: float = 0.4
    color_jitter_max_delta: float = 0.20

    center_crop_prob: float = 0.3
    center_crop_fraction: float = 0.80

    @classmethod
    def from_config(cls, cfg: dict) -> "RobustnessAugment":
        """Build from the `augmentation` block of a loaded YAML config."""
        return cls(
            jpeg_prob=cfg["jpeg"]["prob"],
            jpeg_qualities=cfg["jpeg"]["qualities"],
            blur_prob=cfg["gaussian_blur"]["prob"],
            blur_sigmas=cfg["gaussian_blur"]["sigmas"],
            resize_prob=cfg["resize_roundtrip"]["prob"],
            resize_scales=cfg["resize_roundtrip"]["scales"],
            noise_prob=cfg["gaussian_noise"]["prob"],
            noise_sigmas=cfg["gaussian_noise"]["sigmas"],
            color_jitter_prob=cfg["color_jitter"]["prob"],
            color_jitter_max_delta=cfg["color_jitter"]["max_delta"],
            center_crop_prob=cfg["center_crop"]["prob"],
            center_crop_fraction=cfg["center_crop"]["crop_fraction"],
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.resize_prob:
            img = resize_roundtrip(img, random.choice(self.resize_scales))
        if random.random() < self.center_crop_prob:
            img = center_crop(img, self.center_crop_fraction)
        if random.random() < self.blur_prob:
            img = gaussian_blur(img, random.choice(self.blur_sigmas))
        if random.random() < self.color_jitter_prob:
            img = color_jitter(img, self.color_jitter_max_delta)
        if random.random() < self.noise_prob:
            img = gaussian_noise(img, random.choice(self.noise_sigmas))
        # JPEG last: re-compressing after other pixel-level ops best matches a real
        # "edited then re-uploaded" pipeline.
        if random.random() < self.jpeg_prob:
            img = jpeg_compress(img, random.choice(self.jpeg_qualities))
        return img
