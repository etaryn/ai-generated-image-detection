"""Generates a small synthetic real/fake image dataset for pipeline smoke-testing,
with no network access required -- useful for CI, and for validating the full
train/eval pipeline before the real datasets (SID_Set/CIFAKE/WildFake, see
data/prepare_data.py) are downloaded and laid out.

"Real" images are smooth photographic-style blobs/gradients plus natural-looking
noise. "Fake" images are the exact same base image with a subtle periodic
checkerboard-style pattern added at a fixed spatial frequency -- a simplified,
well-documented proxy for the upsampling/transposed-convolution artifacts that
show up in real GAN/diffusion output (see Zhang et al., "Detecting and Simulating
Artifacts in GAN Fake Images", 2019). This is obviously not a substitute for real
AIGC data (a model that only ever sees this will overfit to "does this image have
a faint sine-wave grid" and nothing else) -- it exists purely so the pipeline
mechanics (data loading, augmentation, training loop, eval, attention rollout) can
be exercised end-to-end before real data is available.

Usage:
    python data/make_synthetic_dataset.py --out data/raw/synthetic --n_per_class 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _base_image(rng: np.random.Generator, size: int = 224) -> np.ndarray:
    """A smooth "scene": a few overlapping soft gaussian blobs + light noise."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size), dtype=np.float32)
    n_blobs = rng.integers(2, 5)
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0, size, size=2)
        sigma = rng.uniform(size * 0.15, size * 0.5)
        amp = rng.uniform(40, 120)
        img += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    img += rng.normal(0, 8, size=img.shape)
    return np.clip(img, 0, 255)


def _add_upsampling_artifact(
    img: np.ndarray,
    rng: np.random.Generator,
    amplitude_range: tuple[float, float] = (3.0, 8.0),
) -> np.ndarray:
    """Adds a subtle periodic pattern at a fixed high spatial frequency, loosely
    modeling the checkerboard-style artifacts transposed-convolution upsampling
    leaves in real generator output.

    The default amplitude (3-8) is deliberately near the base image's own noise
    floor (std 8), since real AIGC artifacts are subtle too. That realism makes
    this dataset a poor fit for a *wiring* check: a model can be perfectly
    correct and still fail to separate the classes in a few hundred steps, which
    is exactly what happened when the pre-flight gate was first pointed at it
    (0.5039 after 300 steps, with gradient norms flat at ~0.5, while the same
    code hit 0.9961 on real CIFAKE). Pass a larger range to generate an
    easy-to-separate variant for gate/plumbing tests -- see scripts/smoke.sbatch.
    """
    size = img.shape[0]
    freq = rng.choice([4, 8, 16])  # cycles across the image
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    pattern = np.sin(2 * np.pi * freq * xx / size) * np.sin(2 * np.pi * freq * yy / size)
    amplitude = rng.uniform(*amplitude_range)
    return np.clip(img + amplitude * pattern, 0, 255)


def generate_dataset(out_dir: Path, n_per_class: int, size: int, seed: int,
                     amplitude_range: tuple[float, float] = (3.0, 8.0)) -> None:
    rng = np.random.default_rng(seed)
    for label in ("real", "fake"):
        (out_dir / label).mkdir(parents=True, exist_ok=True)

    for i in range(n_per_class):
        base = _base_image(rng, size)
        real_arr = np.stack([base] * 3, axis=-1).astype(np.uint8)
        Image.fromarray(real_arr, mode="RGB").save(out_dir / "real" / f"real_{i:04d}.png")

        fake_channel = _add_upsampling_artifact(base.copy(), rng, amplitude_range)
        fake_arr = np.stack([fake_channel] * 3, axis=-1).astype(np.uint8)
        Image.fromarray(fake_arr, mode="RGB").save(out_dir / "fake" / f"fake_{i:04d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/synthetic")
    parser.add_argument("--n_per_class", type=int, default=200)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--amplitude", type=float, nargs=2, default=[3.0, 8.0], metavar=("MIN", "MAX"),
        help="Artifact amplitude range. The default (3 8) sits at the base image's "
             "noise floor and is realistically subtle; use something like (20 30) for "
             "an easily-separable variant to test code wiring against.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    generate_dataset(out_dir, args.n_per_class, args.size, args.seed, tuple(args.amplitude))
    print(f"Wrote {args.n_per_class} real + {args.n_per_class} fake images to {out_dir} "
          f"(artifact amplitude {args.amplitude[0]}-{args.amplitude[1]})")


if __name__ == "__main__":
    main()
