"""Image loading for the feature-extraction pipeline.

model_01 hands the network a normalized tensor and lets the backbone own the
preprocessing. model_02 can't: it runs three extractors with three different
input conventions (DINOv2 wants ImageNet normalization at 224, CLIP wants CLIP
normalization at 224, the FFT block wants un-normalized pixels at its own working
resolution). So the loader here produces one *canonical* representation --

    float32 tensor, shape (3, S, S), values in [0, 1], no normalization --

and each extractor derives what it needs from that (see features/base.py). One
decode + canonicalization per image, shared by all three branches.

Canonicalization is a native-resolution center crop wherever the image is large
enough (see `CanonicalCrop`), falling back to a resize only for inputs smaller
than `canonical_size`. Rescaling writes directly into the frequency band the FFT
branch reads, so it is avoided wherever it can be.

The optional `pil_transform` slot is where the challenge's redistribution
transforms go: `RobustnessAugment` (random, for building an augmented training
cache) or a fixed `SEVERITY_LEVELS` entry (deterministic, for the robustness
matrix). It runs on the PIL image *before* the canonical resize, which is the
right order -- JPEG artifacts and resize round-trips have to be applied at the
image's own resolution to be realistic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from shared import IMAGE_EXTENSIONS, RealFakeImageDataset, apply_named_transform


class CanonicalCrop:
    """PIL -> canonical `size`x`size` PIL image, preferring a native-resolution crop.

    Cropping is preferred over resizing because `features/fft.py` reads the noise
    floor, and rescaling *writes* to exactly that band: an upscale manufactures
    high frequencies by interpolation and a downscale destroys them. Either way
    the spectral columns end up describing our own preprocessing rather than the
    generator, which is the failure the top-level README documents for upsampled
    CIFAKE.

    `prepare_generators.py` already writes training data as native-resolution
    center crops at this size, so training sees no rescaling at all. Inference
    has to cope with whatever resolution a caller hands it, and the two must
    agree -- a model trained on native crops but served resized uploads is a
    train/serve mismatch precisely in the band it relies on. So:

      - larger than `size` on both sides -> center crop at native resolution,
        offset snapped to a multiple of 8 to preserve JPEG grid phase (64 of the
        FFT block's 130 columns are a block-DCT profile keyed to that grid);
      - smaller on either side -> resize the short side up, then crop. This is
        the lossy path and it is unavoidable for small inputs, but it is now the
        exception rather than the rule.

    32x32 CIFAKE always takes the resize path, so existing CIFAKE caches keep
    their previous behaviour.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if min(w, h) < self.size:
            scale = self.size / min(w, h)
            # BILINEAR, matching torchvision's T.Resize default, so the small-image
            # path is byte-identical to the previous behaviour and existing CIFAKE
            # caches and checkpoints stay valid. PIL's resize is antialiased on
            # downscale, which the README requires for the same spectral reason.
            img = img.resize(
                (max(self.size, round(w * scale)), max(self.size, round(h * scale))),
                Image.BILINEAR,
            )
            w, h = img.size
        left = ((w - self.size) // 2) & ~7
        top = ((h - self.size) // 2) & ~7
        return img.crop((left, top, left + self.size, top + self.size))


def canonical_transform(canonical_size: int):
    """PIL -> (3, S, S) float tensor in [0, 1]."""
    return T.Compose([CanonicalCrop(canonical_size), T.ToTensor()])


class NamedSeverity:
    """Callable wrapper applying one deterministic SEVERITY_LEVELS entry."""

    def __init__(self, severity_name: str):
        self.severity_name = severity_name

    def __call__(self, img: Image.Image) -> Image.Image:
        return apply_named_transform(img, self.severity_name)


def build_labeled_samples(dataset_roots: Sequence[str | Path]) -> list[tuple[Path, int]]:
    """(path, label) pairs from `<root>/real` + `<root>/fake` folders (0 = real, 1 = fake)."""
    return list(RealFakeImageDataset(list(dataset_roots), transform=None).samples)


class CanonicalDataset(Dataset):
    """Labeled images as canonical tensors, with a stable group id per source image.

    `group_id` is the index of the *original* image. When the feature cache holds
    several augmented copies of one image, every copy carries the same group id,
    and train.py splits on groups rather than rows -- otherwise a JPEG-recompressed
    copy of a training image lands in validation and the val score is inflated by
    near-duplicate leakage.
    """

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        canonical_size: int,
        pil_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        group_ids: Optional[Sequence[int]] = None,
    ):
        self.samples = list(samples)
        self.pil_transform = pil_transform
        self.to_canonical = canonical_transform(canonical_size)
        self.group_ids = list(group_ids) if group_ids is not None else list(range(len(self.samples)))
        if len(self.group_ids) != len(self.samples):
            raise ValueError("group_ids must be the same length as samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.pil_transform is not None:
            img = self.pil_transform(img)
        return self.to_canonical(img), label, self.group_ids[idx], str(path)


class CanonicalInferenceDataset(Dataset):
    """Unlabeled flat directory of images -> canonical tensors (used by infer.py)."""

    def __init__(self, input_dir: str | Path, canonical_size: int):
        root = Path(input_dir)
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise RuntimeError(f"No images found in {input_dir}")
        self.to_canonical = canonical_transform(canonical_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.to_canonical(img), str(path)


def collate_labeled(batch):
    imgs, labels, groups, paths = zip(*batch)
    return torch.stack(imgs), list(labels), list(groups), list(paths)


def collate_unlabeled(batch):
    imgs, paths = zip(*batch)
    return torch.stack(imgs), list(paths)
