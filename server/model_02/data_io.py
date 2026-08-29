"""Image loading for the feature-extraction pipeline.

model_01 hands the network a normalized tensor and lets the backbone own the
preprocessing. model_02 can't: it runs three extractors with three different
input conventions (DINOv2 wants ImageNet normalization at 224, CLIP wants CLIP
normalization at 224, the FFT block wants un-normalized pixels at its own working
resolution). So the loader here produces one *canonical* representation --

    float32 tensor, shape (3, S, S), values in [0, 1], no normalization --

and each extractor derives what it needs from that (see features/base.py). One
decode + resize per image, shared by all three branches.

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


def canonical_transform(canonical_size: int):
    """PIL -> (3, S, S) float tensor in [0, 1]."""
    return T.Compose([T.Resize((canonical_size, canonical_size)), T.ToTensor()])


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
