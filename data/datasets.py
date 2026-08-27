"""PyTorch Dataset classes for the labeled real/fake image folders.

Expects each dataset to be laid out (see `prepare_data.py`) as:

    data/raw/<dataset_name>/real/*.jpg
    data/raw/<dataset_name>/fake/*.jpg

Label convention: 0 = real (authentic), 1 = fake (AI-generated) — matches the
`pred` field in infer.py's output (probability the image is AI-generated).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


class RealFakeImageDataset(Dataset):
    """Loads (image, label) pairs from one or more `real/` + `fake/` folder pairs.

    Parameters
    ----------
    dataset_roots:
        List of dataset root directories, each containing `real/` and `fake/`
        subfolders (e.g. ["data/raw/sid_set", "data/raw/cifake"]).
    transform:
        Optional callable applied to the PIL image before returning (e.g. a
        RobustnessAugment instance composed with a resize/normalize/ToTensor
        pipeline). Kept generic here so training and eval can plug in different
        pipelines (random augmentation vs. a fixed named severity).
    """

    def __init__(
        self,
        dataset_roots: list[str | Path],
        transform: Optional[Callable[[Image.Image], object]] = None,
    ):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for root in dataset_roots:
            root = Path(root)
            for label, subdir in ((0, "real"), (1, "fake")):
                for img_path in _list_images(root / subdir):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {dataset_roots}. Run data/prepare_data.py "
                "first to download and lay out the datasets."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, str(path)

    def split_train_val(self, val_fraction: float, seed: int = 42):
        """Return two new datasets (train, val) via a deterministic random split.

        Splitting here (rather than by folder) keeps the real/fake ratio roughly
        consistent across both splits.
        """
        rng = random.Random(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        n_val = int(len(indices) * val_fraction)
        val_idx, train_idx = set(indices[:n_val]), set(indices[n_val:])

        def _subset(idx_set):
            ds = RealFakeImageDataset.__new__(RealFakeImageDataset)
            ds.transform = self.transform
            ds.samples = [s for i, s in enumerate(self.samples) if i in idx_set]
            return ds

        return _subset(train_idx), _subset(val_idx)


class ImageFolderInference(Dataset):
    """Loads all images in a flat directory (no labels) — used by infer.py."""

    def __init__(self, input_dir: str | Path, transform: Optional[Callable] = None):
        self.paths = _list_images(Path(input_dir))
        self.transform = transform
        if not self.paths:
            raise RuntimeError(f"No images found in {input_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, str(path)
