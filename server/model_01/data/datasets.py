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
        strict: bool = True,
    ):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.per_dataset_counts: dict[str, dict[str, int]] = {}

        problems: list[str] = []
        for root in dataset_roots:
            root = Path(root)
            counts = {"real": 0, "fake": 0}
            for label, subdir in ((0, "real"), (1, "fake")):
                found = _list_images(root / subdir)
                counts[subdir] = len(found)
                for img_path in found:
                    self.samples.append((img_path, label))
            self.per_dataset_counts[str(root)] = counts

            # A dataset listed in the config but absent (or empty) on disk used to
            # contribute zero images in total silence, so a run would train on less
            # data than the config claimed and nothing in the logs would say so --
            # that is exactly what happened to "sid_set" in job 768468.
            if not root.exists():
                problems.append(f"  {root}: does not exist")
            elif counts["real"] == 0 and counts["fake"] == 0:
                problems.append(f"  {root}: exists but contains no images under real/ or fake/")
            elif counts["real"] == 0 or counts["fake"] == 0:
                problems.append(
                    f"  {root}: one-sided -- real={counts['real']} fake={counts['fake']} "
                    "(a single-class dataset cannot train a binary classifier)"
                )

        if problems and strict:
            raise RuntimeError(
                "Dataset roots are missing, empty, or single-class:\n"
                + "\n".join(problems)
                + "\n\nRun data/prepare_data.py to lay them out, remove them from "
                "`data.train_datasets` in the config, or pass strict=False to load "
                "whatever is present. Refusing to train on silently-reduced data."
            )
        for problem in problems:
            print(f"WARNING: dataset root problem:{problem}")

        if not self.samples:
            raise RuntimeError(
                f"No images found under {dataset_roots}. Run data/prepare_data.py "
                "first to download and lay out the datasets."
            )

    def describe(self) -> str:
        """Per-dataset and overall class counts, for logging at startup."""
        lines = []
        for root, counts in self.per_dataset_counts.items():
            lines.append(f"  {root}: real={counts['real']} fake={counts['fake']}")
        n_fake = sum(label for _, label in self.samples)
        n_real = len(self.samples) - n_fake
        balance = n_fake / len(self.samples) if self.samples else float("nan")
        lines.append(f"  TOTAL: real={n_real} fake={n_fake} n={len(self.samples)} "
                     f"fake_fraction={balance:.4f}")
        return "\n".join(lines)

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
            ds.per_dataset_counts = self.per_dataset_counts
            ds.samples = [s for i, s in enumerate(self.samples) if i in idx_set]
            return ds

        return _subset(train_idx), _subset(val_idx)


class PairedViewDataset(Dataset):
    """Wraps a list of (path, label) samples and returns BOTH a clean and an
    augmented view of the same underlying image per sample.

    This backs train.py's consistency loss: instead of the earlier same-batch
    stand-in (which compared a batch against itself and taught the model
    nothing new), each sample now gets a genuine clean/augmented pair so the
    consistency term actually penalizes the model for predicting differently
    on a redistributed copy of an image versus the original -- which is the
    literal "robust under transform" objective from the challenge brief.
    """

    def __init__(self, samples: list[tuple[Path, int]], clean_transform, aug_transform):
        self.samples = samples
        self.clean_transform = clean_transform
        self.aug_transform = aug_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        clean = self.clean_transform(img)
        augmented = self.aug_transform(img)
        return clean, augmented, label, str(path)


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
