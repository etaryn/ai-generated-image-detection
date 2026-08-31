"""Step 1 driver: turn an image folder into a cached feature matrix.

Because every extractor is frozen, features are computed exactly once and reused
for every classifier experiment afterwards -- that's the whole practical argument
for this architecture. Extraction is the expensive part (two ViT forward passes
plus the spectral statistics per image); training a classifier on the cache
afterwards takes seconds to minutes, so hyperparameters, classifier type and
feature ablations can be explored without touching a GPU again.

Augmented copies
----------------
`--aug-copies N` writes N additional rows per image, each an independently
randomized pass of the challenge's redistribution transforms (model_01's
`RobustnessAugment`). This is how robustness enters a frozen-feature pipeline:
there's no end-to-end training loop to add a consistency loss to, so the
invariance has to come from the classifier seeing both pristine and redistributed
versions of the same image. Every copy inherits the original's `group_id`, and
train.py splits on groups, so the copies can't leak across the train/val boundary.

Usage:
    # training cache (clean + 1 augmented copy per image)
    python extract_features.py --config configs/default.yaml --out features/cache/train_cifake.npz

    # held-out demo set, clean only
    python extract_features.py --config configs/default.yaml --demo-eval-set \\
        --aug-copies 0 --out features/cache/demo_clean.npz

    # the same set under one fixed redistribution severity
    python extract_features.py --config configs/default.yaml --demo-eval-set \\
        --aug-copies 0 --severity jpeg_q30 --out features/cache/demo_jpeg30.npz
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_io import CanonicalDataset, NamedSeverity, build_labeled_samples, collate_labeled
from features.pipeline import FeatureStack
from shared import RobustnessAugment


def resolve_dataset_roots(cfg: dict, names: list[str] | None = None,
                          data_root: str | Path | None = None) -> list[Path]:
    """Dataset directories to read, as `<data_root>/<name>`.

    `data_root` overrides the config's, mirroring model_01's `train.py --data_root`.
    The SLURM scripts use it to point at a copy of the dataset unpacked onto the
    compute node's local disk: $HOME is inode-quota'd on this cluster and reading
    tens of thousands of small files back over NFS is the slow part of extraction.
    """
    root = Path(data_root) if data_root else Path(cfg["data"]["data_root"])
    names = names if names is not None else cfg["data"]["train_datasets"]
    return [root / name for name in names]


def demo_eval_samples(cfg: dict) -> list[tuple[Path, int]]:
    """The challenge's demonstration set: COCO val2017 (real) vs DALL-E Advanced (fake).

    Laid out as two flat folders rather than a real/fake pair, so it can't be
    picked up accidentally by a training glob -- this subset is demonstration-only
    and must never appear in `train_datasets`.
    """
    from shared import IMAGE_EXTENSIONS

    demo = cfg["data"]["demo_eval_set"]
    samples: list[tuple[Path, int]] = []
    for key, label in (("real_dir", 0), ("fake_dir", 1)):
        folder = Path(demo[key])
        found = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not found:
            raise RuntimeError(f"No images found under demo_eval_set.{key} = {folder}")
        samples.extend((p, label) for p in found)
    return samples


def build_rows(
    samples: list[tuple[Path, int]], aug_copies: int, seed: int
) -> tuple[list[tuple[Path, int]], list[int], list[bool]]:
    """Expand the sample list into (rows, group_ids, is_augmented) for extraction."""
    rows: list[tuple[Path, int]] = []
    groups: list[int] = []
    augmented: list[bool] = []
    for gid, sample in enumerate(samples):
        rows.append(sample)
        groups.append(gid)
        augmented.append(False)
        for _ in range(aug_copies):
            rows.append(sample)
            groups.append(gid)
            augmented.append(True)
    return rows, groups, augmented


class FlaggedDataset(CanonicalDataset):
    """CanonicalDataset that augments only the rows marked as augmented copies.

    The plain `pil_transform` slot can't express this: the same file appears as
    both a clean row and an augmented row, so the decision is per row, not per
    dataset.
    """

    def __init__(self, samples, canonical_size, group_ids, aug_flags, augment):
        super().__init__(samples, canonical_size, pil_transform=None, group_ids=group_ids)
        self.aug_flags = list(aug_flags)
        self.augment = augment

    def __getitem__(self, idx: int):
        from PIL import Image

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.aug_flags[idx] and self.augment is not None:
            img = self.augment(img)
        return self.to_canonical(img), label, self.group_ids[idx], str(path)


@torch.no_grad()
def extract(stack: FeatureStack, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feats, labels, groups, paths = [], [], [], []
    for imgs, batch_labels, batch_groups, batch_paths in tqdm(loader, desc="extract"):
        feats.append(stack(imgs))
        labels.extend(batch_labels)
        groups.extend(batch_groups)
        paths.extend(batch_paths)
    return (
        np.concatenate(feats).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
        paths,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=None, help="Output .npz (defaults to the config's cache path)")
    parser.add_argument("--datasets", nargs="*", default=None, help="Override data.train_datasets")
    parser.add_argument("--data-root", default=None,
                        help="Override data.data_root (e.g. a node-local copy of the "
                             "dataset; see model_01/train.py --data_root)")
    parser.add_argument("--demo-eval-set", action="store_true", help="Extract the held-out demo set instead")
    parser.add_argument("--aug-copies", type=int, default=None, help="Override features.train_aug_copies")
    parser.add_argument(
        "--severity",
        default=None,
        help="Apply one fixed SEVERITY_LEVELS transform to every image (for robustness caches)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only extract the first N source images")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    aug_copies = args.aug_copies if args.aug_copies is not None else cfg["features"]["train_aug_copies"]
    if args.severity is not None and aug_copies:
        raise SystemExit(
            "--severity applies one fixed transform to every row; combining it with random "
            "augmented copies would make the cache's severity label meaningless. Pass --aug-copies 0."
        )

    if args.demo_eval_set:
        samples = demo_eval_samples(cfg)
    else:
        samples = build_labeled_samples(
            resolve_dataset_roots(cfg, args.datasets, args.data_root)
        )

    if args.limit is not None:
        rng = random.Random(seed)
        rng.shuffle(samples)
        samples = samples[: args.limit]
    samples.sort()  # deterministic row order regardless of how we got here

    rows, groups, aug_flags = build_rows(samples, aug_copies, seed)
    canonical_size = cfg["data"]["canonical_size"]

    if args.severity is not None:
        dataset = CanonicalDataset(
            rows, canonical_size, pil_transform=NamedSeverity(args.severity), group_ids=groups
        )
    else:
        augment = RobustnessAugment.from_config(cfg["augmentation"]) if aug_copies else None
        dataset = FlaggedDataset(rows, canonical_size, groups, aug_flags, augment)

    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_labeled,
    )

    stack = FeatureStack.from_config(cfg["features"], device=device)
    print(
        f"Extracting {len(rows)} rows ({len(samples)} images x {1 + aug_copies} copies) "
        f"-> {stack.dim} features on {device}"
    )
    for block in stack.blocks:
        print(f"  {block.name:6s} {block.dim:5d} dims  cols [{block.start}:{block.stop})")

    X, y, group_ids, paths = extract(stack, loader)

    out_path = Path(args.out) if args.out else Path(cfg["features"]["cache_dir"]) / cfg["features"]["cache_name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "features": stack.signature(),
        "feature_names": stack.feature_names(),
        "config": {"data": cfg["data"], "features": cfg["features"]},
        "aug_copies": aug_copies,
        "severity": args.severity,
        "n_source_images": len(samples),
        "demo_eval_set": bool(args.demo_eval_set),
    }
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        groups=group_ids,
        paths=np.asarray(paths),
        aug_flags=np.asarray(aug_flags, dtype=bool),
        meta=json.dumps(meta),
    )
    print(
        f"Wrote {X.shape[0]} x {X.shape[1]} features to {out_path} "
        f"({int((y == 0).sum())} real / {int((y == 1).sum())} fake rows)"
    )


if __name__ == "__main__":
    main()
