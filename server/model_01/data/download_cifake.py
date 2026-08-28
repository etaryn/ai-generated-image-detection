"""Downloads CIFAKE via kagglehub and reorganizes it into the
`data/raw/cifake/{real,fake}/` layout `data/datasets.py`'s `RealFakeImageDataset`
expects.

kagglehub is used instead of the raw `kaggle` CLI because it doesn't require
manually placing a `kaggle.json` API token -- it handles auth (prompting once,
then caching) and download caching itself.

CIFAKE ships as `train/{REAL,FAKE}` and `test/{REAL,FAKE}` (32x32 CIFAR-10-based
images: REAL = original CIFAR-10 photos, FAKE = Stable-Diffusion-generated
equivalents). We merge both splits together into one `real/` and one `fake/`
folder -- `RealFakeImageDataset.split_train_val` does its own train/val split on
top, so CIFAKE's own train/test split isn't load-bearing here.

Usage:
    python data/download_cifake.py --out data/raw/cifake
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DATASET_SLUG = "birdy654/cifake-real-and-ai-generated-synthetic-images"


def find_class_dirs(root: Path) -> dict[str, list[Path]]:
    """Finds every REAL/FAKE-named directory under `root` (case-insensitive,
    since Kaggle dataset layouts have been known to vary this across versions),
    keyed by lowercase label."""
    found: dict[str, list[Path]] = {"real": [], "fake": []}
    for path in root.rglob("*"):
        if path.is_dir() and path.name.upper() in ("REAL", "FAKE"):
            found[path.name.lower()].append(path)
    return found


def layout_from_source(source_root: Path, out_dir: Path, symlink: bool = True) -> dict[str, int]:
    """Copies (or symlinks) every image found under any REAL/FAKE directory in
    `source_root` into `out_dir/real/` or `out_dir/fake/`. Returns a
    {"real": n, "fake": n} count of files written, for tests/verification.

    Kept separate from the kagglehub download call itself so this reorganization
    logic can be unit-tested without needing network access or a real download.
    """
    class_dirs = find_class_dirs(source_root)
    if not class_dirs["real"] or not class_dirs["fake"]:
        raise RuntimeError(
            f"Could not find REAL/FAKE folders under {source_root} -- the dataset's "
            "internal layout may have changed; inspect it manually (`find <path> -type d`) "
            "and adjust find_class_dirs() if needed."
        )

    counts = {"real": 0, "fake": 0}
    for label, dirs in class_dirs.items():
        dest = out_dir / label
        dest.mkdir(parents=True, exist_ok=True)
        for src_dir in dirs:
            # Prefix with the parent folder name (train/test) so identically-named
            # files from both splits don't collide once merged.
            split_name = src_dir.parent.name
            for img_path in src_dir.iterdir():
                if not img_path.is_file():
                    continue
                dest_path = dest / f"{split_name}_{img_path.name}"
                if dest_path.exists():
                    continue
                if symlink:
                    try:
                        dest_path.symlink_to(img_path.resolve())
                    except OSError:
                        # e.g. Windows without symlink privilege/developer mode enabled.
                        shutil.copy(img_path, dest_path)
                else:
                    shutil.copy(img_path, dest_path)
                counts[label] += 1
    return counts


def download_and_layout(out_dir: Path, symlink: bool = True) -> dict[str, int]:
    import kagglehub

    print("Downloading CIFAKE via kagglehub (cached locally after the first run)...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"kagglehub cache path: {cache_path}")

    counts = layout_from_source(cache_path, out_dir, symlink=symlink)
    print(f"Wrote {counts['real']} real + {counts['fake']} fake images to {out_dir}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/cifake")
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlinking. Uses more disk but is safer if symlinks "
             "aren't available (e.g. Windows without Developer Mode / admin privileges).",
    )
    args = parser.parse_args()
    download_and_layout(Path(args.out), symlink=not args.copy)


if __name__ == "__main__":
    main()
