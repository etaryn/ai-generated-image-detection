"""Pull a bounded, manifest.json-described sample from CIFAKE for model_03.

CIFAKE (Bird & Lotfi, "CIFAKE: Image Classification and Explainable
Identification of AI-Generated Synthetic Images") is a **two-class** dataset:
32x32 CIFAR-10 photographs (`real`) and their Stable-Diffusion-generated
equivalents (`fake`, written here as `synthetic` to match `eval/evaluate.py`'s
and `eval/robustness.py`'s class vocabulary). There is no `tampered` class and
no mask -- every image is either wholly authentic or wholly generated, never a
real photo with a generated region in it. That is the opposite of SID-Set's
whole point (see `EVALUATION.md`), which is exactly why it's a useful second
dataset: it isolates the "wholly generated" case model_03's own numbers
already say localisation should *not* help on (EVALUATION_RESULTS.md S1's
synthetic row), with nothing partially-AI to blur the picture.

**Read the localisation arm's numbers here knowing this**: CIFAKE ships at a
native 32x32. `mapper/windows.py` does not upscale a small image up to a
working resolution -- it clamps the requested [64, 128, 224] scales down to
the image's own short side, and duplicate clamped scales collapse into one.
At 32px that leaves a *single* scale and a single window covering the whole
frame, so the multi-scale region machinery degenerates to "call the backend
once" by construction -- the same thing the whole-image arm already does. Any
gap between the two arms here comes only from `fuse()`'s max()-of-hypotheses
logic on that one region, not from spatial evidence. This is a stronger
version of the caveat `EVALUATION.md` already carries for CIFAKE
("CIFAKE is 32x32 and these are 224px models") -- worth restating rather than
letting a `+/-0.00x` delta be misread as "localisation doesn't help here
either", when the honest reading is "localisation had nothing to work with
here at all".

Usage:
    python eval/fetch_cifake.py --per_class 100 --out eval_data/cifake_val
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

DATASET_SLUG = "birdy654/cifake-real-and-ai-generated-synthetic-images"
CLASS_NAMES = {"real": "real", "fake": "synthetic"}


def find_class_dirs(root: Path) -> dict[str, list[Path]]:
    """Every REAL/FAKE-named directory under `root`, case-insensitive -- Kaggle
    dataset layouts have been known to vary this across versions. Mirrors
    `server/model_01/data/download_cifake.py`'s function of the same name;
    kept as its own copy so model_03's eval tooling has no cross-package
    import on model_01."""
    found: dict[str, list[Path]] = {"real": [], "fake": []}
    for path in root.rglob("*"):
        if path.is_dir() and path.name.upper() in ("REAL", "FAKE"):
            found[path.name.lower()].append(path)
    return found


def sample_and_layout(
    source_root: Path, out_dir: Path, per_class: int, seed: int = 0
) -> dict:
    """Samples `per_class` images per label from every REAL/FAKE dir found
    under `source_root` (train and test splits pooled, same as
    `download_cifake.py` -- CIFAKE's own train/test split isn't load-bearing
    for an eval sample), copies them into `out_dir/{real,synthetic}/`, and
    writes `manifest.json` in the schema `eval/evaluate.py` and
    `eval/robustness.py` read (the same one `eval/fetch_sid_set.py` writes,
    minus `mask` -- CIFAKE has none).

    Pure filesystem logic, no network -- testable against a synthetic tree
    the way `test_download_cifake.py` tests `layout_from_source`.
    """
    class_dirs = find_class_dirs(source_root)
    if not class_dirs["real"] or not class_dirs["fake"]:
        raise RuntimeError(
            f"Could not find REAL/FAKE folders under {source_root} -- inspect "
            "it manually (`find <path> -type d`) and adjust find_class_dirs() if needed."
        )

    rng = random.Random(seed)
    manifest: list[dict] = []
    counts = {"real": 0, "synthetic": 0}

    for label, dirs in class_dirs.items():
        name = CLASS_NAMES[label]
        dest_dir = out_dir / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        pool: list[Path] = []
        for src_dir in dirs:
            pool.extend(p for p in src_dir.iterdir() if p.is_file())
        rng.shuffle(pool)

        for src_path in pool[:per_class]:
            # Prefix with the split (train/test) folder name, same collision
            # avoidance as download_cifake.py, since both splits get pooled.
            split_name = src_path.parent.parent.name
            stem = f"{name}_{split_name}_{src_path.stem}".replace(" ", "_")
            dest_path = dest_dir / f"{stem}.png"

            with Image.open(src_path) as handle:
                image = handle.convert("RGB")
                image.save(dest_path)

            manifest.append(
                {
                    "stem": stem,
                    "label": 0 if name == "real" else 1,
                    "class": name,
                    "image": str(dest_path.relative_to(out_dir)).replace("\\", "/"),
                    "size": list(image.size),
                    "mask": None,
                }
            )
            counts[name] += 1

    summary = {
        "source": DATASET_SLUG,
        "counts": counts,
        "with_masks": 0,
        "note": "CIFAKE is two-class (real, synthetic) at native 32x32 -- no tampered "
                "class, no mask, and small enough that mapper/windows.py's multi-scale "
                "windowing collapses to a single whole-image window (see this script's "
                "module docstring). Train/test splits from the source are pooled.",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"summary": summary, "items": manifest}, indent=2))
    return summary


def download_source(cache_dir: str | None = None) -> Path:
    import kagglehub

    print("Downloading CIFAKE via kagglehub (cached locally after the first run)...")
    if cache_dir:
        import os

        os.environ.setdefault("KAGGLEHUB_CACHE", cache_dir)
    return Path(kagglehub.dataset_download(DATASET_SLUG))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="eval_data/cifake_val", help="Where to write the sample + manifest.json")
    parser.add_argument("--per_class", type=int, default=100, help="Images per class (real/synthetic)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed, for a reproducible draw")
    parser.add_argument(
        "--source", default=None,
        help="Path to an existing CIFAKE download (containing REAL/FAKE dirs somewhere "
             "under it, case-insensitive). Skips the kagglehub download when given.",
    )
    args = parser.parse_args()

    source_root = Path(args.source) if args.source else download_source()
    print(f"CIFAKE source: {source_root}")

    summary = sample_and_layout(source_root, Path(args.out), args.per_class, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
