"""Fit the mapper's calibrator, so its thresholds mean something.

The mapper thresholds a patch score to decide "likely AI" / "uncertain" /
"likely non-AI". Untouched, that is a cut on a number whose scale nobody
measured: both sibling detectors were trained on whole images and saturate hard,
so their patch scores are neither calibrated nor comparable between backends.
This script measures the correction.

    python scripts/calibrate_mapper.py --data_dir data/raw/cifake \
        --backend hf --out configs/calibration_sdxl_detector.json

`--data_dir` is a folder with `real/` and `fake/` subdirectories -- the same
layout model_01's data pipeline produces. Patches are sampled at the mapper's
own scales from each image, so what is fitted is the distribution the mapper
will actually see, not the distribution of whole images.

An important limitation, stated here because the numbers this produces are easy
to over-trust: a patch inherits the label of the image it came from. For a fully
generated image that is right -- every patch is generated. For an *edited* photo
it is wrong: most patches of a locally-inpainted image are authentic. So fitting
on whole-image datasets like CIFAKE calibrates the "generated vs. photographed"
axis, which is the axis both siblings were trained on, and does *not* calibrate
the "locally edited" case. Fitting that properly needs a dataset with tamper
masks, and the script accepts one via `--mask_dir`: when a mask is present, each
patch is labelled by whether its centre falls inside the tampered area.

Reports ECE before and after, on a held-out split, because a calibrator that
does not improve calibration is worth knowing about before it is deployed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapper.backends import build_backend  # noqa: E402
from mapper.calibration import expected_calibration_error, fit_isotonic, fit_platt  # noqa: E402
from mapper.windows import plan_windows  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path


def sample_patches(
    image: Image.Image,
    scales: list[int],
    per_image: int,
    rng: random.Random,
    mask: np.ndarray | None = None,
    image_label: int = 0,
):
    """Return [(patch, label)] sampled from one image's window grid."""
    plan = plan_windows(image.width, image.height, scales, overlap=0.5)
    windows = [w for group in plan.values() for w in group]
    if not windows:
        return []
    rng.shuffle(windows)

    out = []
    for window in windows[:per_image]:
        patch = image.crop(window.box)
        if mask is None:
            label = image_label
        else:
            cy = (window.y0 + window.y1) // 2
            cx = (window.x0 + window.x1) // 2
            label = int(bool(mask[min(cy, mask.shape[0] - 1), min(cx, mask.shape[1] - 1)]))
        out.append((patch, label))
    return out


def collect(
    data_dir: Path,
    mask_dir: Path | None,
    scales: list[int],
    per_image: int,
    max_images: int,
    seed: int,
):
    rng = random.Random(seed)
    patches, labels = [], []

    for class_name, image_label in (("real", 0), ("fake", 1)):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise SystemExit(
                f"Expected {class_dir} to exist. --data_dir must contain real/ and fake/ "
                f"subdirectories (the layout model_01's data pipeline produces)."
            )
        paths = list(iter_images(class_dir))
        rng.shuffle(paths)
        for path in paths[:max_images]:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            mask = None
            if mask_dir is not None:
                mask_path = mask_dir / f"{path.stem}.png"
                if mask_path.exists():
                    with Image.open(mask_path) as handle:
                        mask = np.asarray(handle.convert("L").resize(image.size)) > 127
            for patch, label in sample_patches(image, scales, per_image, rng, mask, image_label):
                patches.append(patch)
                labels.append(label)

    return patches, np.array(labels, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Folder with real/ and fake/ subdirectories")
    parser.add_argument("--mask_dir", default=None,
                        help="Optional tamper masks (<image_stem>.png, white = tampered). "
                             "With masks, patches are labelled by their centre pixel, which is "
                             "the only way to calibrate the locally-edited case honestly.")
    parser.add_argument(
        "--backend",
        default="hf",
        help="Patch scorer to calibrate: 'hf', 'hf:<hub model id>', a bare Hub id, "
             "or 'model_01' / 'model_02'. A calibrator is only valid for the backend "
             "it was fitted on -- the fit is recorded in the file's metadata.",
    )
    parser.add_argument("--checkpoint", default=None, help="model_01 / model_02 only")
    parser.add_argument("--scales", type=int, nargs="+", default=[64, 128, 224],
                        help="Must match the mapper's scales -- the fit describes the "
                             "patch distribution the mapper will actually produce")
    parser.add_argument("--per_image", type=int, default=6, help="Patches sampled per image")
    parser.add_argument("--max_images", type=int, default=400, help="Images per class")
    parser.add_argument("--method", default="platt", choices=["platt", "isotonic"])
    parser.add_argument("--val_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="configs/calibration.json")
    args = parser.parse_args()

    patches, labels = collect(
        Path(args.data_dir),
        Path(args.mask_dir) if args.mask_dir else None,
        args.scales,
        args.per_image,
        args.max_images,
        args.seed,
    )
    if len(patches) < 64:
        raise SystemExit(f"Only {len(patches)} patches collected; need at least 64 to fit anything.")
    print(f"collected {len(patches)} patches ({int(labels.sum())} positive)")

    kwargs = {}
    if args.checkpoint:
        kwargs["checkpoint"] = args.checkpoint
    scorer = build_backend(args.backend, **kwargs)

    print(f"scoring with {scorer.name} ...")
    scores = np.array(scorer.score_patches(patches), dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(scores))
    n_val = max(16, int(len(scores) * args.val_frac))
    val_idx, fit_idx = order[:n_val], order[n_val:]

    fit = fit_platt if args.method == "platt" else fit_isotonic
    calibrator = fit(scores[fit_idx], labels[fit_idx])

    before = expected_calibration_error(scores[val_idx], labels[val_idx])
    after = expected_calibration_error(calibrator.apply(scores[val_idx]), labels[val_idx])
    print(f"held-out ECE: {before:.4f} -> {after:.4f} ({len(val_idx)} patches)")
    if after > before:
        print(
            "WARNING: calibration made held-out ECE worse. Do not ship this file -- "
            "collect more patches, or try --method isotonic."
        )

    calibrator.meta.update(
        {
            "backend": scorer.name,
            "scales": args.scales,
            "method": args.method,
            "ece_before": before,
            "ece_after": after,
            "labelled_by": "tamper_mask" if args.mask_dir else "whole_image_label",
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(out_path)
    print(f"wrote {out_path}")
    print(json.dumps(calibrator.meta, indent=2))
    print("\nSet mapper.calibration_path in your config to this file to use it.")


if __name__ == "__main__":
    main()
