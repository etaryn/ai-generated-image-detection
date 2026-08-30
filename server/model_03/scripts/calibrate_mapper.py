"""Fit the mapper's calibrators, so its thresholds mean something.

The mapper thresholds a patch score to decide "likely AI" / "uncertain" /
"likely non-AI". Untouched, that is a cut on a number nobody measured: the
public detectors were trained on whole images and saturate hard, so their patch
scores are neither calibrated nor comparable across scales or backends.

**Why this fits one calibrator per scale.** Measured on SID-Set with
Organika/sdxl-detector, the fraction of patches from *authentic photographs*
scoring above the nominal 0.75 threshold:

    64px    36.6%
    128px   16.3%
    224px   10.4%

while patches of fully-synthetic images scored ~0.71 at every scale. The fine
scale is not more sensitive to generated content; it is more prone to calling
authentic content generated, because a 64px crop blown up to the detector's
224px input looks smooth and textureless -- which is what these models were
trained to read as "generated". One shared calibrator cannot fix that: any
monotone map that pulls the fine scale's false positives down drags the coarse
scales' true positives down too. Three fits, one per scale, is the correction
the data calls for.

    python scripts/calibrate_mapper.py --manifest eval_data/sid_set_cal/manifest.json \
        --backend hf --out configs/calibration_sdxl_detector.json

**Fit on different images than you evaluate on.** The manifest given here should
not be the one eval/evaluate.py runs against -- fetch a second shard
(`eval/fetch_sid_set.py --shard 1`). Calibrating and scoring on the same images
would report the fit's training error as if it were a result.

**Patch labels.** With a SID-Set manifest, a patch's label is its own, not its
image's: real images contribute authentic patches, fully-synthetic images
contribute generated ones, and for tampered images each patch is labelled by
whether its centre falls inside the tamper mask. That last case is the whole
point -- calibrating on a two-class whole-image dataset teaches the map that
every patch of an edited photograph is generated, which is false for most of
them and biases exactly the thresholds routing depends on.

Reports held-out ECE per scale, before and after, because a calibrator that does
not improve calibration is worth knowing about before it ships.
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
from mapper.calibration import (  # noqa: E402
    Calibrator,
    ScaleCalibrators,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
)
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
    """Return [(patch, label, scale)] sampled from one image's window grid.

    Sampling is per scale rather than from a pooled list, so every scale gets
    roughly the same number of patches; a pooled shuffle would over-sample the
    fine scale simply because it has far more windows, and the coarse scales'
    fits would be the noisiest exactly where they are needed.
    """
    out = []
    plan = plan_windows(image.width, image.height, scales, overlap=0.5)
    per_scale = max(1, per_image // max(1, len(plan)))

    for scale, windows in plan.items():
        windows = list(windows)
        rng.shuffle(windows)
        for window in windows[:per_scale]:
            patch = image.crop(window.box)
            if mask is None:
                label = image_label
            else:
                cy = min((window.y0 + window.y1) // 2, mask.shape[0] - 1)
                cx = min((window.x0 + window.x1) // 2, mask.shape[1] - 1)
                label = int(bool(mask[cy, cx]))
            out.append((patch, label, scale))
    return out


def collect_from_manifest(manifest_path: Path, scales, per_image, max_images, seed):
    """Collect labelled patches from a SID-Set style manifest."""
    rng = random.Random(seed)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent

    items = list(manifest["items"])
    rng.shuffle(items)

    patches, labels, patch_scales = [], [], []
    counts: dict[str, int] = {}
    for row in items:
        cls = row["class"]
        if counts.get(cls, 0) >= max_images:
            continue

        with Image.open(root / row["image"]) as handle:
            image = handle.convert("RGB")

        mask, image_label = None, 0
        if cls == "synthetic":
            image_label = 1
        elif cls == "tampered" and row.get("mask"):
            with Image.open(root / row["mask"]) as handle:
                mask = np.asarray(handle.convert("L").resize(image.size)) > 127

        for patch, label, scale in sample_patches(image, scales, per_image, rng, mask, image_label):
            patches.append(patch)
            labels.append(label)
            patch_scales.append(scale)
        counts[cls] = counts.get(cls, 0) + 1

    print(f"sampled from {counts} images")
    return patches, np.array(labels, dtype=np.float64), np.array(patch_scales, dtype=int)


def collect_from_folders(data_dir: Path, mask_dir: Path | None, scales, per_image, max_images, seed):
    """Collect from a real/ + fake/ folder pair (the model_01 data layout)."""
    rng = random.Random(seed)
    patches, labels, patch_scales = [], [], []

    for class_name, image_label in (("real", 0), ("fake", 1)):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise SystemExit(
                f"Expected {class_dir} to exist. --data_dir must contain real/ and fake/ "
                f"subdirectories, or use --manifest."
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
            for patch, label, scale in sample_patches(image, scales, per_image, rng, mask, image_label):
                patches.append(patch)
                labels.append(label)
                patch_scales.append(scale)

    return patches, np.array(labels, dtype=np.float64), np.array(patch_scales, dtype=int)


def fit_one(scores, labels, method: str, min_samples: int) -> tuple[Calibrator | None, dict]:
    """Fit and score one calibrator, returning (calibrator or None, report)."""
    report = {"n": int(scores.size), "positives": int(labels.sum())}
    if scores.size < min_samples:
        report["skipped"] = f"only {scores.size} patches (need {min_samples})"
        return None, report
    if len(np.unique(labels)) < 2:
        report["skipped"] = "only one class present"
        return None, report

    rng = np.random.default_rng(0)
    order = rng.permutation(scores.size)
    n_val = max(16, int(scores.size * 0.3))
    val, fit_idx = order[:n_val], order[n_val:]

    try:
        fitter = fit_platt if method == "platt" else fit_isotonic
        calibrator = fitter(scores[fit_idx], labels[fit_idx])
    except ValueError as exc:
        report["skipped"] = f"fit failed: {exc}"
        return None, report

    before = expected_calibration_error(scores[val], labels[val])
    after = expected_calibration_error(calibrator.apply(scores[val]), labels[val])
    report.update({"ece_before": before, "ece_after": after, "improved": bool(after < before)})
    calibrator.meta.update(report)
    return calibrator, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=None,
                        help="A SID-Set style manifest.json from eval/fetch_sid_set.py (preferred)")
    parser.add_argument("--data_dir", default=None, help="Folder with real/ and fake/ subdirectories")
    parser.add_argument("--mask_dir", default=None, help="Tamper masks for --data_dir (<stem>.png)")
    parser.add_argument("--backend", default="hf",
                        help="'hf', 'hf:<hub id>', a bare Hub id, or 'model_01' / 'model_02'")
    parser.add_argument("--checkpoint", default=None, help="model_01 / model_02 only")
    parser.add_argument("--scales", type=int, nargs="+", default=[64, 128, 224],
                        help="Must match the mapper's scales")
    parser.add_argument("--per_image", type=int, default=9, help="Patches per image, split across scales")
    parser.add_argument("--max_images", type=int, default=400, help="Images per class")
    parser.add_argument("--method", default="isotonic", choices=["isotonic", "platt"],
                        help="isotonic by default: these detectors saturate into a bimodal "
                             "pile at 0 and 1, which a two-parameter logistic cannot "
                             "represent. Measured on SID-Set, platt made held-out ECE "
                             "WORSE at every scale (0.221 -> 0.273 at 128px) while isotonic "
                             "cut it by roughly two thirds (0.221 -> 0.080).")
    parser.add_argument("--min_samples", type=int, default=200, help="Minimum patches to fit one scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="configs/calibration.json")
    args = parser.parse_args()

    if not args.manifest and not args.data_dir:
        raise SystemExit("Pass --manifest (preferred) or --data_dir.")

    if args.manifest:
        patches, labels, scales = collect_from_manifest(
            Path(args.manifest), args.scales, args.per_image, args.max_images, args.seed
        )
    else:
        patches, labels, scales = collect_from_folders(
            Path(args.data_dir),
            Path(args.mask_dir) if args.mask_dir else None,
            args.scales, args.per_image, args.max_images, args.seed,
        )

    if len(patches) < args.min_samples:
        raise SystemExit(f"Only {len(patches)} patches collected; need at least {args.min_samples}.")
    print(f"collected {len(patches)} patches ({int(labels.sum())} positive)")

    kwargs = {"checkpoint": args.checkpoint} if args.checkpoint else {}
    scorer = build_backend(args.backend, **kwargs)
    print(f"scoring with {scorer.name} ...")

    try:
        from tqdm import tqdm

        chunks, size = [], 256
        for start in tqdm(range(0, len(patches), size), desc="scoring", unit="batch"):
            chunks.extend(scorer.score_patches(patches[start : start + size]))
        scores = np.array(chunks, dtype=np.float64)
    except ImportError:
        scores = np.array(scorer.score_patches(patches), dtype=np.float64)

    per_scale: dict[int, Calibrator] = {}
    reports: dict[str, dict] = {}
    print(f"\n{'scale':>8} {'n':>7} {'pos':>7} {'ECE before':>11} {'ECE after':>10}")
    for scale in sorted(set(scales.tolist())):
        sel = scales == scale
        calibrator, report = fit_one(scores[sel], labels[sel], args.method, args.min_samples)
        reports[str(scale)] = report
        if calibrator is not None:
            per_scale[scale] = calibrator
            print(f"{scale:>8} {report['n']:>7} {report['positives']:>7} "
                  f"{report['ece_before']:>11.4f} {report['ece_after']:>10.4f}"
                  f"{'' if report['improved'] else '   <-- WORSE, not shipped'}")
            if not report["improved"]:
                del per_scale[scale]
        else:
            print(f"{scale:>8} {report['n']:>7} {report['positives']:>7} "
                  f"{'--':>11} {'--':>10}   {report['skipped']}")

    # A shared fallback over all scales, for any scale the config adds later.
    # Held to the same standard as the per-scale fits: a calibrator that makes
    # held-out ECE worse is not a fallback, it is damage with a wider blast
    # radius, since it applies to every scale that was not fitted individually.
    shared, shared_report = fit_one(scores, labels, args.method, args.min_samples)
    reports["shared"] = shared_report
    if shared is not None and not shared_report.get("improved"):
        print(f"  shared fit rejected: ECE {shared_report['ece_before']:.4f} -> "
              f"{shared_report['ece_after']:.4f}")
        shared = None

    if not per_scale and shared is None:
        raise SystemExit(
            "Nothing could be fitted that improved held-out calibration, so no file was "
            "written -- running uncalibrated (and capped at 0.60 confidence) is the "
            "honest state, not a calibrator that makes the map worse. Try --method "
            "isotonic, or collect more patches."
        )

    calibrators = ScaleCalibrators(per_scale, shared or Calibrator.identity())
    calibrators.shared.meta.update({"backend": scorer.name, "method": args.method})
    for cal in per_scale.values():
        cal.meta.update({"backend": scorer.name, "method": args.method})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calibrators.save(out_path)

    print(f"\nwrote {out_path}")
    print(json.dumps({"scales_fitted": sorted(per_scale), "reports": reports}, indent=2))
    print("\nSet mapper.calibration_path to this file. It is valid only for "
          f"{scorer.name} at scales {sorted(set(scales.tolist()))}.")


if __name__ == "__main__":
    main()
