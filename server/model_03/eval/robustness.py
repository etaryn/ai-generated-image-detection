"""How the region-aware pipeline degrades under real-world redistribution.

model_01 and model_02 both report a transform x severity matrix, because the
challenge's premise is that a detector must survive JPEG recompression,
resizing, blur and noise. model_03 needs the same treatment, and needs it more:
its evidence is *local*, and every one of those transforms attacks local
statistics specifically.

The expected failure mode is worth stating before measuring it, so that the
measurement can contradict it. Heavy JPEG destroys the noise floor and imposes
its own 8x8 lattice everywhere; blur removes the fine detail the inpainting
specialist reads; downscaling averages away the blending seam. So the
prediction is that the specialists lose their evidence first, confidence falls,
and verdicts collapse toward `uncertain` -- which would be the *right* failure
(the system admitting it can no longer tell) rather than a dangerous one
(confidently changing its mind). A pipeline that kept the same confident
verdicts under heavy degradation would be the worrying result.

Three things are tracked per condition, because they can move independently:

* **AUC** -- can it still separate AI from authentic at all?
* **verdict stability** -- how often the verdict matches the same image's clean
  verdict. A detector whose score drifts but whose verdict holds is more useful
  than the reverse.
* **localisation recall and confidence** -- does it still find the edit, and
  does it correctly become less sure?

Usage:
    python eval/robustness.py --data_dir eval_data/sid_set_val --limit 15
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import RegionAwareAnalyzer, load_config  # noqa: E402
from eval.evaluate import auc, mask_metrics, predicted_mask  # noqa: E402


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def downscale(image: Image.Image, factor: float) -> Image.Image:
    """Resize down and back up -- a round trip, as redistribution actually does."""
    small = (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
    return image.resize(small, Image.BICUBIC).resize(image.size, Image.BICUBIC)


def noise(image: Image.Image, sigma: float) -> Image.Image:
    rng = np.random.default_rng(0)  # fixed, so conditions are comparable across images
    arr = np.asarray(image, dtype=np.float64)
    arr = arr + rng.normal(0.0, float(sigma) * 255.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def saturate(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(image).enhance(float(factor))


def crop(image: Image.Image, keep: float) -> Image.Image:
    w, h = image.size
    dx, dy = int(w * (1 - keep) / 2), int(h * (1 - keep) / 2)
    return image.crop((dx, dy, w - dx, h - dy))


# Severities run mild -> severe, matching the ranges the sibling models evaluate.
CONDITIONS: list[tuple[str, object, object]] = [
    ("clean", lambda im: im, None),
    *[(f"jpeg_q{q}", (lambda q: (lambda im: jpeg(im, q)))(q), q) for q in (90, 60, 30)],
    *[(f"blur_s{s}", (lambda s: (lambda im: blur(im, s)))(s), s) for s in (0.5, 1.0, 2.0)],
    *[(f"downscale_{f}", (lambda f: (lambda im: downscale(im, f)))(f), f) for f in (0.5, 0.25)],
    *[(f"noise_s{s}", (lambda s: (lambda im: noise(im, s)))(s), s) for s in (0.02, 0.05)],
    *[(f"crop_{k}", (lambda k: (lambda im: crop(im, k)))(k), k) for k in (0.9, 0.8)],
    ("saturate_1.5", lambda im: saturate(im, 1.5), 1.5),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="eval_data/sid_set_val")
    parser.add_argument("--out", default="eval_results/robustness.json")
    parser.add_argument("--limit", type=int, default=15, help="Images per class")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--max_side", type=int, default=768,
                        help="Lower than the shipped default to keep this run tractable; "
                             "the comparison across conditions is what matters here, and "
                             "every condition sees the same setting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    items, seen = [], Counter()
    for row in manifest["items"]:
        if seen[row["class"]] < args.limit:
            items.append(row)
            seen[row["class"]] += 1

    config = load_config()
    config["mapper"]["max_side"] = args.max_side
    if args.backend:
        config["backend"]["name"] = args.backend
        config["backend"]["model_id"] = None
    analyzer = RegionAwareAnalyzer(config)

    images = {}
    truths = {}
    for row in items:
        with Image.open(data_dir / row["image"]) as handle:
            images[row["stem"]] = handle.convert("RGB")
        if row.get("mask"):
            with Image.open(data_dir / row["mask"]) as handle:
                truths[row["stem"]] = np.asarray(handle.convert("L")) > 127

    results = {}
    clean_verdicts: dict[str, str] = {}

    for cond_name, transform, severity in CONDITIONS:
        rows = []
        for row in items:
            stem = row["stem"]
            image = transform(images[stem])
            report = analyzer.analyse(image)
            pred = predicted_mask(report, image.size)

            record = {
                "stem": stem,
                "class": row["class"],
                "score": float(report.score),
                "verdict": report.verdict.verdict,
                "confidence": float(report.verdict.confidence),
                "n_regions": len(report.verdict.findings),
            }
            if stem in truths:
                truth = truths[stem]
                if pred.shape != truth.shape:  # crop changes the frame
                    truth = np.asarray(
                        Image.fromarray(truth.astype(np.uint8) * 255).resize(
                            (pred.shape[1], pred.shape[0]), Image.NEAREST
                        )
                    ) > 127
                record["localisation"] = mask_metrics(pred, truth)
            rows.append(record)

        if cond_name == "clean":
            clean_verdicts = {r["stem"]: r["verdict"] for r in rows}

        real = np.array([r["score"] for r in rows if r["class"] == "real"])
        tampered = np.array([r["score"] for r in rows if r["class"] == "tampered"])
        synthetic = np.array([r["score"] for r in rows if r["class"] == "synthetic"])
        loc = [r["localisation"] for r in rows if "localisation" in r]

        results[cond_name] = {
            "severity": severity,
            "auc_real_vs_all_ai": auc(np.concatenate([tampered, synthetic]), real),
            "auc_real_vs_tampered": auc(tampered, real),
            "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
            "verdict_stability_vs_clean": float(
                np.mean([r["verdict"] == clean_verdicts.get(r["stem"]) for r in rows])
            ),
            "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "mean_localisation_recall": float(np.mean([m["recall"] for m in loc])) if loc else None,
            "mean_localisation_iou": float(np.mean([m["iou"] for m in loc])) if loc else None,
            "mean_regions": float(np.mean([r["n_regions"] for r in rows])),
        }
        r = results[cond_name]
        print(
            f"{cond_name:16s} AUC(all)={r['auc_real_vs_all_ai']:.3f} "
            f"AUC(tamp)={r['auc_real_vs_tampered']:.3f} "
            f"conf={r['mean_confidence']:.2f} "
            f"stable={r['verdict_stability_vs_clean']:.2f} "
            f"loc_recall={(r['mean_localisation_recall'] or 0):.3f}",
            flush=True,
        )

    describe = getattr(analyzer.scorer, "describe", None)
    payload = {
        "dataset": manifest["summary"],
        "backend": describe() if describe else {"backend": getattr(analyzer.scorer, "name", "?")},
        "images_per_class": args.limit,
        "max_side": args.max_side,
        "conditions": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
