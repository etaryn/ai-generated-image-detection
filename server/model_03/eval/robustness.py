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


def _bar(total: int):
    """One bar across the whole matrix, since per-condition bars flicker past."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc="robustness", unit="run", dynamic_ncols=True)


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


def _class_stats(rows: list[dict]) -> dict | None:
    """Per-class view of how much of the frame the map lit up.

    `frac_regions_fired` on the real rows is the false-positive rate of the
    localisation stage: every real image that grows a region is an image the
    local hypothesis can push above the decision threshold on its own.
    """
    if not rows:
        return None
    return {
        "n": len(rows),
        "mean_score": float(np.mean([r["score"] for r in rows])),
        "mean_whole_image_score": float(np.mean([r["whole_image_score"] for r in rows])),
        "mean_regions": float(np.mean([r["n_regions"] for r in rows])),
        "frac_regions_fired": float(np.mean([r["n_regions"] > 0 for r in rows])),
        "mean_map_score": float(np.mean([r["map"]["mean_score"] for r in rows])),
        "mean_map_p95": float(np.mean([r["map"]["p95_score"] for r in rows])),
        "mean_map_median": float(np.mean([r["map"].get("median_score", float("nan")) for r in rows])),
        "mean_frac_likely_ai": float(np.mean([r["map"]["frac_likely_ai"] for r in rows])),
        "mean_frac_uncertain": float(np.mean([r["map"]["frac_uncertain"] for r in rows])),
        "mean_frac_likely_non_ai": float(np.mean([r["map"]["frac_likely_non_ai"] for r in rows])),
        "mean_confidence_uncapped": float(np.mean([r["confidence_uncapped"] for r in rows])),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="eval_data/sid_set_val")
    parser.add_argument("--out", default="eval_results/robustness.json")
    parser.add_argument("--limit", type=int, default=15, help="Images per class")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--dual", action="store_true",
                        help="Wrap the pipeline in DualBackendAnalyzer: fall back to the "
                             "whole-image pathway when the trust signal says localisation "
                             "is unreliable for this image")
    parser.add_argument("--trust_threshold", type=float, default=None,
                        help="--dual only: trust signal cut. Default is the value frozen "
                             "in dual_backend.py, tuned on shard 4")
    parser.add_argument("--fallback_backend", default=None,
                        help="--dual only: 'self' (default) reuses the primary's own "
                             "whole-image pass; a Hub id routes to a separate model")
    parser.add_argument("--threshold_mode", default=None,
                        choices=["absolute", "quantile", "median_shift"],
                        help="How the map's AI/non-AI cuts are placed. 'absolute' is the "
                             "shipped default; the adaptive modes remove the common-mode "
                             "distribution shift that degradation induces (mapper/heatmap.py)")
    parser.add_argument("--adaptive_ref_median", type=float, default=None,
                        help="median_shift only: the clean-corpus map median to correct toward")
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
    if args.threshold_mode:
        config["mapper"]["threshold_mode"] = args.threshold_mode
    if args.adaptive_ref_median is not None:
        config["mapper"]["adaptive_ref_median"] = args.adaptive_ref_median
    if args.backend:
        config["backend"]["name"] = args.backend
        config["backend"]["model_id"] = None
    analyzer = RegionAwareAnalyzer(config)
    dual = None
    if args.dual:
        from dual_backend import DualBackendAnalyzer

        dual_cfg = {"trust": {}, "fallback": {}}
        if args.trust_threshold is not None:
            dual_cfg["trust"]["threshold"] = args.trust_threshold
        if args.fallback_backend is not None:
            dual_cfg["fallback"]["backend"] = args.fallback_backend
        # Score both arms on every image so the gated and ungated numbers come
        # from one pass over the same frames.
        dual_cfg["fallback"]["eager"] = True
        dual = DualBackendAnalyzer(dual_cfg, primary=analyzer)

    images = {}
    truths = {}
    for row in items:
        with Image.open(data_dir / row["image"]) as handle:
            images[row["stem"]] = handle.convert("RGB")
        if row.get("mask"):
            with Image.open(data_dir / row["mask"]) as handle:
                truths[row["stem"]] = np.asarray(handle.convert("L")) > 127

    results = {}
    per_image: dict[str, list[dict]] = {}
    clean_verdicts: dict[str, str] = {}

    bar = _bar(len(CONDITIONS) * len(items))

    for cond_name, transform, severity in CONDITIONS:
        rows = []
        for row in items:
            stem = row["stem"]
            image = transform(images[stem])
            if dual is not None:
                outcome = dual.analyse(image)
                report = outcome.report
            else:
                outcome, report = None, analyzer.analyse(image)
            pred = predicted_mask(report, image.size)
            if outcome is not None and not outcome.trusted:
                # The gate declined to believe the regions, so the report does
                # not show them -- and localisation must be scored on what the
                # system actually reported, not on what it privately computed.
                pred = np.zeros_like(pred)
            if bar is not None:
                bar.set_postfix({"condition": cond_name}, refresh=False)
                bar.update(1)

            record = {
                "stem": stem,
                "class": row["class"],
                "score": float(outcome.score) if outcome is not None else float(report.score),
                # fuse() computes this unconditionally as one whole-image backend
                # call, whether or not any region fires -- so recording it costs
                # nothing extra and gives the with/without-localisation pairing
                # eval/ablation.py does on clean data, per condition here too.
                "whole_image_score": float(report.verdict.details["whole_image_score"]),
                "verdict": report.verdict.verdict,
                "confidence": float(report.verdict.confidence),
                # The capped value saturates at UNCALIBRATED_CONFIDENCE_CAP on an
                # uncalibrated map, which is every run that ships. Without the
                # pre-cap number this experiment cannot test its own stated
                # hypothesis -- that confidence falls as the evidence degrades.
                "confidence_uncapped": float(
                    report.verdict.details.get("confidence_uncapped", report.verdict.confidence)
                ),
                "confidence_capped": bool(
                    report.verdict.details.get("confidence_capped_by_calibration", False)
                ),
                "n_regions": 0 if (outcome is not None and not outcome.trusted)
                             else len(report.verdict.findings),
                # The map's own score distribution. Region proposal thresholds are
                # absolute (threshold_lo/hi), so a degradation that shifts this
                # distribution bodily up or down changes how much of the frame
                # fires without the image's content changing at all. That is the
                # difference between a detector that degrades and one that inverts.
                "map": dict(report.verdict.details["map"]),
            }
            if outcome is not None:
                record["dual"] = {
                    "trusted": bool(outcome.trusted),
                    "source": outcome.source,
                    "signal_value": float(outcome.signal_value),
                    "fallback_score": outcome.fallback_score,
                    "ungated_score": float(report.score),
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
        per_image[cond_name] = rows

        real = np.array([r["score"] for r in rows if r["class"] == "real"])
        tampered = np.array([r["score"] for r in rows if r["class"] == "tampered"])
        synthetic = np.array([r["score"] for r in rows if r["class"] == "synthetic"])
        loc = [r["localisation"] for r in rows if "localisation" in r]

        whole_real = np.array([r["whole_image_score"] for r in rows if r["class"] == "real"])
        whole_tampered = np.array([r["whole_image_score"] for r in rows if r["class"] == "tampered"])
        whole_synthetic = np.array([r["whole_image_score"] for r in rows if r["class"] == "synthetic"])

        results[cond_name] = {
            "severity": severity,
            "auc_real_vs_all_ai": auc(np.concatenate([tampered, synthetic]), real),
            "auc_real_vs_tampered": auc(tampered, real),
            # "without localisation" arm: the same backend, same image, scored once
            # as a whole -- exactly what eval/ablation.py compares on clean data,
            # here per degradation condition.
            "auc_real_vs_all_ai_whole_image": auc(np.concatenate([whole_tampered, whole_synthetic]), whole_real),
            "auc_real_vs_tampered_whole_image": auc(whole_tampered, whole_real),
            "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
            "verdict_stability_vs_clean": float(
                np.mean([r["verdict"] == clean_verdicts.get(r["stem"]) for r in rows])
            ),
            "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "mean_localisation_recall": float(np.mean([m["recall"] for m in loc])) if loc else None,
            "mean_localisation_iou": float(np.mean([m["iou"] for m in loc])) if loc else None,
            "mean_regions": float(np.mean([r["n_regions"] for r in rows])),
            "frac_distrusted": (float(np.mean([r["dual"]["trusted"] is False for r in rows]))
                                if "dual" in rows[0] else None),
            "mean_confidence_uncapped": float(np.mean([r["confidence_uncapped"] for r in rows])),
            "frac_confidence_capped": float(np.mean([r["confidence_capped"] for r in rows])),
            # Split by class, because the aggregate hides the failure that
            # matters: regions firing on *real* images is what destroys AUC,
            # and it looks identical to regions firing on tampered ones until
            # the two are separated.
            "by_class": {
                cls: _class_stats([r for r in rows if r["class"] == cls])
                for cls in ("real", "tampered", "synthetic")
            },
        }
        r = results[cond_name]
        print(
            f"{cond_name:16s} AUC(all)={r['auc_real_vs_all_ai']:.3f} "
            f"AUC(tamp)={r['auc_real_vs_tampered']:.3f} "
            f"[no-loc AUC(tamp)={r['auc_real_vs_tampered_whole_image']:.3f}] "
            f"conf={r['mean_confidence']:.2f} "
            f"stable={r['verdict_stability_vs_clean']:.2f} "
            f"loc_recall={(r['mean_localisation_recall'] or 0):.3f}",
            flush=True,
        )

    if bar is not None:
        bar.close()

    describe = getattr(analyzer.scorer, "describe", None)
    payload = {
        "dataset": manifest["summary"],
        "backend": describe() if describe else {"backend": getattr(analyzer.scorer, "name", "?")},
        "images_per_class": args.limit,
        "max_side": args.max_side,
        "dual": bool(args.dual),
        "trust_threshold": args.trust_threshold,
        "fallback_backend": args.fallback_backend,
        "threshold_mode": args.threshold_mode or "absolute",
        "adaptive_ref_median": args.adaptive_ref_median,
        "conditions": results,
        # Kept so a follow-up question about *which* images moved can be answered
        # from the artefact instead of costing another GPU hour.
        "per_image": per_image,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
