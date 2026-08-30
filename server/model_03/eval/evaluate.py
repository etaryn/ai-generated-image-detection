"""Does model_03 actually find where the edit is? Measured against ground truth.

Everything else in this project is either a unit test (does the machinery behave
as written?) or a demonstration (here is a heatmap, does it look right?).
Neither can answer the question the design rests on. This does, on SID-Set's
three classes, with the tamper masks as ground truth.

What is measured, and why each is here:

**Detection (image level).** AUC of the fused score, real vs AI, computed twice:
against all AI images, and against *tampered only*. The second is the number
that matters -- a whole-image detector can score well on fully-synthetic images
while being blind to a photograph with a small generated object in it, and
separating those two cases is model_03's entire reason to exist. Reported
alongside the backend's own whole-image score on the same images, so the
question "did the region machinery add anything, or is it just reporting what
the detector already said?" has an answer.

**Localisation (pixel level).** IoU, precision, recall and F1 of the union of
proposed regions against the ground-truth mask, plus hit rate at several IoU
thresholds. Stratified by how much of the frame the true edit covers, because a
sliding-window mapper's floor is set by its finest scale and an aggregate number
would hide that completely.

**False positives on authentic images.** How often a real photograph gets a
region proposed at all, and how much of the frame those spurious regions cover.
For a moderation use case this is the expensive error, and it is not visible in
any localisation metric.

**Verdict confusion.** The three-way verdict against the true class -- does a
tampered photo read as `ai_edited` rather than `ai_generated`, and vice versa?
No image-level detector can be scored on this at all.

**Routing behaviour.** Which specialist each class's regions get sent to. There
is no ground truth for routing, so this is descriptive, not scored: it says
whether the router is discriminating between the classes or sending everything
to one branch.

Usage:
    python eval/fetch_sid_set.py --per_class 120
    python eval/evaluate.py --data_dir eval_data/sid_set_val --out eval_results/sid_set.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import RegionAwareAnalyzer, load_config  # noqa: E402


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank-based AUC with ties counted as half. No sklearn dependency."""
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[order]

    ranks = np.empty(combined.size, dtype=np.float64)
    assigned = np.empty(combined.size, dtype=np.float64)
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        assigned[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks[order] = assigned

    n_pos = positive.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * negative.size))


def predicted_mask(report, size: tuple[int, int]) -> np.ndarray:
    """Union of the proposed regions, resampled to the original image size."""
    working = np.zeros(report.amap.labels.shape, dtype=bool)
    for finding in report.verdict.findings:
        working |= finding.region.mask
    if not working.any():
        return np.zeros((size[1], size[0]), dtype=bool)
    resized = Image.fromarray(working.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    return np.asarray(resized) > 127


def mask_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    inter = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    pred_area, true_area = float(pred.sum()), float(truth.sum())
    return {
        "iou": inter / union if union else 0.0,
        "precision": inter / pred_area if pred_area else 0.0,
        "recall": inter / true_area if true_area else 0.0,
        "f1": (2 * inter / (pred_area + true_area)) if (pred_area + true_area) else 0.0,
        "pred_area_frac": pred_area / pred.size,
        "true_area_frac": true_area / truth.size,
    }


def summarise(rows: list[dict]) -> dict:
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    scores = {name: np.array([r["score"] for r in items]) for name, items in by_class.items()}
    backend_scores = {
        name: np.array([r["whole_image_score"] for r in items]) for name, items in by_class.items()
    }
    real = scores.get("real", np.array([]))
    real_backend = backend_scores.get("real", np.array([]))

    ai_all = np.concatenate([scores.get("synthetic", np.array([])), scores.get("tampered", np.array([]))])
    ai_all_backend = np.concatenate(
        [backend_scores.get("synthetic", np.array([])), backend_scores.get("tampered", np.array([]))]
    )

    detection = {
        "auc_real_vs_all_ai": auc(ai_all, real),
        "auc_real_vs_tampered": auc(scores.get("tampered", np.array([])), real),
        "auc_real_vs_synthetic": auc(scores.get("synthetic", np.array([])), real),
        # The same three for the raw whole-image detector, so the region
        # machinery's contribution is visible rather than assumed.
        "backend_auc_real_vs_all_ai": auc(ai_all_backend, real_backend),
        "backend_auc_real_vs_tampered": auc(backend_scores.get("tampered", np.array([])), real_backend),
        "backend_auc_real_vs_synthetic": auc(backend_scores.get("synthetic", np.array([])), real_backend),
        "mean_score": {name: float(v.mean()) for name, v in scores.items()},
    }

    tampered = [r for r in by_class.get("tampered", []) if r.get("localisation")]
    loc_rows = [r["localisation"] for r in tampered]
    localisation = {}
    if loc_rows:
        ious = np.array([m["iou"] for m in loc_rows])
        localisation = {
            "n": len(loc_rows),
            "mean_iou": float(ious.mean()),
            "median_iou": float(np.median(ious)),
            "mean_f1": float(np.mean([m["f1"] for m in loc_rows])),
            "mean_precision": float(np.mean([m["precision"] for m in loc_rows])),
            "mean_recall": float(np.mean([m["recall"] for m in loc_rows])),
            "hit_rate_iou_0.10": float((ious >= 0.10).mean()),
            "hit_rate_iou_0.25": float((ious >= 0.25).mean()),
            "hit_rate_iou_0.50": float((ious >= 0.50).mean()),
            "any_region_proposed": float(np.mean([r["n_regions"] > 0 for r in tampered])),
            # Any overlap at all: "did it look in the right place", which is a
            # different and easier question than "did it get the extent right".
            "touch_rate": float(np.mean([m["recall"] > 0.05 for m in loc_rows])),
        }
        # Stratified by true edit size -- the finest window scale sets a floor,
        # and an aggregate mean would bury it.
        bands = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 1.01)]
        by_size = {}
        for low, high in bands:
            sel = [
                m for m in loc_rows if low <= m["true_area_frac"] < high
            ]
            if sel:
                by_size[f"{low:.2f}-{high:.2f}"] = {
                    "n": len(sel),
                    "mean_iou": float(np.mean([m["iou"] for m in sel])),
                    "mean_recall": float(np.mean([m["recall"] for m in sel])),
                    "touch_rate": float(np.mean([m["recall"] > 0.05 for m in sel])),
                }
        localisation["by_true_area_frac"] = by_size

    real_rows = by_class.get("real", [])
    false_positives = {
        "n": len(real_rows),
        "images_with_any_region": float(np.mean([r["n_regions"] > 0 for r in real_rows])) if real_rows else 0.0,
        "mean_regions": float(np.mean([r["n_regions"] for r in real_rows])) if real_rows else 0.0,
        "mean_flagged_area_frac": float(np.mean([r["pred_area_frac"] for r in real_rows])) if real_rows else 0.0,
        "verdict_not_authentic": float(
            np.mean([not r["verdict"].startswith("likely_authentic") for r in real_rows])
        ) if real_rows else 0.0,
    }

    confusion = {
        name: dict(Counter(r["verdict"] for r in items)) for name, items in by_class.items()
    }
    routing = {
        name: dict(Counter(route for r in items for route in r["routed_to"]))
        for name, items in by_class.items()
    }
    confidence = {name: float(np.mean([r["confidence"] for r in items])) for name, items in by_class.items()}

    return {
        "detection": detection,
        "localisation": localisation,
        "false_positives_on_real": false_positives,
        "verdict_confusion": confusion,
        "routing": routing,
        "mean_confidence": confidence,
        "mean_seconds_per_image": float(np.mean([r["seconds"] for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="eval_data/sid_set_val")
    parser.add_argument("--out", default="eval_results/sid_set.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Images per class (default: all)")
    parser.add_argument("--max_side", type=int, default=None, help="Override mapper.max_side")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    items = manifest["items"]

    if args.limit:
        kept, seen = [], Counter()
        for row in items:
            if seen[row["class"]] < args.limit:
                kept.append(row)
                seen[row["class"]] += 1
        items = kept

    config = load_config(args.config)
    if args.backend:
        config["backend"]["name"] = args.backend
        config["backend"]["model_id"] = None
    if args.max_side:
        config["mapper"]["max_side"] = args.max_side
    analyzer = RegionAwareAnalyzer(config)

    rows = []
    for i, row in enumerate(items, start=1):
        with Image.open(data_dir / row["image"]) as handle:
            image = handle.convert("RGB")

        start = time.perf_counter()
        report = analyzer.analyse(image)
        seconds = time.perf_counter() - start

        pred = predicted_mask(report, image.size)
        record = {
            "stem": row["stem"],
            "class": row["class"],
            "label": row["label"],
            "score": float(report.score),
            "whole_image_score": float(report.verdict.details["whole_image_score"]),
            "verdict": report.verdict.verdict,
            "confidence": float(report.verdict.confidence),
            "n_regions": len(report.verdict.findings),
            "routed_to": [f.route.primary for f in report.verdict.findings],
            "pred_area_frac": float(pred.mean()),
            "seconds": seconds,
        }

        if row.get("mask"):
            with Image.open(data_dir / row["mask"]) as handle:
                truth = np.asarray(handle.convert("L")) > 127
            record["localisation"] = mask_metrics(pred, truth)

        rows.append(record)
        if i % 10 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] {row['class']:9s} {report.verdict.verdict:24s} "
                  f"score={report.score:.3f} regions={len(report.verdict.findings)} {seconds:.1f}s",
                  flush=True)

    summary = summarise(rows)
    backend = rows and analyzer.scorer
    describe = getattr(analyzer.scorer, "describe", None)

    payload = {
        "dataset": manifest["summary"],
        "backend": describe() if describe else {"backend": getattr(analyzer.scorer, "name", "?")},
        "config": {"mapper": analyzer.config["mapper"], "regions": analyzer.config["regions"]},
        "n_images": len(rows),
        "summary": summary,
        "per_image": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
