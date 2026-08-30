"""How should the likelihood map produce a real-vs-AI decision?

The objective is one binary call -- real or AI -- that holds up whether or not
the image has been degraded. Localisation is a means to it: the point of finding
regions is that AI content occupying part of a frame gets averaged into
invisibility by a whole-image score. So the question this script answers is not
"where is the edit" but "given the map, what number best separates real from AI".

That question was previously answered by hand: threshold the map at 0.75,
extract connected regions, noisy-OR their probabilities. Those constants were
chosen when the map carried no spatial signal at all, so there is no reason to
think they are right now that it does. Measured with the patch-trained backend,
they are visibly wrong -- every real image trips the threshold.

Rather than guess again, this caches each image's map once (the expensive part)
and then sweeps many ways of reducing it to a score, scoring each by the actual
objective:

    AUC(real vs AI)        the decision, over both AI subsets
    AUC on partially-AI    the subset a whole-image detector cannot see
    FPR at 80% recall      what it costs in false alarms on real photographs

Candidate reductions, from crudest to most structured:

    max / p99 / p95 / p90 / mean      order statistics of the map
    frac_above(t)                     how much of the frame is suspicious
    region_evidence(hi, min_area)     the current design: threshold, connect,
                                      take the strongest region's mass

The comparison is the point. If a percentile beats region extraction, then the
region machinery is not earning its place in the *decision* even if it is useful
for explanation, and that should be known rather than assumed.

    python scripts/fit_decision.py --backend checkpoints/patch_scorer --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import load_config  # noqa: E402
from eval.evaluate import auc  # noqa: E402
from regions.components import label_components  # noqa: E402

CACHE_SIDE = 512  # maps are cached at this resolution; fractions are unaffected


def build_cache(manifest_path: Path, backend: str, limit: int, out: Path) -> dict:
    from mapper.backends import build_backend
    from mapper.heatmap import AILikelihoodMapper

    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    items = manifest["items"]

    picked, seen = [], {}
    for row in items:
        if seen.get(row["class"], 0) < limit:
            picked.append(row)
            seen[row["class"]] = seen.get(row["class"], 0) + 1

    config = load_config()
    scorer = build_backend(backend, batch_size=64)
    mapper = AILikelihoodMapper(
        scorer=scorer,
        scales=config["mapper"]["scales"],
        overlap=config["mapper"]["overlap"],
        max_side=config["mapper"]["max_side"],
        scale_combine=config["mapper"]["scale_combine"],
        smoothing=config["mapper"]["smoothing"],
        smooth_radius=config["mapper"]["smooth_radius"],
    )

    try:
        from tqdm import tqdm

        iterator = tqdm(picked, desc="mapping", unit="img")
    except ImportError:
        iterator = picked

    heats, classes, whole = [], [], []
    for row in iterator:
        with Image.open(root / row["image"]) as handle:
            image = handle.convert("RGB")
        amap = mapper.run(image)
        heat = np.nan_to_num(amap.heat, nan=0.0).astype(np.float32)
        small = np.asarray(
            Image.fromarray(heat).resize((CACHE_SIDE, CACHE_SIDE), Image.BILINEAR),
            dtype=np.float16,
        )
        heats.append(small)
        classes.append(row["class"])
        whole.append(float(scorer.score_patches([amap.working_image])[0]))

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, heats=np.stack(heats), classes=np.array(classes), whole=np.array(whole)
    )
    return {"n": len(heats), "counts": seen}


def region_evidence(heat: np.ndarray, hi: float, min_area_frac: float) -> float:
    """The current design's reduction: strongest connected region's evidence mass."""
    mask = heat >= hi
    if not mask.any():
        return 0.0
    labels, count = label_components(mask, connectivity=8)
    if count == 0:
        return 0.0
    best = 0.0
    total = float(heat.size)
    for cid in range(1, count + 1):
        sel = labels == cid
        area = float(sel.sum()) / total
        if area < min_area_frac:
            continue
        best = max(best, float(heat[sel].mean()))
    return best


def evaluate(scores: np.ndarray, classes: np.ndarray) -> dict:
    real = scores[classes == "real"]
    full = scores[classes == "synthetic"]
    part = scores[classes == "tampered"]
    ai = np.concatenate([full, part])
    if real.size == 0 or ai.size == 0:
        return {}

    threshold = float(np.quantile(ai, 0.20))  # 80% recall on AI
    return {
        "auc_real_vs_ai": auc(ai, real),
        "auc_real_vs_fully_ai": auc(full, real) if full.size else float("nan"),
        "auc_real_vs_partially_ai": auc(part, real) if part.size else float("nan"),
        "fpr_at_80_recall": float((real >= threshold).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="eval_data/sid_set_val/manifest.json")
    parser.add_argument("--backend", default="checkpoints/patch_scorer")
    parser.add_argument("--limit", type=int, default=50, help="Images per class")
    parser.add_argument("--cache", default="eval_results/decision_cache.npz")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--out", default="eval_results/decision_sweep.json")
    args = parser.parse_args()

    cache = Path(args.cache)
    if args.rebuild or not cache.exists():
        info = build_cache(Path(args.manifest), args.backend, args.limit, cache)
        print(f"cached {info['n']} maps: {info['counts']}")

    data = np.load(cache, allow_pickle=True)
    heats = data["heats"].astype(np.float32)
    classes = data["classes"]
    whole = data["whole"]
    print(f"{len(heats)} maps  ({dict(zip(*np.unique(classes, return_counts=True)))})\n")

    results = {}

    # The baseline every reduction has to beat: ignore the map entirely.
    results["whole_image_only"] = evaluate(whole, classes)

    for name, fn in [
        ("max", lambda h: h.max()),
        ("p99", lambda h: np.percentile(h, 99)),
        ("p95", lambda h: np.percentile(h, 95)),
        ("p90", lambda h: np.percentile(h, 90)),
        ("mean", lambda h: h.mean()),
    ]:
        results[name] = evaluate(np.array([fn(h) for h in heats]), classes)

    for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
        results[f"frac_above_{t}"] = evaluate(
            np.array([(h >= t).mean() for h in heats]), classes
        )

    for hi in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
        for min_area in (0.002, 0.01, 0.04):
            key = f"region_hi{hi}_area{min_area}"
            results[key] = evaluate(
                np.array([region_evidence(h, hi, min_area) for h in heats]), classes
            )

    ranked = sorted(
        (k for k, v in results.items() if v),
        key=lambda k: -results[k]["auc_real_vs_ai"],
    )
    print(f"{'reduction':28s} {'AUC real/AI':>11s} {'partial':>8s} {'fully':>8s} {'FPR@80':>7s}")
    print("-" * 68)
    for key in ranked:
        v = results[key]
        print(f"{key:28s} {v['auc_real_vs_ai']:11.3f} {v['auc_real_vs_partially_ai']:8.3f} "
              f"{v['auc_real_vs_fully_ai']:8.3f} {v['fpr_at_80_recall']:7.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    best = ranked[0]
    print(f"best by AUC(real vs AI): {best}  ({results[best]['auc_real_vs_ai']:.3f}, "
          f"FPR@80 {results[best]['fpr_at_80_recall']:.3f})")


if __name__ == "__main__":
    main()
