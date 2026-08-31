"""Score a trained checkpoint against an already-extracted feature cache.

This is the other half of leave-one-generator-out. `train.py` reports metrics on a
validation split of the families it *trained* on; the number that actually matters
is performance on the family it never saw. That requires running a checkpoint over
a different cache than the one it was fitted to, which no existing script does:
`infer.py` starts from images and re-runs the frozen extractors, which for a
held-out family already sitting in a cache is pure waste (two ViT passes per image
to recompute numbers that are on disk).

So this reads the cache directly and applies exactly the preprocessing the
checkpoint was trained with -- the trained column selection, then the training
scaler -- via the same helper `infer.py` uses, so a score here and a score there
cannot drift apart.

By default only clean rows are scored. A training cache built with
`features.train_aug_copies > 0` holds augmented duplicates of every source image;
including them would report a number weighted toward whichever images happened to
get the harsher augmentation draw, and would not be comparable to a clean-data
evaluation. `--include-augmented` scores everything, and `--augmented-only` gives
the robustness-flavoured counterpart.

Usage:
    # the headline leave-one-out number: trained without dalle3, scored on it
    python eval/score_cache.py --checkpoint checkpoints/loo_dalle3.pt \\
        --cache features/cache/gen_dalle3.npz

    # several caches at once, each reported separately
    python eval/score_cache.py --checkpoint checkpoints/best.pt \\
        --cache features/cache/gen_*.npz --per-cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infer import apply_checkpoint_preprocessing  # noqa: E402
from classifiers import load_predictor  # noqa: E402
from shared import compute_all_metrics  # noqa: E402
from train import load_cache  # noqa: E402


def score_one(bundle: dict, predictor, cache: dict, subset: str) -> dict:
    aug = cache["aug_flags"].astype(bool)
    if subset == "clean":
        keep = ~aug
    elif subset == "augmented":
        keep = aug
    else:
        keep = np.ones(len(aug), dtype=bool)

    if not keep.any():
        raise RuntimeError(
            f"No rows left after selecting '{subset}' -- this cache has "
            f"{int(aug.sum())} augmented of {len(aug)} rows. Use --include-augmented."
        )

    X = apply_checkpoint_preprocessing(cache["X"][keep], bundle)
    y = cache["y"][keep]
    probs = predictor(X)

    threshold = bundle.get("threshold", 0.5)
    metrics = compute_all_metrics(y, probs, threshold)
    metrics["n"] = float(len(y))
    metrics["n_real"] = float((y == 0).sum())
    metrics["n_fake"] = float((y == 1).sum())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt to score with")
    parser.add_argument("--cache", nargs="+", required=True, help="Feature cache .npz file(s)")
    parser.add_argument("--per-cache", action="store_true",
                        help="Report each cache separately instead of pooling them")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include-augmented", action="store_true",
                       help="Score augmented rows as well as clean ones")
    group.add_argument("--augmented-only", action="store_true",
                       help="Score only the augmented rows (robustness-flavoured)")
    parser.add_argument("--json-out", default=None, help="Also write the metrics as JSON")
    args = parser.parse_args()

    subset = "all" if args.include_augmented else ("augmented" if args.augmented_only else "clean")

    # weights_only=False: the bundle carries the scaler arrays, the feature config
    # and (for xgboost) raw booster bytes, not just tensors.
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    predictor = load_predictor(bundle)

    trained_on = bundle.get("cache_paths") or [bundle.get("cache_path")]
    trained_names = {Path(str(c)).name for c in trained_on if c}
    print(f"checkpoint : {args.checkpoint}")
    print(f"  trained on: {', '.join(sorted(trained_names)) or 'unknown'}")
    print(f"  classifier: {bundle.get('classifier_type')} | rows scored: {subset}\n")

    results: dict[str, dict] = {}
    if args.per_cache:
        groups = [(p, [p]) for p in args.cache]
    else:
        groups = [("+".join(Path(p).stem for p in args.cache), args.cache)]

    for name, paths in groups:
        caches = [load_cache(p) for p in paths]
        merged = {
            "X": np.vstack([c["X"] for c in caches]),
            "y": np.concatenate([c["y"] for c in caches]),
            "aug_flags": np.concatenate([c["aug_flags"] for c in caches]),
        }
        metrics = score_one(bundle, predictor, merged, subset)

        # Whether this cache was part of training decides how the number reads:
        # in-distribution validation, or genuine held-out transfer.
        seen = Path(paths[0]).name in trained_names if len(paths) == 1 else None
        tag = "" if seen is None else ("  [SEEN in training]" if seen else "  [HELD OUT]")
        print(f"{Path(name).stem}{tag}")
        for key in ("accuracy", "balanced_accuracy", "f1", "auc", "fpr_at_threshold"):
            if key in metrics:
                print(f"    {key:20s} {metrics[key]:.4f}")
        print(f"    {'n':20s} {int(metrics['n'])} "
              f"({int(metrics['n_real'])} real / {int(metrics['n_fake'])} fake)\n")
        results[Path(name).stem] = metrics

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
