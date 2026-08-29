"""Which of the three feature branches is actually doing the work?

The premise of model_02 is that DINOv2 (texture/structure), CLIP (semantics) and
the FFT statistics (pixel noise) contribute *different* evidence. That's a claim,
and this script tests it: it retrains the classifier on each branch alone and on
each combination, from the same cached features, and prints a table.

This is nearly free -- the expensive step (extraction) already happened, and each
retrain reads the same matrix with different columns -- so there's no excuse for
shipping the three-branch design without the numbers that justify it. If `clip`
alone matches `dino+clip+fft`, the extra two branches are inference cost with no
return, and the writeup should say so.

Usage:
    python eval/ablation.py --config configs/default.yaml
    python eval/ablation.py --config configs/default.yaml --classifier xgboost
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from shared import compute_all_metrics  # noqa: E402
from train import apply_scaler, fit_scaler, group_split, load_cache, select_blocks, train_classifier  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--classifier", default=None, choices=["mlp", "xgboost"])
    parser.add_argument(
        "--combos",
        nargs="*",
        default=None,
        help="Explicit combinations to test, comma-separated, e.g. --combos fft dino,clip dino,clip,fft "
             "(default: every non-empty subset of the cache's blocks)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["train"]["seed"]
    cache_path = args.cache or str(Path(cfg["features"]["cache_dir"]) / cfg["features"]["cache_name"])
    cache = load_cache(cache_path)
    X_all, y, groups = cache["X"], cache["y"], cache["groups"]
    block_spec = cache["meta"]["features"]["blocks"]
    all_names = cache["meta"]["feature_names"]
    available = [b["name"] for b in block_spec]

    if args.combos:
        combos = [c.split(",") for c in args.combos]
    else:
        combos = [
            list(combo)
            for r in range(1, len(available) + 1)
            for combo in itertools.combinations(available, r)
        ]

    train_idx, val_idx = group_split(groups, cfg["data"]["val_split"], seed)
    classifier_type = args.classifier or cfg["classifier"]["type"]
    threshold = cfg["eval"]["threshold"]

    rows = []
    for blocks in combos:
        print(f"\n{'=' * 70}\nblocks: {'+'.join(blocks)}\n{'=' * 70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        X, _, col_idx = select_blocks(X_all, block_spec, blocks)
        feature_names = [all_names[i] for i in col_idx]
        scaler = fit_scaler(X[train_idx])
        X_tr, X_va = apply_scaler(X[train_idx], scaler), apply_scaler(X[val_idx], scaler)

        payload, _ = train_classifier(
            cfg, classifier_type, X_tr, y[train_idx], X_va, y[val_idx], feature_names, seed
        )
        metrics = compute_all_metrics(y[val_idx], np.asarray(payload["val_probs"]), threshold)
        rows.append({"blocks": "+".join(blocks), "n_features": X.shape[1], **metrics})

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    out_dir = Path(cfg["eval"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"ablation_{classifier_type}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}\nablation summary ({classifier_type}, validation split)\n{'=' * 70}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
