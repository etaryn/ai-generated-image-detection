"""What is each feature branch worth *after* redistribution degradation?

`eval/ablation.py` answers "which branch carries the signal" on ONE cache -- i.e.
on pristine images. That is the wrong question for the failure we actually have.
Both papers summarised in improvement.md make the same claim: detectors lean on
local, high-frequency, generator-specific artifacts, and degradation destroys
exactly that cue. model_02's `fft` branch IS that cue in explicit form -- a
high-pass residual (image - blur(image)) reduced to radial/angular spectral bins.

So the testable prediction is not "fft is useful" but "fft's usefulness collapses
faster than dino's as degradation increases". A single-cache ablation cannot see
that; it needs the degradation axis. This script adds it:

    rows    = feature-block subsets (fft / dino / clip / dino+clip / all)
    columns = RealDeg-Bench resize strengths (clean, 0.9, 0.7, 0.5, 0.3, 0.2)
    cell    = balanced accuracy on the SAME held-out source images

Protocol: train once on CLEAN features, then score the degraded caches without
retraining. That is deliberate and it is the deployment question -- a model fitted
on pristine-ish training data meeting pixelated inputs at inference. Retraining per
condition (--retrain-per-condition) answers a different, weaker question ("is there
any signal left at this strength"), and is available for comparison.

The scaler is likewise fitted on clean training rows only and reused, because a
per-condition refit would silently absorb part of the degradation's effect --
standardising away the very distribution shift being measured.

Metric is balanced accuracy, per RealDeg-Bench: a detector whose cue has been
destroyed collapses to predicting one class, which raw accuracy rewards on any
unbalanced split.

Usage:
    python eval/fft_degradation_ablation.py --config configs/default.yaml \\
        --clean-cache features/cache/resize_clean.npz \\
        --degraded resize_0.9=features/cache/resize_0.9.npz \\
        --degraded resize_0.7=features/cache/resize_0.7.npz \\
        --degraded resize_0.5=features/cache/resize_0.5.npz \\
        --degraded resize_0.3=features/cache/resize_0.3.npz \\
        --degraded resize_0.2=features/cache/resize_0.2.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from classifiers import load_predictor  # noqa: E402
from shared import balanced_accuracy, compute_all_metrics  # noqa: E402
from train import (  # noqa: E402
    apply_scaler,
    fit_scaler,
    group_split,
    load_cache,
    select_blocks,
    train_classifier,
)

# The subsets that actually answer the question. `dino+clip` vs `dino+clip+fft` is
# the load-bearing pair: their difference is what the FFT branch is worth at that
# degradation level. `fft` alone shows the branch's own decay curve.
DEFAULT_COMBOS = [
    ["fft"],
    ["dino"],
    ["clip"],
    ["dino", "clip"],
    ["dino", "clip", "fft"],
]


def parse_degraded(specs: list[str]) -> list[tuple[str, str]]:
    """--degraded label=path/to.npz -> [(label, path)], order preserved."""
    out = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--degraded expects label=path, got {spec!r} "
                "(e.g. --degraded resize_0.3=features/cache/resize_0.3.npz)"
            )
        label, path = spec.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out


def check_alignment(clean: dict, other: dict, label: str) -> None:
    """A degraded cache must be the same images in the same row order.

    Without this the val indices computed on the clean cache would address
    different images in the degraded one, and every number below would be quietly
    meaningless. extract_features.py sorts its samples, so alignment holds when the
    caches were built with the same --limit and seed -- but that is a precondition
    worth enforcing rather than assuming.
    """
    if other["X"].shape[0] != clean["X"].shape[0]:
        raise SystemExit(
            f"cache {label!r} has {other['X'].shape[0]} rows, clean has {clean['X'].shape[0]}. "
            "Rebuild both with the same --limit and seed."
        )
    if not np.array_equal(np.asarray(other["paths"]), np.asarray(clean["paths"])):
        raise SystemExit(
            f"cache {label!r} holds different images (or a different row order) than the clean "
            "cache. Rebuild both with the same --limit and seed."
        )
    if not np.array_equal(other["y"], clean["y"]):
        raise SystemExit(f"cache {label!r} has different labels than the clean cache.")
    if other["meta"]["features"]["blocks"] != clean["meta"]["features"]["blocks"]:
        raise SystemExit(
            f"cache {label!r} has a different feature-block layout than the clean cache."
        )


def check_group_universe(train: dict, clean: dict) -> None:
    """A separate training cache need not be row-aligned, but it must cover the same
    source images, or the group-based train/val split would not partition the same set.

    Row counts legitimately differ (an --aug-copies 1 cache holds 2 rows per image),
    so alignment is checked on the group ids rather than the rows.
    """
    if train["meta"]["features"]["blocks"] != clean["meta"]["features"]["blocks"]:
        raise SystemExit("train cache has a different feature-block layout than the clean cache.")
    t, c = np.unique(train["groups"]), np.unique(clean["groups"])
    if not np.array_equal(t, c):
        raise SystemExit(
            f"train cache covers {len(t)} source images, clean cache {len(c)}, and the group ids "
            "differ. Rebuild both with the same --limit and seed."
        )


def describe_condition(cache: dict) -> str:
    """Human-readable degradation label, read back out of the cache's own metadata."""
    meta = cache["meta"]
    rd = meta.get("realdeg")
    if rd:
        return f"{rd['op']}@{rd['strength']:g}"
    if meta.get("severity"):
        return str(meta["severity"])
    return "clean"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--clean-cache", required=True, help="Clean (undegraded) feature cache")
    parser.add_argument(
        "--train-cache",
        default=None,
        help="Cache to FIT on (default: --clean-cache). Point this at an --aug-copies>0 cache to "
             "ask how much realistic mixture-augmentation recovers, rather than training clean-only. "
             "It may hold several rows per source image; it is matched to the eval caches by GROUP "
             "id, not by row, so it does not need to be row-aligned to them.",
    )
    parser.add_argument(
        "--degraded",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A degraded cache and the column label to give it. Repeatable.",
    )
    parser.add_argument("--classifier", default=None, choices=["mlp", "xgboost"])
    parser.add_argument(
        "--combos",
        nargs="*",
        default=None,
        help="Comma-separated block subsets, e.g. --combos fft dino,clip dino,clip,fft. "
             "Default: the five subsets that isolate the FFT branch's contribution.",
    )
    parser.add_argument(
        "--retrain-per-condition",
        action="store_true",
        help="Also refit on each degraded cache (train-degraded/test-degraded), answering "
             "'is any signal left' rather than 'does the learned rule survive'.",
    )
    parser.add_argument("--out-prefix", default="fft_degradation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["train"]["seed"]
    threshold = cfg["eval"]["threshold"]
    classifier_type = args.classifier or cfg["classifier"]["type"]
    combos = [c.split(",") for c in args.combos] if args.combos else DEFAULT_COMBOS

    clean = load_cache(args.clean_cache)
    block_spec = clean["meta"]["features"]["blocks"]
    all_names = clean["meta"]["feature_names"]
    available = [b["name"] for b in block_spec]
    for combo in combos:
        unknown = set(combo) - set(available)
        if unknown:
            raise SystemExit(f"unknown block(s) {sorted(unknown)}; cache has {available}")

    conditions: list[tuple[str, dict]] = [("clean", clean)]
    for label, path in parse_degraded(args.degraded):
        cache = load_cache(path)
        check_alignment(clean, cache, label)
        conditions.append((label, cache))

    # Split on source image, exactly as train.py does, so the held-out set here is
    # the held-out set everywhere else. The clean cache defines the eval rows; every
    # degraded cache is row-aligned to it (checked above).
    #
    # group_split derives the val GROUPS from the sorted unique group ids and the
    # seed, so running it on either cache selects the same held-out source images
    # even when the training cache has more rows per image.
    train_cache = clean
    if args.train_cache and args.train_cache != args.clean_cache:
        train_cache = load_cache(args.train_cache)
        check_group_universe(train_cache, clean)

    train_idx, _ = group_split(train_cache["groups"], cfg["data"]["val_split"], seed)
    _, val_idx = group_split(clean["groups"], cfg["data"]["val_split"], seed)
    y_train = train_cache["y"]
    y = clean["y"]
    y_val = y[val_idx]

    print(f"clean cache : {args.clean_cache}")
    print(f"train cache : {args.train_cache or args.clean_cache}"
          + (f"  (aug_copies={train_cache['meta'].get('aug_copies')})" if train_cache is not clean else ""))
    print(f"rows        : fit on {len(train_idx)} rows / score {len(val_idx)} rows, split by source image")
    print(f"val balance : {int((y_val == 0).sum())} real / {int((y_val == 1).sum())} fake")
    print(f"classifier  : {classifier_type}")
    print("conditions  : " + ", ".join(f"{lab} [{describe_condition(c)}]" for lab, c in conditions))
    print("combos      : " + ", ".join("+".join(c) for c in combos))

    trained_on_label = "clean" if train_cache is clean else "augmented"

    records = []
    for combo in combos:
        name = "+".join(combo)
        print(f"\n{'=' * 72}\nblocks: {name}  (fit on {trained_on_label.upper()}, score every condition)\n{'=' * 72}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        _, _, col_idx = select_blocks(clean["X"], block_spec, combo)
        feature_names = [all_names[i] for i in col_idx]

        X_fit = train_cache["X"][:, col_idx]
        scaler = fit_scaler(X_fit[train_idx])
        payload, _ = train_classifier(
            cfg,
            classifier_type,
            apply_scaler(X_fit[train_idx], scaler),
            y_train[train_idx],
            apply_scaler(clean["X"][:, col_idx][val_idx], scaler),
            y_val,
            feature_names,
            seed,
        )
        payload.pop("val_probs", None)
        predictor = load_predictor(payload)

        for label, cache in conditions:
            X_cond = cache["X"][:, col_idx]
            probs = predictor(apply_scaler(X_cond[val_idx], scaler))
            metrics = compute_all_metrics(y_val, probs, threshold)
            bacc = balanced_accuracy(y_val, (probs >= threshold).astype(int))
            records.append(
                {
                    "blocks": name,
                    "condition": label,
                    "degradation": describe_condition(cache),
                    "trained_on": trained_on_label,
                    "balanced_accuracy": bacc,
                    "auc": metrics["auc"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "mean_prob": float(np.mean(probs)),
                    "pred_std": float(np.std(probs)),
                }
            )
            print(
                f"  {label:14s} bacc={bacc:.4f} auc={metrics['auc']:.4f} "
                f"acc={metrics['accuracy']:.4f} pred_std={np.std(probs):.4f}"
            )

        if args.retrain_per_condition:
            for label, cache in conditions:
                if label == "clean":
                    continue
                torch.manual_seed(seed)
                np.random.seed(seed)
                X_cond = cache["X"][:, col_idx]
                cond_train_idx, _ = group_split(cache["groups"], cfg["data"]["val_split"], seed)
                sc = fit_scaler(X_cond[cond_train_idx])
                pay, _ = train_classifier(
                    cfg,
                    classifier_type,
                    apply_scaler(X_cond[cond_train_idx], sc),
                    y[cond_train_idx],
                    apply_scaler(X_cond[val_idx], sc),
                    y_val,
                    feature_names,
                    seed,
                )
                probs = np.asarray(pay["val_probs"])
                metrics = compute_all_metrics(y_val, probs, threshold)
                records.append(
                    {
                        "blocks": name,
                        "condition": label,
                        "degradation": describe_condition(cache),
                        "trained_on": "degraded",
                        "balanced_accuracy": balanced_accuracy(y_val, (probs >= threshold).astype(int)),
                        "auc": metrics["auc"],
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "mean_prob": float(np.mean(probs)),
                        "pred_std": float(np.std(probs)),
                    }
                )

    df = pd.DataFrame(records)
    out_dir = Path(cfg["eval"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out_prefix}_{classifier_type}.csv"
    df.to_csv(csv_path, index=False)

    order = [label for label, _ in conditions]
    clean_trained = df[df["trained_on"] == trained_on_label]

    for metric in ("balanced_accuracy", "auc"):
        pivot = (
            clean_trained.pivot(index="blocks", columns="condition", values=metric)
            .reindex(columns=order)
            .reindex(["+".join(c) for c in combos])
        )
        print(f"\n{'=' * 72}\n{metric} -- trained on CLEAN, scored per condition\n{'=' * 72}")
        print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    # The headline number: what the FFT branch adds on top of dino+clip, per
    # condition. If the local-artifact hypothesis holds, this shrinks toward zero
    # (or goes negative) as the resize scale drops.
    pivot_b = (
        clean_trained.pivot(index="blocks", columns="condition", values="balanced_accuracy")
        .reindex(columns=order)
    )
    if {"dino+clip", "dino+clip+fft"}.issubset(set(pivot_b.index)):
        delta = pivot_b.loc["dino+clip+fft"] - pivot_b.loc["dino+clip"]
        print(f"\n{'=' * 72}\nFFT branch contribution: bacc(dino+clip+fft) - bacc(dino+clip)\n{'=' * 72}")
        print(delta.to_string(float_format=lambda v: f"{v:+.4f}"))

    if "fft" in pivot_b.index and "clean" in pivot_b.columns:
        fft_row = pivot_b.loc["fft"]
        print(f"\n{'=' * 72}\nfft-alone decay (balanced accuracy, and drop from clean)\n{'=' * 72}")
        for cond in order:
            print(f"  {cond:14s} {fft_row[cond]:.4f}  ({fft_row[cond] - fft_row['clean']:+.4f})")
        if "dino" in pivot_b.index:
            dino_row = pivot_b.loc["dino"]
            print("\n  for comparison, dino-alone:")
            for cond in order:
                print(f"  {cond:14s} {dino_row[cond]:.4f}  ({dino_row[cond] - dino_row['clean']:+.4f})")

    print(f"\nWrote {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for blocks in pivot_b.index:
            ax.plot(order, [pivot_b.loc[blocks, c] for c in order], marker="o", label=blocks)
        ax.axhline(0.5, ls="--", lw=1, c="grey", label="chance")
        ax.set_xlabel("RealDeg resize condition")
        ax.set_ylabel("balanced accuracy (held-out images)")
        ax.set_title("model_02: per-branch decay under RealDeg resize sweep\n(trained on clean features)")
        ax.legend()
        fig.tight_layout()
        png_path = out_dir / f"{args.out_prefix}_{classifier_type}.png"
        fig.savefig(png_path, dpi=150)
        print(f"Wrote {png_path}")
    except ImportError as exc:
        print(f"(skipping plot: {exc})")


if __name__ == "__main__":
    main()
