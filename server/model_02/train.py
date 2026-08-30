"""Step 2 driver: train the classifier on a cached feature matrix.

Reads the .npz written by extract_features.py, splits it, standardizes it, fits
either the MLP or XGBoost, and writes a single self-contained checkpoint holding
everything infer.py needs: the scaler, the classifier, the feature config the
cache was built with, and the block layout.

Because Step 1 is frozen and cached, this script is cheap to re-run -- change the
classifier, the hyperparameters, or the feature subset and retrain in minutes
without touching a single image.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --classifier xgboost
    python train.py --config configs/default.yaml --blocks fft        # FFT-only ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from shared import compute_all_metrics, threshold_for_target_fpr

EPS = 1e-8


def load_cache(path: str | Path) -> dict:
    """Load a feature cache written by extract_features.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature cache {path} not found. Run extract_features.py first:\n"
            f"    python extract_features.py --config configs/default.yaml --out {path}"
        )
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "groups": data["groups"].astype(np.int64),
        "paths": data["paths"],
        "aug_flags": data["aug_flags"],
        "meta": meta,
    }


def group_split(groups: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by source image, not by row.

    Augmented copies share their original's group id, so splitting on raw rows
    would put a JPEG-recompressed copy of a training image into validation and
    report a val score inflated by near-duplicate leakage.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_val = int(len(shuffled) * val_fraction)
    val_groups = set(shuffled[:n_val].tolist())
    is_val = np.array([g in val_groups for g in groups])
    return np.where(~is_val)[0], np.where(is_val)[0]


def fit_scaler(X: np.ndarray) -> dict:
    """Per-column mean/std from the training rows only (never the full matrix --
    that would leak validation statistics into the fitted scaler)."""
    return {"mean": X.mean(axis=0), "std": X.std(axis=0) + EPS}


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    return ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)


def select_blocks(X: np.ndarray, block_spec: list[dict], names: list[str] | None):
    """Keep only the named feature blocks (used for ablations and FFT-only runs).

    Returns (X_subset, kept_block_spec_renumbered, column_indices).
    """
    if not names:
        return X, block_spec, np.arange(X.shape[1])

    known = {b["name"] for b in block_spec}
    unknown = set(names) - known
    if unknown:
        raise ValueError(f"Unknown feature block(s) {sorted(unknown)}; cache has {sorted(known)}")

    cols, kept, offset = [], [], 0
    for block in block_spec:
        if block["name"] in names:
            cols.append(np.arange(block["start"], block["stop"]))
            kept.append({"name": block["name"], "dim": block["dim"], "start": offset, "stop": offset + block["dim"]})
            offset += block["dim"]
    col_idx = np.concatenate(cols)
    return X[:, col_idx], kept, col_idx


def train_classifier(cfg: dict, classifier_type: str, X_tr, y_tr, X_va, y_va, feature_names, seed):
    threshold = cfg["eval"]["threshold"]
    if classifier_type == "mlp":
        from classifiers.mlp import train_mlp

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return train_mlp(X_tr, y_tr, X_va, y_va, cfg["classifier"]["mlp"], threshold, device, seed)
    if classifier_type == "xgboost":
        from classifiers.xgb import train_xgb

        return train_xgb(
            X_tr, y_tr, X_va, y_va, cfg["classifier"]["xgboost"], threshold, seed, feature_names
        )
    raise ValueError(f"classifier.type must be 'mlp' or 'xgboost', got {classifier_type!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", default=None, help="Feature cache .npz (defaults to the config's path)")
    parser.add_argument("--classifier", default=None, choices=["mlp", "xgboost"], help="Override classifier.type")
    parser.add_argument(
        "--blocks",
        nargs="*",
        default=None,
        help="Train on a subset of feature blocks, e.g. --blocks dino fft (default: all)",
    )
    parser.add_argument("--out", default=None, help="Checkpoint path (default: <checkpoint_dir>/best.pt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cache_path = args.cache or str(Path(cfg["features"]["cache_dir"]) / cfg["features"]["cache_name"])
    cache = load_cache(cache_path)
    X, y, groups = cache["X"], cache["y"], cache["groups"]
    block_spec = cache["meta"]["features"]["blocks"]
    feature_names = cache["meta"]["feature_names"]

    X, block_spec, col_idx = select_blocks(X, block_spec, args.blocks)
    feature_names = [feature_names[i] for i in col_idx]

    train_idx, val_idx = group_split(groups, cfg["data"]["val_split"], seed)
    scaler = fit_scaler(X[train_idx])
    X_tr, X_va = apply_scaler(X[train_idx], scaler), apply_scaler(X[val_idx], scaler)
    y_tr, y_va = y[train_idx], y[val_idx]

    classifier_type = args.classifier or cfg["classifier"]["type"]
    print(
        f"cache: {cache_path}\n"
        f"rows: {len(X)} ({len(train_idx)} train / {len(val_idx)} val, split by source image)\n"
        f"features: {X.shape[1]} in blocks " + ", ".join(f"{b['name']}({b['dim']})" for b in block_spec) + "\n"
        f"classifier: {classifier_type}"
    )

    payload, history = train_classifier(
        cfg, classifier_type, X_tr, y_tr, X_va, y_va, feature_names, seed
    )

    val_probs = np.asarray(payload.pop("val_probs"))
    metrics = compute_all_metrics(y_va, val_probs, cfg["eval"]["threshold"])
    # A moderation deployment cares more about not flagging real content than
    # about the default 0.5 cutoff, so calibrate and record an FPR-budget
    # threshold alongside the default one (see model_01's README on this trade-off).
    fpr_budget = cfg["eval"].get("target_fpr", 0.05)
    calibrated = threshold_for_target_fpr(y_va, val_probs, fpr_budget)

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else ckpt_dir / "best.pt"

    bundle = {
        **payload,
        "scaler": scaler,
        "block_spec": block_spec,
        "feature_names": feature_names,
        "features_config": cache["meta"]["config"]["features"],
        "resolved_backbones": cache["meta"].get("resolved_backbones"),
        "canonical_size": cache["meta"]["config"]["data"]["canonical_size"],
        "blocks_used": args.blocks,
        # Column indices into the FULL extracted vector. infer.py always runs the
        # complete feature stack and then selects these, so an ablation checkpoint
        # (--blocks fft) stays runnable without a second, differently-shaped cache.
        "feature_columns": col_idx.tolist(),
        "config": cfg,
        "cache_path": str(cache_path),
        "val_metrics": metrics,
        "threshold": cfg["eval"]["threshold"],
        "calibrated_threshold": {"target_fpr": fpr_budget, "threshold": calibrated},
    }
    torch.save(bundle, out_path)

    log_path = ckpt_dir / "training_log.csv"
    if history:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print("\nvalidation metrics @ threshold " f"{cfg['eval']['threshold']}:")
    for key, value in metrics.items():
        print(f"  {key:20s} {value:.4f}")
    print(f"  threshold for <= {fpr_budget:.0%} FPR: {calibrated:.4f}")
    if "feature_importance_top20" in bundle:
        print("\ntop features by gain:")
        for name, gain in bundle["feature_importance_top20"][:10]:
            print(f"  {name:30s} {gain:.1f}")
    print(f"\nSaved checkpoint to {out_path}")
    if history:
        print(f"Per-epoch log: {log_path}")


if __name__ == "__main__":
    main()
