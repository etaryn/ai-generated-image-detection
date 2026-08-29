"""Builds the transform x severity evaluation matrix for model_02 -- the same
"Robustness Evaluation Summary" deliverable model_01/eval/robustness_eval.py
produces, computed on the same severities with the same metric code, so the two
tables can be laid side by side.

For each named severity in model_01's `data/transforms.SEVERITY_LEVELS`, the eval
set is re-extracted through the frozen feature stack with that transform applied,
scored, and summarized. Note the cost: unlike model_01 (one forward pass per
image per severity), model_02 re-runs *feature extraction* per severity, which is
two ViTs plus the spectral statistics. Use --limit while iterating.

The FFT block is the one expected to suffer most here -- JPEG recompression and
resizing directly overwrite the high-frequency evidence it reads. Comparing this
matrix against `--blocks dino clip` (an FFT-free checkpoint from train.py) is the
honest way to show what the spectral branch is worth *after* redistribution, not
just on pristine images.

Usage:
    python eval/robustness_eval.py --config configs/default.yaml \\
        --checkpoint checkpoints/best.pt --limit 2000
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
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from classifiers import load_predictor  # noqa: E402
from data_io import CanonicalDataset, NamedSeverity, build_labeled_samples, collate_labeled  # noqa: E402
from extract_features import demo_eval_samples, resolve_dataset_roots  # noqa: E402
from features.pipeline import FeatureStack  # noqa: E402
from shared import SEVERITY_LEVELS, compute_all_metrics  # noqa: E402


@torch.no_grad()
def score_severity(stack, predictor, samples, severity, cfg, bundle, batch_size, num_workers):
    dataset = CanonicalDataset(
        samples, bundle["canonical_size"], pil_transform=NamedSeverity(severity)
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_labeled
    )
    feats, labels = [], []
    for imgs, batch_labels, _, _ in tqdm(loader, desc=severity, leave=False):
        feats.append(stack(imgs))
        labels.extend(batch_labels)

    X = np.concatenate(feats).astype(np.float32)
    columns = bundle.get("feature_columns")
    if columns is not None:
        X = X[:, columns]
    scaler = bundle["scaler"]
    X = ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)
    return predictor(X), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--eval_datasets",
        nargs="*",
        default=None,
        help="Dataset names under data.data_root to evaluate on (default: the "
             "config's held-out demo_eval_set)",
    )
    parser.add_argument("--severities", nargs="*", default=None, help="Override the config's severity list")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on a random subset of N images")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or cfg["data"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["data"]["num_workers"]

    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    predictor = load_predictor(bundle)
    stack = FeatureStack.from_config(bundle["features_config"], device=device)

    if args.eval_datasets:
        samples = build_labeled_samples(resolve_dataset_roots(cfg, args.eval_datasets))
    else:
        samples = demo_eval_samples(cfg)

    if args.limit is not None and args.limit < len(samples):
        rng = np.random.default_rng(cfg["train"]["seed"])
        keep = rng.choice(len(samples), size=args.limit, replace=False)
        samples = [samples[i] for i in sorted(keep)]

    severities = args.severities or cfg["eval"]["severities"]
    if severities == ["all"]:
        severities = list(SEVERITY_LEVELS)

    rows = []
    for severity in severities:
        preds, labels = score_severity(
            stack, predictor, samples, severity, cfg, bundle, batch_size, num_workers
        )
        metrics = compute_all_metrics(labels, preds, cfg["eval"]["threshold"])
        rows.append({"severity": severity, **metrics})
        print(
            f"{severity:22s} acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} "
            f"f1={metrics['f1']:.4f} fpr={metrics['fpr_at_threshold']:.4f}"
        )

    df = pd.DataFrame(rows).set_index("severity")
    out_dir = Path(cfg["eval"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "robustness_matrix.csv"
    df.to_csv(csv_path)
    print(f"\nWrote {csv_path}")

    # The heatmap is for the writeup; a missing matplotlib/seaborn shouldn't cost
    # you the CSV you just spent GPU-hours computing.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 0.4 * len(df) + 2))
        sns.heatmap(df[["accuracy", "auc", "f1"]], annot=True, fmt=".3f", vmin=0.5, vmax=1.0, cmap="viridis", ax=ax)
        ax.set_title("model_02 robustness: metrics per redistribution severity")
        fig.tight_layout()
        png_path = out_dir / "robustness_matrix.png"
        fig.savefig(png_path, dpi=150)
        print(f"Wrote {png_path}")
    except ImportError as exc:
        print(f"(skipping heatmap: {exc})")


if __name__ == "__main__":
    main()
