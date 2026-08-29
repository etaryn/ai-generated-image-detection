"""Required deliverable script: takes an image directory as input and outputs a
JSON file with a confidence score (probability of being AI-generated) per image.

Output format is identical to model_01/infer.py, so the two models are drop-in
comparable on the same scoring harness:

    [
      {"image_path": "/path/to/images/img001.jpg", "pred": 0.93},
      {"image_path": "/path/to/images/img002.jpg", "pred": 0.07}
    ]

Everything needed to reproduce training-time preprocessing lives in the
checkpoint -- the feature config, the canonical resolution, the column selection
and the scaler -- so this script takes no feature flags of its own.

Usage:
    python infer.py --input_dir /path/to/images --checkpoint checkpoints/best.pt \\
        --output predictions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers import load_predictor
from data_io import CanonicalInferenceDataset, collate_unlabeled
from features.pipeline import FeatureStack


@torch.no_grad()
def extract_features(stack: FeatureStack, loader: DataLoader) -> tuple[np.ndarray, list[str]]:
    feats, paths = [], []
    for imgs, batch_paths in tqdm(loader, desc="infer"):
        feats.append(stack(imgs))
        paths.extend(batch_paths)
    return np.concatenate(feats).astype(np.float32), paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--checkpoint", required=True, help="Path to a train.py checkpoint (.pt)")
    parser.add_argument("--output", default="predictions.json", help="Where to write the JSON output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    parser.add_argument(
        "--use_calibrated_threshold",
        action="store_true",
        help="Also report a 0/1 label using the checkpoint's FPR-budget threshold "
             "instead of 0.5 (the `pred` probability is unaffected)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # weights_only=False: the bundle intentionally carries the scaler arrays, the
    # feature config and (for xgboost) the raw booster bytes, not just tensors.
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    stack = FeatureStack.from_config(bundle["features_config"], device=device)
    dataset = CanonicalInferenceDataset(args.input_dir, bundle["canonical_size"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_unlabeled,
    )

    X, paths = extract_features(stack, loader)

    columns = bundle.get("feature_columns")
    if columns is not None:
        if X.shape[1] <= max(columns):
            raise RuntimeError(
                f"The checkpoint expects a {max(columns) + 1}-dim feature vector but the "
                f"rebuilt extractor stack produced {X.shape[1]}. The checkpoint's "
                f"features_config and the installed extractors have diverged."
            )
        X = X[:, columns]

    scaler = bundle["scaler"]
    X = ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)

    probs = load_predictor(bundle)(X)

    threshold = bundle["threshold"]
    if args.use_calibrated_threshold:
        threshold = bundle["calibrated_threshold"]["threshold"]

    results = []
    for path, prob in zip(paths, probs):
        row = {"image_path": path, "pred": float(prob)}
        if args.use_calibrated_threshold:
            row["label"] = int(prob >= threshold)
        results.append(row)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
