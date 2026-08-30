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

Also exposes `predict_image(pil_image) -> float`, and its batch sibling
`predict_images`, for single-image callers such as the Streamlit client in
../../client/app.py, which needs a score per upload rather than a batch job over
a directory. The signature matches model_01/infer.py's so a caller can switch
which model it imports without changing how it calls it.

Usage:
    python infer.py --input_dir /path/to/images --checkpoint checkpoints/best.pt \
        --output predictions.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers import load_predictor
from data_io import CanonicalInferenceDataset, canonical_transform, collate_unlabeled
from features.pipeline import FeatureStack

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best.pt"

def _warn_if_backbone_mismatch(bundle: dict, stack: FeatureStack) -> None:
    """Flag a checkpoint whose features were produced by a different tower than the
    one about to serve it.

    The motivating case is the QuickGELU activation. features/clip.py used to build
    open_clip's plain "ViT-B-16" config with pretrained="openai", whose weights were
    trained with QuickGELU -- open_clip only warns, then silently runs the wrong
    activation. Features shift by ~0.235 relative L2, so a classifier fitted on the
    old features is being served new ones, and the dimension check cannot catch it
    because the width is identical.

    The comparison is against `stack`, the extractor set this process just built, so
    it reflects what will actually run rather than a second derivation of it. The
    checkpoint side comes from `resolved_backbones`, recorded at extraction time.

    An earlier version of this check compared the checkpoint's *requested* backbone
    name against the resolved one. That fires on every checkpoint ever written --
    the config records "ViT-B-16" whether or not the fix is in effect -- so it
    warned on correct and stale checkpoints alike and told you nothing. A missing
    `resolved_backbones` is now the stale signal, since only caches written before
    the field existed lack it, and those are exactly the pre-fix ones.
    """
    if not ((bundle.get("features_config") or {}).get("clip") or {}).get("enabled", False):
        return
    serving = stack.resolved_backbones()
    recorded = bundle.get("resolved_backbones")

    if recorded is None:
        warnings.warn(
            "checkpoint does not record which backbones extracted its features, so it "
            "predates the QuickGELU activation fix: its CLIP features were almost "
            "certainly produced by the plain 'ViT-B-16' tower running standard GELU "
            "against QuickGELU-trained weights, while this build serves the correct "
            f"tower ({serving.get('clip')!r}). Re-extract features and refit the "
            "classifier before trusting these scores. The backbones are frozen, so "
            "this costs an extraction pass, not a retrain.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    differing = {
        name: (was, now)
        for name, now in serving.items()
        if (was := recorded.get(name)) is not None and was != now
    }
    if differing:
        detail = "; ".join(f"{n}: fitted on {w!r}, serving {s!r}" for n, (w, s) in differing.items())
        warnings.warn(
            f"checkpoint's features were extracted with a different backbone than this "
            f"build serves ({detail}). The feature width is unchanged, so nothing else "
            f"catches this. Re-extract and refit before trusting these scores.",
            RuntimeWarning,
            stacklevel=2,
        )


# Building the stack instantiates (and on a cold cache downloads) the frozen
# DINOv2/CLIP backbones, which is far too slow to redo per upload, so the loaded
# bundle is cached and reused across calls.
_LOADED: dict = {}


def load_model(
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
):
    """Load a checkpoint and return (bundle, stack, predictor, device).

    The checkpoint is self-contained -- it carries the feature config, the
    canonical resolution, the trained column selection and the scaler -- so the
    extractor stack is rebuilt from the file rather than from a config that may
    have moved on since the run.
    """
    checkpoint = Path(
        checkpoint or os.environ.get("AIGC_MODEL02_CHECKPOINT") or DEFAULT_CHECKPOINT
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Pass one explicitly, or set "
            f"$AIGC_MODEL02_CHECKPOINT to point at your .pt file."
        )

    device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (str(checkpoint.resolve()), device)
    if key in _LOADED:
        return _LOADED[key]

    # weights_only=False: the bundle intentionally carries the scaler arrays, the
    # feature config and (for xgboost) the raw booster bytes, not just tensors.
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stack = FeatureStack.from_config(bundle["features_config"], device=device)
    _warn_if_backbone_mismatch(bundle, stack)
    predictor = load_predictor(bundle)

    _LOADED[key] = (bundle, stack, predictor, device)
    return _LOADED[key]


def apply_checkpoint_preprocessing(X: np.ndarray, bundle: dict) -> np.ndarray:
    """Narrow raw features to the trained columns and apply the training scaler."""
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
    return ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)


@torch.no_grad()
def predict_images(images, checkpoint: str | Path | None = None) -> list[float]:
    """Score a list of PIL images. Returns P(AI-generated) in [0, 1] for each.

    Batched into a single pass through the frozen extractors, so scoring N
    transformed copies of one upload costs one pass rather than N.
    """
    bundle, stack, predictor, _ = load_model(checkpoint)
    to_canonical = canonical_transform(bundle["canonical_size"])
    batch = torch.stack([to_canonical(img.convert("RGB")) for img in images])
    X = apply_checkpoint_preprocessing(stack(batch), bundle)
    return [float(p) for p in predictor(X)]


def predict_image(image, checkpoint: str | Path | None = None) -> float:
    """Score one PIL image. Returns P(AI-generated) in [0, 1].

    This is the single-image entry point the Streamlit client calls.
    """
    return predict_images([image], checkpoint)[0]


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
    parser.add_argument("--checkpoint", default=None,
                        help=f"Path to a train.py checkpoint (.pt). Default: {DEFAULT_CHECKPOINT}")
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

    bundle, stack, predictor, _ = load_model(args.checkpoint, args.device)

    dataset = CanonicalInferenceDataset(args.input_dir, bundle["canonical_size"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_unlabeled,
    )

    X, paths = extract_features(stack, loader)
    probs = predictor(apply_checkpoint_preprocessing(X, bundle))

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
