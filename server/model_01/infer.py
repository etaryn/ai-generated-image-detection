"""Required deliverable script: takes an image directory as input and outputs a
JSON file with a confidence score (probability of being AI-generated) per image.

Usage:
    python infer.py --input_dir /path/to/images --checkpoint checkpoints/best.pt \\
        --output predictions.json

Output format (list of objects, one per image):
    [
      {"image_path": "/path/to/images/img001.jpg", "pred": 0.93},
      {"image_path": "/path/to/images/img002.jpg", "pred": 0.07}
    ]

Also exposes `predict_image(pil_image) -> float` for single-image callers such as
the Streamlit client in ../../client/app.py, which needs a score per upload rather
than a batch job over a directory.

Note on image size: it is read from the checkpoint, never assumed. The model's
positional embedding is a fixed-size parameter sized to input_image_size, so
feeding a differently-sized image is a hard shape error, not a soft degradation.
The 32px CIFAKE checkpoint and a 224px full-resolution one therefore need
different preprocessing, and only the checkpoint knows which it is.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from data.datasets import ImageFolderInference
from model.detector import AIGCDetector

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_inference_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def collate(batch):
    imgs, paths = zip(*batch)
    return torch.stack(imgs), list(paths)


@torch.no_grad()
def run_inference(model, loader, device, use_freq_branch: bool):
    results = []
    for imgs, paths in tqdm(loader, desc="infer"):
        imgs = imgs.to(device)
        raw = imgs if use_freq_branch else None
        probs = model.predict_proba(imgs, raw).cpu().tolist()
        for path, prob in zip(paths, probs):
            results.append({"image_path": path, "pred": float(prob)})
    return results


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best.pt"

# Loading the checkpoint takes long enough that doing it per-upload makes the UI
# feel broken, so the built model is cached and reused across calls.
_LOADED: dict = {}


def load_model(checkpoint: str | Path | None = None, device: torch.device | None = None):
    """Build the detector from a checkpoint and return (model, cfg, device).

    The checkpoint carries the full training config, so the architecture is
    reconstructed from the file itself rather than from a config that may have
    moved on since the run.
    """
    checkpoint = Path(checkpoint or os.environ.get("AIGC_CHECKPOINT") or DEFAULT_CHECKPOINT)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. Pass one explicitly, or set "
            f"$AIGC_CHECKPOINT to point at your .pt file."
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(checkpoint), str(device))
    if key in _LOADED:
        return _LOADED[key]

    # weights_only=False: the file holds the training config dict, not just tensors.
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = AIGCDetector.from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _LOADED[key] = (model, cfg, device)
    return _LOADED[key]


def checkpoint_image_size(cfg: dict) -> int:
    """The size this checkpoint's positional embedding was trained for."""
    return cfg["model"].get("input_image_size") or cfg["data"]["image_size"]


@torch.no_grad()
def predict_image(image, checkpoint: str | Path | None = None) -> float:
    """Score one PIL image. Returns P(AI-generated) in [0, 1].

    This is the single-image entry point the Streamlit client calls.
    """
    model, cfg, device = load_model(checkpoint)
    transform = build_inference_transform(checkpoint_image_size(cfg))
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    raw = tensor if cfg["model"]["use_freq_branch"] else None
    return float(model.predict_proba(tensor, raw).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--checkpoint", default=None,
                        help=f"Path to a training checkpoint (.pt). Default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--output", default="predictions.json", help="Where to write the JSON output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override the input size. Defaults to the size recorded in the "
                             "checkpoint, which is almost always what you want -- the model's "
                             "positional embedding is sized for it.")
    args = parser.parse_args()

    model, cfg, device = load_model(args.checkpoint)
    image_size = args.image_size or checkpoint_image_size(cfg)
    print(f"scoring at {image_size}px (checkpoint was trained at "
          f"{checkpoint_image_size(cfg)}px)")

    dataset = ImageFolderInference(args.input_dir, transform=build_inference_transform(image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    results = run_inference(model, loader, device, cfg["model"]["use_freq_branch"])

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
