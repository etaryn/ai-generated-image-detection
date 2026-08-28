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
"""
from __future__ import annotations

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint (.pt)")
    parser.add_argument("--output", default="predictions.json", help="Where to write the JSON output")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = AIGCDetector.from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = ImageFolderInference(args.input_dir, transform=build_inference_transform(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    results = run_inference(model, loader, device, cfg["model"]["use_freq_branch"])

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
