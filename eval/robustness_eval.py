"""Builds the transform x severity evaluation matrix — the core "Robustness
Evaluation Summary" deliverable (challenge brief, section 5.5 item 4).

For each named severity in data/transforms.SEVERITY_LEVELS (clean + each
transform at each listed parameter, plus a couple of compounded combos), this
scores the model on the same held-out set and reports accuracy/AUC/F1/FPR, then
writes both a CSV table and a heatmap PNG for the demo/Devpost writeup.

Usage:
    python eval/robustness_eval.py --config configs/default.yaml --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from data.datasets import RealFakeImageDataset
from data.transforms import SEVERITY_LEVELS, apply_named_transform
from eval.metrics import compute_all_metrics
from model.detector import AIGCDetector

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class NamedSeverityTransform:
    """Applies one fixed SEVERITY_LEVELS entry, then the standard resize/normalize."""

    def __init__(self, severity_name: str, image_size: int):
        self.severity_name = severity_name
        self.post = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __call__(self, img):
        img = apply_named_transform(img, self.severity_name)
        return self.post(img)


def collate(batch):
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32), list(paths)


@torch.no_grad()
def score_dataset(model, loader, device, use_freq_branch: bool):
    all_preds, all_labels = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        raw = imgs if use_freq_branch else None
        probs = model.predict_proba(imgs, raw).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--eval_datasets",
        nargs="*",
        default=None,
        help="Dataset roots to evaluate on (defaults to the config's demo_eval_set, "
             "i.e. the held-out WildFake COCO/DALL-E subset).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = AIGCDetector.from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.eval_datasets:
        dataset_roots = args.eval_datasets
    else:
        demo = cfg["data"]["demo_eval_set"]
        # RealFakeImageDataset expects <root>/real and <root>/fake; the demo set's
        # two halves live under different folder names, so point at their parent
        # via a small ad-hoc root layout note in prepare_data.py, or pass
        # --eval_datasets explicitly once your demo set is laid out.
        dataset_roots = [str(Path(demo["real_dir"]).parent)]

    output_dir = Path(cfg["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for severity_name in SEVERITY_LEVELS:
        transform = NamedSeverityTransform(severity_name, cfg["data"]["image_size"])
        dataset = RealFakeImageDataset(dataset_roots, transform=transform)
        loader = DataLoader(
            dataset, batch_size=cfg["data"]["batch_size"], shuffle=False,
            num_workers=cfg["data"]["num_workers"], collate_fn=collate,
        )
        preds, labels = score_dataset(loader=loader, model=model, device=device,
                                       use_freq_branch=cfg["model"]["use_freq_branch"])
        metrics = compute_all_metrics(labels, preds, threshold=cfg["eval"]["threshold"])
        metrics["severity"] = severity_name
        rows.append(metrics)
        print(f"{severity_name:20s} acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} "
              f"f1={metrics['f1']:.3f} fpr={metrics['fpr_at_threshold']:.3f}")

    df = pd.DataFrame(rows).set_index("severity")
    csv_path = output_dir / "robustness_matrix.csv"
    df.to_csv(csv_path)
    print(f"\nWrote metrics table to {csv_path}")

    # Heatmap for the demo video / Devpost writeup.
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    sns.heatmap(df[["accuracy", "auc", "f1", "fpr_at_threshold"]], annot=True, fmt=".2f",
                cmap="RdYlGn_r", vmin=0, vmax=1, ax=ax)
    ax.set_title("Robustness matrix: clean vs. transformed")
    fig.tight_layout()
    png_path = output_dir / "robustness_matrix.png"
    fig.savefig(png_path, dpi=150)
    print(f"Wrote heatmap to {png_path}")


if __name__ == "__main__":
    main()
