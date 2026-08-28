"""Pulls representative false positives and false negatives, for the "Error
Analysis Note" deliverable (challenge brief, section 5.5 item 5).

Also reports the threshold that would hit a target false-positive-rate budget, as
a starting point for the "calibrate against an FPR budget, not a fixed 0.5 cutoff"
discussion in the README's limitations section.

Usage:
    python eval/error_analysis.py --config configs/default.yaml --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data.datasets import RealFakeImageDataset
from eval.metrics import fpr_at_threshold, threshold_for_target_fpr
from model.detector import AIGCDetector

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@torch.no_grad()
def collect_predictions(model, loader, device, use_freq_branch: bool):
    preds, labels, paths = [], [], []
    for imgs, lbls, batch_paths in loader:
        imgs = imgs.to(device)
        raw = imgs if use_freq_branch else None
        probs = model.predict_proba(imgs, raw).cpu().numpy()
        preds.append(probs)
        labels.append(lbls.numpy())
        paths.extend(batch_paths)
    return np.concatenate(preds), np.concatenate(labels), paths


def collate(batch):
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32), list(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_datasets", nargs="*", required=True,
                         help="Dataset roots to analyze (e.g. a held-out val split or the demo set).")
    parser.add_argument("--top_k", type=int, default=20, help="How many examples of each error type to save.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = AIGCDetector.from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = T.Compose([
        T.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    dataset = RealFakeImageDataset(args.eval_datasets, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, collate_fn=collate)

    preds, labels, paths = collect_predictions(model, loader, device, cfg["model"]["use_freq_branch"])
    threshold = cfg["eval"]["threshold"]

    pred_labels = (preds >= threshold).astype(int)
    is_fp = (labels == 0) & (pred_labels == 1)   # real, predicted fake
    is_fn = (labels == 1) & (pred_labels == 0)   # fake, predicted real

    def _top_examples(mask, ascending_by_confidence_gap):
        idx = np.where(mask)[0]
        # Sort by how confidently wrong the model was (most confident mistakes first).
        confidence_gap = np.abs(preds[idx] - threshold)
        order = idx[np.argsort(-confidence_gap)]
        return order[: args.top_k]

    fp_idx = _top_examples(is_fp, ascending_by_confidence_gap=False)
    fn_idx = _top_examples(is_fn, ascending_by_confidence_gap=False)

    output_dir = Path(cfg["eval"]["output_dir"]) / "error_analysis"
    (output_dir / "false_positives").mkdir(parents=True, exist_ok=True)
    (output_dir / "false_negatives").mkdir(parents=True, exist_ok=True)

    report = {"false_positive_rate": fpr_at_threshold(labels, preds, threshold),
              "threshold_used": threshold,
              "suggested_threshold_for_5pct_fpr": threshold_for_target_fpr(labels, preds, 0.05),
              "false_positives": [], "false_negatives": []}

    for name, idx_list, key in (("false_positives", fp_idx, "false_positives"),
                                 ("false_negatives", fn_idx, "false_negatives")):
        for i in idx_list:
            src = Path(paths[i])
            dst = output_dir / name / src.name
            try:
                shutil.copy(src, dst)
            except OSError:
                pass  # source may already be gone / permissions issue; skip copy, keep record
            report[key].append({"image_path": str(src), "pred": float(preds[i]), "label": int(labels[i])})

    report_path = output_dir / "error_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"False-positive rate @ threshold={threshold}: {report['false_positive_rate']:.3f}")
    print(f"Threshold for ~5% FPR budget: {report['suggested_threshold_for_5pct_fpr']:.3f}")
    print(f"Wrote {len(fp_idx)} false positives and {len(fn_idx)} false negatives to {output_dir}")
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
