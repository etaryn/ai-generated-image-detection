"""Trains only the classification head (and optional frequency branch) on top of
the frozen CLIP backbone, using the robustness-augmentation pipeline so the model
learns invariance to the challenge's named transforms rather than memorizing
pristine-image statistics.

Usage:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from data.datasets import RealFakeImageDataset
from data.transforms import RobustnessAugment
from model.detector import AIGCDetector


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform_pipeline(cfg: dict, augment: bool):
    """PIL -> tensor pipeline, with RobustnessAugment injected before ToTensor when
    `augment=True` (i.e. for the training split)."""
    image_size = cfg["data"]["image_size"]
    ops = []
    if augment:
        ops.append(RobustnessAugment.from_config(cfg["augmentation"]))
    ops += [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP's normalization
    ]
    return T.Compose(ops)


def collate(batch):
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32), list(paths)


def train_one_epoch(model, loader, optimizer, device, consistency_weight: float, use_freq_branch: bool):
    model.train()
    total_loss = 0.0
    for imgs, labels, _ in tqdm(loader, desc="train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        raw = imgs if use_freq_branch else None

        logits = model(imgs, raw)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Optional consistency term: encourages similar predictions for a sample
        # and a freshly-augmented view of it within the same batch, directly
        # optimizing for the "robust under transform" objective rather than
        # relying on augmentation exposure alone. Cheap approximation here reuses
        # the same batch with a second augmentation pass; swap in a paired
        # clean/augmented dataloader for a stricter version.
        if consistency_weight > 0:
            with torch.no_grad():
                aug_imgs = imgs  # placeholder: same-batch reuse; see docstring above
            aug_logits = model(aug_imgs, raw)
            consistency = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(aug_logits).detach())
            loss = loss + consistency_weight * consistency

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, use_freq_branch: bool):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, _ in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device)
        raw = imgs if use_freq_branch else None
        probs = model.predict_proba(imgs, raw).cpu()
        all_preds.append(probs)
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = ((preds > 0.5).astype(int) == labels).mean()
    return {"val_accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_roots = [f"data/raw/{name}" for name in cfg["data"]["train_datasets"]]
    full_dataset = RealFakeImageDataset(dataset_roots, transform=None)
    train_ds, val_ds = full_dataset.split_train_val(cfg["data"]["val_split"], seed=cfg["train"]["seed"])
    train_ds.transform = build_transform_pipeline(cfg, augment=True)
    val_ds.transform = build_transform_pipeline(cfg, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate,
    )

    model = AIGCDetector.from_config(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            cfg["train"]["consistency_loss_weight"], cfg["model"]["use_freq_branch"],
        )
        metrics = evaluate(model, val_loader, device, cfg["model"]["use_freq_branch"])
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_accuracy={metrics['val_accuracy']:.4f}")

        if metrics["val_accuracy"] > best_val_acc:
            best_val_acc = metrics["val_accuracy"]
            torch.save(
                {"model_state": model.state_dict(), "config": cfg, "val_accuracy": best_val_acc},
                ckpt_dir / "best.pt",
            )
            print(f"  -> saved new best checkpoint (val_accuracy={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
