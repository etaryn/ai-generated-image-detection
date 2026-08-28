"""Trains the detector using the robustness-augmentation pipeline, so the model
learns invariance to the challenge's named transforms rather than memorizing
pristine-image statistics. With the default `model.architecture: cnn_transformer`
this trains the whole model end-to-end (no frozen component); with
`clip_frozen` it trains only the head (+ optional frequency branch).

Each training sample is loaded as a genuine clean/augmented PAIR (see
`data.datasets.PairedViewDataset`) so the consistency loss compares the model's
prediction on an image against its prediction on a redistributed copy of the
SAME image -- which is the literal "robust under transform" objective from the
challenge brief, rather than an approximation.

Usage:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from data.datasets import PairedViewDataset, RealFakeImageDataset
from data.transforms import RobustnessAugment
from model.detector import AIGCDetector

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_num_workers(cfg_value: int) -> int:
    """DataLoader workers, capped by the CPUs the batch scheduler actually gave us.

    Hardcoding num_workers above the cgroup's CPU allocation doesn't buy throughput --
    the workers just contend for the same cores, and on SLURM an oversubscribed job
    gets throttled or OOM-killed (each worker is a fork with its own copy of the
    dataset index).
    """
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated:
        return max(1, min(int(cfg_value), int(allocated)))
    return cfg_value


def build_transform_pipeline(cfg: dict, augment: bool):
    """PIL -> tensor pipeline, with RobustnessAugment injected before ToTensor when
    `augment=True`."""
    image_size = cfg["data"]["image_size"]
    ops = []
    if augment:
        ops.append(RobustnessAugment.from_config(cfg["augmentation"]))
    ops += [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
    return T.Compose(ops)


def collate_paired(batch):
    clean, aug, labels, paths = zip(*batch)
    return torch.stack(clean), torch.stack(aug), torch.tensor(labels, dtype=torch.float32), list(paths)


def collate_single(batch):
    imgs, labels, paths = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32), list(paths)


def train_one_epoch(model, loader, optimizer, device, consistency_weight: float, use_freq_branch: bool):
    """Each batch contains a clean view and an independently-augmented view of the
    same images. Both are classified (so the model still learns from clean data,
    not just augmented data); a consistency term additionally penalizes the model
    for disagreeing between the two views of the same underlying image -- this
    directly optimizes the "robust under transform" objective rather than relying
    on augmentation exposure alone to produce it as a side effect.
    """
    model.train()
    total_loss = 0.0
    for clean_imgs, aug_imgs, labels, _ in tqdm(loader, desc="train", leave=False):
        clean_imgs = clean_imgs.to(device)
        aug_imgs = aug_imgs.to(device)
        labels = labels.to(device)
        raw_clean = clean_imgs if use_freq_branch else None
        raw_aug = aug_imgs if use_freq_branch else None

        clean_logits = model(clean_imgs, raw_clean)
        aug_logits = model(aug_imgs, raw_aug)

        cls_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(clean_logits, labels)
            + F.binary_cross_entropy_with_logits(aug_logits, labels)
        )

        loss = cls_loss
        if consistency_weight > 0:
            consistency = F.mse_loss(torch.sigmoid(clean_logits), torch.sigmoid(aug_logits))
            loss = loss + consistency_weight * consistency

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clean_imgs.size(0)
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
    parser.add_argument(
        "--data_root",
        default="data/raw",
        help="Parent dir holding <dataset_name>/{real,fake}. Override to read from "
             "node-local scratch instead of network storage.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoints/last.pt if it exists (walltime limits, requeue).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_roots = [f"{args.data_root}/{name}" for name in cfg["data"]["train_datasets"]]
    full_dataset = RealFakeImageDataset(dataset_roots, transform=None)
    train_ds, val_ds = full_dataset.split_train_val(cfg["data"]["val_split"], seed=cfg["train"]["seed"])

    clean_transform = build_transform_pipeline(cfg, augment=False)
    aug_transform = build_transform_pipeline(cfg, augment=True)

    paired_train_ds = PairedViewDataset(train_ds.samples, clean_transform, aug_transform)
    val_ds.transform = clean_transform  # held-out accuracy tracked on clean data;
    # eval/robustness_eval.py handles the per-transform-severity breakdown separately.

    num_workers = resolve_num_workers(cfg["data"]["num_workers"])
    print(f"device={device} num_workers={num_workers} "
          f"train={len(paired_train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        paired_train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=num_workers, collate_fn=collate_paired,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=num_workers, collate_fn=collate_single,
    )

    model = AIGCDetector.from_config(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    start_epoch = 0

    resume_path = ckpt_dir / "last.pt"
    if args.resume and resume_path.exists():
        # weights_only=False: this file holds optimizer/scheduler state and the config
        # dict, not just tensors. It's our own artifact, not untrusted input.
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["config"]["model"] != cfg["model"]:
            raise RuntimeError(
                f"{resume_path} was trained with a different model config than "
                f"{args.config}. Resuming would load mismatched weights -- either "
                "revert the config or move the old checkpoint aside and start fresh."
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = state["epoch"]
        best_val_acc = state["best_val_acc"]
        print(f"resumed from {resume_path}: starting at epoch {start_epoch}, "
              f"best_val_accuracy={best_val_acc:.4f}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            cfg["train"]["consistency_loss_weight"], cfg["model"]["use_freq_branch"],
        )
        scheduler.step()
        metrics = evaluate(model, val_loader, device, cfg["model"]["use_freq_branch"])
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_accuracy={metrics['val_accuracy']:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if metrics["val_accuracy"] > best_val_acc:
            best_val_acc = metrics["val_accuracy"]
            torch.save(
                {"model_state": model.state_dict(), "config": cfg, "val_accuracy": best_val_acc},
                ckpt_dir / "best.pt",
            )
            print(f"  -> saved new best checkpoint (val_accuracy={best_val_acc:.4f})")

        # Full training state every epoch, so a job killed at the walltime limit (or
        # requeued after preemption) resumes instead of restarting. Written to a temp
        # file and renamed: a crash mid-write would otherwise leave a truncated
        # last.pt that --resume can't load. rename is atomic within a filesystem.
        tmp_path = ckpt_dir / "last.pt.tmp"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_val_acc,
                "config": cfg,
            },
            tmp_path,
        )
        tmp_path.replace(ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()
