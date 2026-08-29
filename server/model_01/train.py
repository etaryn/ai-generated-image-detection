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

Instrumentation note: job 768468 ran 15 epochs and reported only val_accuracy,
which sat near 0.5 the whole way. That single number cannot distinguish "learning
slowly" from "collapsed to a constant output", and the run was in fact the latter
(train_loss floored at exactly ln(2) = 0.6931 from epoch 4). Every epoch now logs
AUC, prediction spread, and per-class mean prediction to `<checkpoint_dir>/metrics.jsonl`,
and a collapse detector aborts the job rather than burning the rest of the
allocation on a dead model.

Usage:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
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
from eval.metrics import compute_all_metrics
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


class CheckpointDirLock:
    """Refuses to start if another training process is writing to this checkpoint dir.

    Two runs sharing a checkpoint_dir both write `last.pt.tmp` at the SAME path
    before renaming it over `last.pt`, so their writes interleave and the "atomic"
    rename can publish a half-written file that --resume cannot load. They also
    interleave rows in metrics.jsonl. This is easy to trigger on a cluster --
    submitting the same sbatch twice is all it takes (jobs 770759 and 770767 ran
    concurrently for ~6 minutes this way).

    The lock records the job/host/pid and is refreshed every epoch. A lock older
    than `stale_after` seconds is assumed to belong to a dead job and is taken
    over, so a killed run never permanently blocks the directory.
    """

    def __init__(self, ckpt_dir: Path, stale_after: float = 1800.0):
        self.path = ckpt_dir / "RUNNING.lock"
        self.stale_after = stale_after
        self.identity = {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
            "host": os.environ.get("SLURMD_NODENAME", "unknown"),
            "pid": os.getpid(),
        }

    def acquire(self):
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            try:
                holder = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                holder = {}
            if age < self.stale_after and holder.get("pid") != self.identity["pid"]:
                raise RuntimeError(
                    f"{self.path} is held by another run "
                    f"(job={holder.get('slurm_job_id')} host={holder.get('host')} "
                    f"pid={holder.get('pid')}, refreshed {age:.0f}s ago).\n"
                    f"Two runs sharing a checkpoint dir corrupt last.pt and interleave "
                    f"metrics.jsonl -- refusing to start.\n"
                    f"Either cancel that job, point this run at a different "
                    f"train.checkpoint_dir, or delete the lock if you know the holder "
                    f"is dead."
                )
            print(f"taking over a stale lock ({age:.0f}s old, "
                  f"job={holder.get('slurm_job_id')})")
        self.refresh()
        return self

    def refresh(self):
        self.path.write_text(json.dumps({**self.identity, "updated": time.time()}))

    def release(self):
        try:
            if self.path.exists() and json.loads(self.path.read_text()).get("pid") == self.identity["pid"]:
                self.path.unlink()
        except (json.JSONDecodeError, OSError):
            pass


def build_transform_pipeline(cfg: dict, augment: bool):
    """PIL -> tensor pipeline, with RobustnessAugment injected before ToTensor when
    `augment=True`.

    The augmentation runs on the ORIGINAL-resolution PIL image, before the resize,
    which is what makes RobustnessAugment's per-image severity clamping meaningful.
    """
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


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #

def build_scheduler(optimizer, cfg: dict):
    """Linear warmup over `warmup_epochs`, then cosine decay to zero.

    A from-scratch transformer with no warmup takes very large early steps while
    its attention and positional embeddings are still random, which is how job
    768468 went from a learning epoch 0 (train_loss 0.6714) to a diverged epoch 1
    (0.6925) at lr 1e-3. Warmup is the standard fix and costs nothing.
    """
    epochs = cfg["train"]["epochs"]
    warmup = int(cfg["train"].get("warmup_epochs", 0) or 0)

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def consistency_weight_at(epoch: int, cfg: dict) -> float:
    """Ramp the consistency weight 0 -> target over `consistency_warmup_epochs`.

    The consistency term penalizes disagreement between the clean and augmented
    views of one image. Before the model can classify at all, the cheapest way to
    satisfy it is to emit the same constant for every input -- which also sits at
    the BCE minimum for a balanced dataset (p=0.5, loss=ln(2)). Applying it from
    step 0 therefore actively steers into the collapse it is meant to guard
    against. Ramping it in only once the classifier has real signal to preserve is
    the standard consistency-regularization recipe (cf. Mean Teacher's ramp-up).
    """
    target = cfg["train"]["consistency_loss_weight"]
    warm = int(cfg["train"].get("consistency_warmup_epochs", 0) or 0)
    if warm <= 0 or epoch >= warm:
        return target
    return target * (epoch / warm)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, device, consistency_weight: float,
                    use_freq_branch: bool, grad_clip_norm: float = 1.0, clip_params=None):
    """Each batch contains a clean view and an independently-augmented view of the
    same images. Both are classified (so the model still learns from clean data,
    not just augmented data); a consistency term additionally penalizes the model
    for disagreeing between the two views of the same underlying image -- this
    directly optimizes the "robust under transform" objective rather than relying
    on augmentation exposure alone to produce it as a side effect.

    `clip_params` is the parameter list to gradient-clip. It is passed in
    explicitly (rather than read off `model` here) so this function works whether
    `model` is a plain AIGCDetector or a DistributedDataParallel-wrapped one --
    DDP doesn't forward custom methods like trainable_parameters() the way it
    forwards forward(). train_ddp.py relies on this. Defaults to the model's own
    parameters when omitted.

    Returns per-epoch training diagnostics, not just the loss: training accuracy
    on the clean view tells you whether the model is fitting at all, and the mean
    gradient norm tells you whether it is exploding or has gone flat.
    """
    if clip_params is None:
        clip_params = list(model.parameters())
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_consistency = 0.0
    total_grad_norm = 0.0
    n_correct = 0
    n_seen = 0
    n_batches = 0

    for clean_imgs, aug_imgs, labels, _ in tqdm(loader, desc="train", leave=False):
        clean_imgs = clean_imgs.to(device, non_blocking=True)
        aug_imgs = aug_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        raw_clean = clean_imgs if use_freq_branch else None
        raw_aug = aug_imgs if use_freq_branch else None

        clean_logits = model(clean_imgs, raw_clean)
        aug_logits = model(aug_imgs, raw_aug)

        cls_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(clean_logits, labels)
            + F.binary_cross_entropy_with_logits(aug_logits, labels)
        )

        loss = cls_loss
        consistency = torch.zeros((), device=device)
        if consistency_weight > 0:
            consistency = F.mse_loss(torch.sigmoid(clean_logits), torch.sigmoid(aug_logits))
            loss = loss + consistency_weight * consistency

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, grad_clip_norm)
        else:
            # Measure without clipping, so the number is still logged.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()

        batch_size = clean_imgs.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_consistency += float(consistency) * batch_size
        total_grad_norm += float(grad_norm)
        n_correct += int(((torch.sigmoid(clean_logits) > 0.5).float() == labels).sum())
        n_seen += batch_size
        n_batches += 1

    return {
        "train_loss": total_loss / n_seen,
        "train_cls_loss": total_cls_loss / n_seen,
        "train_consistency": total_consistency / n_seen,
        "train_accuracy": n_correct / n_seen,
        "mean_grad_norm": total_grad_norm / max(1, n_batches),
    }


@torch.no_grad()
def evaluate(model, loader, device, use_freq_branch: bool, threshold: float = 0.5):
    """Held-out metrics, reported richly enough to diagnose a failure from the log.

    `accuracy` alone is ambiguous near 0.5: a collapsed model and a weak-but-real
    classifier both score ~0.5 at a 0.5 threshold. `auc` is threshold-free, so it
    separates them, and `pred_std` catches the collapse outright.
    """
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, _ in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        raw = imgs if use_freq_branch else None
        probs = model.predict_proba(imgs, raw).cpu()
        all_preds.append(probs)
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics = {f"val_{k}": float(v) for k, v in
               compute_all_metrics(labels, preds, threshold=threshold).items()}
    real_mask = labels == 0
    fake_mask = labels == 1
    metrics.update({
        "val_pred_mean": float(preds.mean()),
        "val_pred_std": float(preds.std()),
        "val_pred_min": float(preds.min()),
        "val_pred_max": float(preds.max()),
        "val_mean_pred_on_real": float(preds[real_mask].mean()) if real_mask.any() else float("nan"),
        "val_mean_pred_on_fake": float(preds[fake_mask].mean()) if fake_mask.any() else float("nan"),
        "val_frac_predicted_fake": float((preds > threshold).mean()),
    })
    metrics["val_class_separation"] = (
        metrics["val_mean_pred_on_fake"] - metrics["val_mean_pred_on_real"]
    )
    return metrics


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

    image_size = cfg["data"]["image_size"]
    model_size = cfg["model"].get("input_image_size", image_size)
    if model_size != image_size:
        raise ValueError(
            f"data.image_size ({image_size}) != model.input_image_size ({model_size}). "
            "The model's token grid is derived from input_image_size, so a mismatch "
            "silently changes the number of tokens the positional embedding covers."
        )

    dataset_roots = [f"{args.data_root}/{name}" for name in cfg["data"]["train_datasets"]]
    full_dataset = RealFakeImageDataset(dataset_roots, transform=None)
    print("dataset composition:")
    print(full_dataset.describe())

    train_ds, val_ds = full_dataset.split_train_val(cfg["data"]["val_split"], seed=cfg["train"]["seed"])

    clean_transform = build_transform_pipeline(cfg, augment=False)
    aug_transform = build_transform_pipeline(cfg, augment=True)

    paired_train_ds = PairedViewDataset(train_ds.samples, clean_transform, aug_transform)
    val_ds.transform = clean_transform  # held-out accuracy tracked on clean data;
    # eval/robustness_eval.py handles the per-transform-severity breakdown separately.

    # Report what augmentation will ACTUALLY do at this resolution -- the configured
    # severities get clamped per-image, and a silently-disabled transform should be
    # visible in the log rather than discovered later.
    probe_img_path = paired_train_ds.samples[0][0]
    from PIL import Image as _Image
    with _Image.open(probe_img_path) as _probe:
        native_min_side = min(_probe.size)
    augmenter = RobustnessAugment.from_config(cfg["augmentation"])
    print(f"native image min side = {native_min_side}px, training at {image_size}px")
    print(f"augmentation in effect: {augmenter.describe_for_size(native_min_side)}")

    num_workers = resolve_num_workers(cfg["data"]["num_workers"])
    print(f"device={device} num_workers={num_workers} "
          f"train={len(paired_train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        paired_train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=num_workers, collate_fn=collate_paired, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=num_workers, collate_fn=collate_single, pin_memory=True,
    )

    model = AIGCDetector.from_config(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"model={cfg['model']['architecture']} params={n_params/1e6:.2f}M "
          f"trainable={n_trainable/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = build_scheduler(optimizer, cfg)
    grad_clip_norm = cfg["train"].get("grad_clip_norm", 0.0) or 0.0
    threshold = cfg["eval"]["threshold"]

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lock = CheckpointDirLock(ckpt_dir).acquire()
    metrics_path = ckpt_dir / "metrics.jsonl"
    # Kept alongside metrics.jsonl because the README and train_ddp.py both refer
    # to it: a 4-column curve anyone can open in a spreadsheet, where the jsonl
    # carries the full per-epoch diagnostic record.
    csv_log_path = ckpt_dir / "training_log.csv"
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

    collapse_threshold = cfg["train"].get("collapse_std_threshold", 0.02)
    collapse_patience = int(cfg["train"].get("collapse_patience", 2))
    # Don't arm the detector during warmup. A healthy model legitimately passes
    # through a near-constant phase in its first epochs while it learns the base
    # rate, and aborting there would kill good runs -- the failure this guards
    # against is a model that is STILL constant well after warmup.
    collapse_grace = int(cfg["train"].get(
        "collapse_grace_epochs", max(2, int(cfg["train"].get("warmup_epochs", 0) or 0))))
    collapse_streak = 0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        epoch_start = time.time()
        cons_weight = consistency_weight_at(epoch, cfg)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, cons_weight,
            cfg["model"]["use_freq_branch"], grad_clip_norm,
        )
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()
        metrics = evaluate(model, val_loader, device, cfg["model"]["use_freq_branch"], threshold)

        record = {
            "epoch": epoch,
            "lr": lr_now,
            "consistency_weight": cons_weight,
            "epoch_seconds": round(time.time() - epoch_start, 1),
            **train_stats,
            **metrics,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        write_header = not csv_log_path.exists()
        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_accuracy", "lr"])
            writer.writerow([epoch, train_stats["train_loss"], metrics["val_accuracy"], lr_now])

        print(
            f"epoch {epoch}: train_loss={train_stats['train_loss']:.4f} "
            f"train_acc={train_stats['train_accuracy']:.4f} "
            f"val_acc={metrics['val_accuracy']:.4f} "
            f"val_auc={metrics['val_auc']:.4f} "
            f"sep={metrics['val_class_separation']:+.4f} "
            f"pred_std={metrics['val_pred_std']:.4f} "
            f"grad={train_stats['mean_grad_norm']:.3f} "
            f"cons_w={cons_weight:.3f} lr={lr_now:.2e} "
            f"({record['epoch_seconds']:.0f}s)"
        )

        # --- collapse detection -------------------------------------------- #
        # A model emitting a near-constant probability for every input is not
        # training, and no amount of further epochs recovers it on its own. Job
        # 768468 spent 11 epochs and ~80 minutes in this state.
        # Require BOTH a flat prediction distribution AND no ranking signal. A model
        # with low spread but a real AUC is merely poorly scaled, not collapsed, and
        # must not be killed for it.
        is_collapsed = metrics["val_pred_std"] < collapse_threshold and metrics["val_auc"] < 0.55
        if is_collapsed:
            collapse_streak += 1
            print(f"  !! WARNING: prediction std {metrics['val_pred_std']:.5f} < "
                  f"{collapse_threshold} and AUC {metrics['val_auc']:.4f} < 0.55 -- the "
                  f"model is emitting a near-constant output "
                  f"({collapse_streak}/{collapse_patience} epochs)")
        else:
            collapse_streak = 0

        if collapse_streak >= collapse_patience and epoch >= collapse_grace:
            print(
                f"\nABORTING: the model has produced a near-constant prediction for "
                f"{collapse_streak} consecutive epochs (mean={metrics['val_pred_mean']:.4f}, "
                f"std={metrics['val_pred_std']:.5f}, AUC={metrics['val_auc']:.4f}).\n"
                f"This is the ln(2) collapse, not slow learning. Continuing would waste "
                f"the rest of the allocation.\n"
                f"Check, in order: the consistency weight ({cons_weight:.3f}) and its "
                f"warmup, the learning rate ({lr_now:.2e}) and warmup, and whether the "
                f"augmented view still carries class information at this resolution.\n"
                f"Per-epoch history: {metrics_path}",
                file=sys.stderr,
            )
            lock.release()
            sys.exit(2)

        if metrics["val_accuracy"] > best_val_acc:
            best_val_acc = metrics["val_accuracy"]
            torch.save(
                {"model_state": model.state_dict(), "config": cfg,
                 "val_accuracy": best_val_acc, "metrics": metrics, "epoch": epoch},
                ckpt_dir / "best.pt",
            )
            print(f"  -> saved new best checkpoint (val_accuracy={best_val_acc:.4f})")

        # Full training state every epoch, so a job killed at the walltime limit (or
        # requeued after preemption) resumes instead of restarting. Written to a temp
        # file and renamed: a crash mid-write would otherwise leave a truncated
        # last.pt that --resume can't load. rename is atomic within a filesystem.
        lock.refresh()
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

    lock.release()
    print(f"\ndone. best val_accuracy={best_val_acc:.4f}. per-epoch metrics: {metrics_path}")


if __name__ == "__main__":
    main()
