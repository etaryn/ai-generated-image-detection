"""Multi-machine distributed training: combines GPUs across several laptops into
one training job via PyTorch DistributedDataParallel (DDP), so a single epoch
finishes roughly N-times faster across N machines (network overhead eats into
that ideal, especially over WiFi rather than a datacenter interconnect).

Reuses train.py's model/data/loss logic unchanged -- this script only adds the
distributed-specific wiring: process-group setup, a DistributedSampler so each
machine trains on a different shard of the data each epoch, wrapping the model
in DDP (which auto-synchronizes gradients across machines on every `.backward()`
call), and gating checkpoint/log writes to a single rank so 3 machines don't
race to write the same file.

*** BEFORE RUNNING, EVERY MACHINE NEEDS: ***
  1. The exact same repo commit and `pip install -r requirements.txt` environment.
  2. The exact same dataset laid out at the exact same relative path (e.g. run
     `python data/download_cifake.py --out data/raw/cifake` on every machine --
     DistributedSampler assumes every rank can independently build the identical
     full dataset and just partitions it by index, so the datasets must match).
  3. Mutual network reachability on the chosen port (default 29500). The most
     reliable way to get 3 laptops on different WiFi networks/behind NAT talking
     to each other is a mesh VPN like Tailscale (https://tailscale.com) -- install
     it on all 3, they'll each get a stable 100.x.x.x address, use the "master"
     machine's Tailscale IP as --master_addr below. Raw LAN IPs can work too if
     all 3 are genuinely on the same network and it doesn't do client isolation
     (many venue/guest WiFi networks block this by design), but a Windows firewall
     prompt allowing python.exe through is still likely needed either way.

Usage (run on EACH of the 3 laptops, changing only --node_rank):
    # On the "master" laptop (whichever one you pick -- pick the one whose IP
    # the other two can reach, e.g. its Tailscale address), node_rank 0:
    torchrun --nnodes=3 --nproc_per_node=1 --node_rank=0 \\
        --master_addr=<MASTER_LAPTOP_IP> --master_port=29500 train_ddp.py --config configs/default.yaml

    # On the second laptop, node_rank 1 (same command otherwise):
    torchrun --nnodes=3 --nproc_per_node=1 --node_rank=1 \\
        --master_addr=<MASTER_LAPTOP_IP> --master_port=29500 train_ddp.py --config configs/default.yaml

    # On the third laptop, node_rank 2:
    torchrun --nnodes=3 --nproc_per_node=1 --node_rank=2 \\
        --master_addr=<MASTER_LAPTOP_IP> --master_port=29500 train_ddp.py --config configs/default.yaml

All 3 processes need to start within a few minutes of each other -- torchrun
will wait at startup for all `nnodes` to connect, then run in lockstep.
Checkpoints, the CSV training log, and progress printouts only come from
node_rank 0 (rank 0 overall), so watch that machine's terminal.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.datasets import PairedViewDataset, RealFakeImageDataset
from model.detector import AIGCDetector
from train import (
    build_transform_pipeline,
    collate_paired,
    collate_single,
    evaluate,
    set_seed,
    train_one_epoch,
)


def setup_distributed() -> tuple[int, int, int]:
    """torchrun sets RANK/WORLD_SIZE/LOCAL_RANK env vars automatically -- no need
    to pass them as CLI args. Returns (rank, world_size, local_rank)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # gloo works on Windows/Mac/Linux and supports both CPU and GPU tensors --
    # the portable choice for a mixed/unknown set of laptops. nccl is faster for
    # GPU-to-GPU transfer but is Linux+NVIDIA only (no Windows support at all),
    # so it's not a safe default here even if some machines could use it.
    backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])  # same seed on every rank -> same model init before DDP broadcasts it anyway
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        print(f"[rank {rank}/{world_size}] backend=gloo device={device} "
              f"(each rank trains on a disjoint ~{100 // world_size}% shard of the data per epoch)")

    dataset_roots = [f"data/raw/{name}" for name in cfg["data"]["train_datasets"]]
    full_dataset = RealFakeImageDataset(dataset_roots, transform=None)
    train_ds, val_ds = full_dataset.split_train_val(cfg["data"]["val_split"], seed=cfg["train"]["seed"])

    clean_transform = build_transform_pipeline(cfg, augment=False)
    aug_transform = build_transform_pipeline(cfg, augment=True)

    paired_train_ds = PairedViewDataset(train_ds.samples, clean_transform, aug_transform)
    val_ds.transform = clean_transform

    # DistributedSampler partitions the dataset by index across ranks and
    # reshuffles differently each epoch (via set_epoch below) -- this is what
    # makes DDP a REAL speedup rather than 3 machines redundantly training on
    # the same full dataset. DataLoader's own shuffle must be False when a
    # sampler is given (they're mutually exclusive).
    train_sampler = DistributedSampler(paired_train_ds, num_replicas=world_size, rank=rank, shuffle=True,
                                        seed=cfg["train"]["seed"])
    train_loader = DataLoader(
        paired_train_ds, batch_size=cfg["data"]["batch_size"], sampler=train_sampler,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_paired,
    )
    # Only rank 0 evaluates -- val_loader isn't distributed, keeps eval logic
    # identical to the single-machine path and avoids 3 ranks racing to print/log.
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_single,
    ) if is_main else None

    model = AIGCDetector.from_config(cfg["model"]).to(device)
    ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
    ddp_model = DDP(model, **ddp_kwargs)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    best_val_acc = 0.0
    log_path = ckpt_dir / "training_log.csv"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_accuracy", "lr"])

    for epoch in range(cfg["train"]["epochs"]):
        train_sampler.set_epoch(epoch)  # re-shuffle differently each epoch; without this every epoch sees the same shard order
        # train_one_epoch returns a dict of per-epoch diagnostics (loss, accuracy,
        # mean grad norm, ...), not a bare float.
        train_stats = train_one_epoch(
            ddp_model, train_loader, optimizer, device,
            cfg["train"]["consistency_loss_weight"], cfg["model"]["use_freq_branch"],
            grad_clip_norm=cfg["train"].get("grad_clip_norm", 1.0) or 0.0,
            clip_params=model.trainable_parameters(),  # unwrapped model -- DDP doesn't forward custom methods
        )
        train_loss = train_stats["train_loss"]
        scheduler.step()

        dist.barrier()  # keep ranks roughly in step before rank 0's (potentially slow) eval/save
        if is_main:
            metrics = evaluate(model, val_loader, device, cfg["model"]["use_freq_branch"])  # unwrapped model
            lr = scheduler.get_last_lr()[0]
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_accuracy={metrics['val_accuracy']:.4f} lr={lr:.2e}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, metrics["val_accuracy"], lr])

            if metrics["val_accuracy"] > best_val_acc:
                best_val_acc = metrics["val_accuracy"]
                torch.save(
                    {"model_state": model.state_dict(), "config": cfg, "val_accuracy": best_val_acc},
                    ckpt_dir / "best.pt",
                )
                print(f"  -> saved new best checkpoint (val_accuracy={best_val_acc:.4f})")
        dist.barrier()  # non-main ranks wait for rank 0's eval/save before starting the next epoch

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
