"""Post-mortem diagnostics for a finished training run.

Yesterday's run (job 768468) flatlined at chance: train_loss floored at ~0.673
against ln(2)=0.6931, val_accuracy wandered in 0.50-0.58. That is consistent with
BOTH "the model collapsed to a constant prediction" and "there is a real bug in
the model/data path", and accuracy alone cannot tell those apart. Each part below
discriminates between them:

  --part probe    What does the trained checkpoint actually output? A collapsed
                  model has near-zero spread in its predictions and AUC ~= 0.5.
                  A model that learned something weak has AUC > 0.5 even when
                  accuracy at the 0.5 cutoff looks like chance.

  --part overfit  Can the model memorize 256 images with augmentation and the
                  consistency loss switched OFF? Any correctly-wired classifier
                  can drive a 256-image training set to ~100%. If this fails,
                  the bug is in the code. If it passes, the code is fine and the
                  training recipe is what failed.

  --part control  Can a stock ResNet-18 (from scratch, native 32x32, no
                  augmentation) learn this dataset? Published CIFAKE baselines
                  reach ~92-95%. If the control learns and our model doesn't,
                  the data and labels are fine and the fault is ours. If the
                  control also flatlines, the dataset layout/labels are wrong.

Usage (see scripts/diagnose.sbatch):
    python scripts/diagnose.py --part all --data_root /tmp/$USER/raw
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run as `python scripts/diagnose.py`, so sys.path[0] is scripts/, not the repo
# root -- unlike train.py, which sits at the root and imports cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from data.datasets import RealFakeImageDataset
from model.detector import AIGCDetector

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def clean_pipeline(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def collate(batch):
    imgs, labels, _ = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.float32)


def random_subset(dataset, n: int, seed: int = 0) -> Subset:
    """A seeded RANDOM subset -- never a contiguous slice.

    RealFakeImageDataset appends every real image before every fake one, and
    split_train_val preserves that ordering, so `Subset(ds, range(n))` is
    single-class for any n below the real-image count. That silently turned the
    first version of the control run into a one-class training set: it scored a
    perfectly flat val_acc=0.7510 across all epochs at AUC 0.5, which is the
    majority-class rate of the equally-skewed val slice, not a real result.
    """
    n = min(n, len(dataset))
    idx = np.random.default_rng(seed).permutation(len(dataset))[:n]
    return Subset(dataset, idx.tolist())


def class_balance(dataset) -> str:
    labels = [dataset.dataset.samples[i][1] for i in dataset.indices] \
        if isinstance(dataset, Subset) else [l for _, l in dataset.samples]
    n_fake = sum(labels)
    return f"n={len(labels)} real={len(labels) - n_fake} fake={n_fake}"


def load_splits(cfg, data_root):
    roots = [f"{data_root}/{name}" for name in cfg["data"]["train_datasets"]]
    present = [r for r in roots if Path(r).exists()]
    missing = [r for r in roots if not Path(r).exists()]
    if missing:
        print(f"!! config lists datasets that do not exist on disk: {missing}")
        print("   RealFakeImageDataset skips missing folders SILENTLY, so the run")
        print("   trained on fewer datasets than the config claims.")
    full = RealFakeImageDataset(present, transform=None)
    train_ds, val_ds = full.split_train_val(cfg["data"]["val_split"], seed=cfg["train"]["seed"])
    n_fake = sum(lbl for _, lbl in full.samples)
    print(f"datasets present={present}")
    print(f"total={len(full)} real={len(full) - n_fake} fake={n_fake} "
          f"train={len(train_ds)} val={len(val_ds)}")
    return train_ds, val_ds


# --------------------------------------------------------------------------- #
# Part 1: what does the saved checkpoint actually predict?
# --------------------------------------------------------------------------- #

@torch.no_grad()
def probe(ckpt_path: Path, val_ds, device, limit: int, batch_size: int):
    from sklearn.metrics import roc_auc_score

    if not ckpt_path.exists():
        print(f"-- {ckpt_path}: MISSING, skipping")
        return
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = AIGCDetector.from_config(cfg["model"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    val_ds.transform = clean_pipeline(cfg["data"]["image_size"])
    subset = random_subset(val_ds, limit, seed=0)
    loader = DataLoader(subset, batch_size=batch_size, num_workers=4, collate_fn=collate)

    preds, labels = [], []
    for imgs, lbl in loader:
        preds.append(model.predict_proba(imgs.to(device), None).cpu())
        labels.append(lbl)
    p = torch.cat(preds).numpy()
    y = torch.cat(labels).numpy()

    recorded = state.get("val_accuracy", state.get("best_val_acc"))
    epoch = state.get("epoch", "n/a")
    print(f"\n-- {ckpt_path.name} (epoch={epoch}, recorded val_accuracy={recorded})")
    print(f"   n={len(y)}  real={int((y == 0).sum())}  fake={int((y == 1).sum())}")
    print(f"   P(fake): mean={p.mean():.4f} std={p.std():.4f} "
          f"min={p.min():.4f} max={p.max():.4f}")
    print(f"            p01={np.percentile(p, 1):.4f} p50={np.percentile(p, 50):.4f} "
          f"p99={np.percentile(p, 99):.4f}")
    print(f"   mean P(fake) on real images = {p[y == 0].mean():.4f}")
    print(f"   mean P(fake) on fake images = {p[y == 1].mean():.4f}   "
          f"<- separation = {p[y == 1].mean() - p[y == 0].mean():+.4f}")
    print(f"   accuracy@0.5 = {((p > 0.5) == y).mean():.4f}")
    print(f"   AUC          = {roc_auc_score(y, p):.4f}   "
          "<- 0.5 = no signal at all; >0.55 = weak signal a bad threshold is hiding")
    frac_fake = float((p > 0.5).mean())
    print(f"   fraction predicted fake = {frac_fake:.4f}")
    if p.std() < 0.02:
        print("   VERDICT: COLLAPSED -- the model emits a near-constant value for every")
        print("            input. It is not classifying; it found the degenerate minimum.")
    elif abs(roc_auc_score(y, p) - 0.5) < 0.03:
        print("   VERDICT: NO SIGNAL -- predictions vary but carry no class information.")
    else:
        print("   VERDICT: WEAK SIGNAL PRESENT -- the ranking is better than chance.")


# --------------------------------------------------------------------------- #
# Part 2: can the model memorize a tiny set? (tests the code, not the recipe)
# --------------------------------------------------------------------------- #

def overfit(cfg, train_ds, device, n_images: int, steps: int, lr: float, image_size: int):
    print(f"\n-- overfit test: {n_images} images, image_size={image_size}, lr={lr}, "
          f"NO augmentation, NO consistency loss")
    model_cfg = dict(cfg["model"])
    model_cfg["input_image_size"] = image_size
    model = AIGCDetector.from_config(model_cfg).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.0)

    # Balanced tiny subset drawn from the real training split.
    reals = [i for i, (_, l) in enumerate(train_ds.samples) if l == 0][: n_images // 2]
    fakes = [i for i, (_, l) in enumerate(train_ds.samples) if l == 1][: n_images // 2]
    train_ds.transform = clean_pipeline(image_size)
    subset = Subset(train_ds, reals + fakes)
    loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate)

    model.train()
    step = 0
    t0 = time.time()
    while step < steps:
        for imgs, lbl in loader:
            imgs, lbl = imgs.to(device), lbl.to(device)
            logits = model(imgs, None)
            loss = F.binary_cross_entropy_with_logits(logits, lbl)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)  # measure only
            opt.step()
            step += 1
            if step % 20 == 0 or step == 1:
                acc = ((torch.sigmoid(logits) > 0.5).float() == lbl).float().mean()
                print(f"   step {step:4d}  loss={loss.item():.4f}  batch_acc={acc:.3f}  "
                      f"grad_norm={gnorm:.3f}")
            if step >= steps:
                break

    model.eval()
    with torch.no_grad():
        preds, labels = [], []
        for imgs, lbl in DataLoader(subset, batch_size=64, num_workers=4, collate_fn=collate):
            preds.append(model.predict_proba(imgs.to(device), None).cpu())
            labels.append(lbl)
        p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    acc = float(((p > 0.5) == y).mean())
    print(f"   final train accuracy on the {len(y)} memorized images = {acc:.4f} "
          f"({time.time() - t0:.0f}s)")
    if acc > 0.95:
        print("   VERDICT: CODE IS WIRED CORRECTLY. Gradients flow, labels line up,")
        print("            the model has enough capacity. The full run's failure is a")
        print("            TRAINING RECIPE problem, not a bug.")
    elif acc > 0.75:
        print("   VERDICT: PARTIAL. It learns, but far too slowly for a 256-image set --")
        print("            suspect optimization settings (lr/warmup/init).")
    else:
        print("   VERDICT: REAL BUG. A correctly-wired classifier memorizes 256 images.")
        print("            Look at label plumbing, normalization, and the head.")
    return acc


# --------------------------------------------------------------------------- #
# Part 3: independent control -- is the DATA learnable at all?
# --------------------------------------------------------------------------- #

def control(train_ds, val_ds, device, epochs: int, batch_size: int, limit: int):
    from sklearn.metrics import roc_auc_score
    from torchvision.models import resnet18

    print(f"\n-- control: stock ResNet-18 from scratch, native 32x32, no augmentation")
    model = resnet18(weights=None, num_classes=1)
    # CIFAR-style stem: the ImageNet 7x7/stride-2 + maxpool throws away a 32x32 image.
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    tf = clean_pipeline(32)
    train_ds.transform = tf
    val_ds.transform = tf
    tr = random_subset(train_ds, limit, seed=0)
    va = random_subset(val_ds, limit // 5, seed=1)
    print(f"   train subset: {class_balance(tr)}")
    print(f"   val subset:   {class_balance(va)}")
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate)
    vl = DataLoader(va, batch_size=batch_size, num_workers=8, collate_fn=collate)

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for imgs, lbl in tl:
            imgs, lbl = imgs.to(device), lbl.to(device)
            loss = F.binary_cross_entropy_with_logits(model(imgs).squeeze(-1), lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for imgs, lbl in vl:
                preds.append(torch.sigmoid(model(imgs.to(device)).squeeze(-1)).cpu())
                labels.append(lbl)
            p, y = torch.cat(preds).numpy(), torch.cat(labels).numpy()
        print(f"   epoch {ep}: val_acc={((p > 0.5) == y).mean():.4f} "
              f"AUC={roc_auc_score(y, p):.4f} ({time.time() - t0:.0f}s)")

    acc = ((p > 0.5) == y).mean()
    if acc > 0.85:
        print("   VERDICT: THE DATA IS FINE AND LEARNABLE. A plain CNN at native")
        print("            resolution separates these classes easily. Yesterday's")
        print("            failure is entirely on our model/recipe side.")
    elif acc > 0.65:
        print("   VERDICT: DATA CARRIES SIGNAL but less than published CIFAKE baselines")
        print("            (~92-95%). Check the real/fake folder layout.")
    else:
        print("   VERDICT: THE DATA OR LABELS ARE BROKEN. Nothing can be trained on this;")
        print("            inspect data/prepare_data.py's real/fake assignment first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default="data/raw")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Override data.train_datasets from the config. Lets you run a "
                         "config's MODEL against a different dataset -- e.g. the 224px "
                         "smoke model against real CIFAKE, to tell a model-config bug "
                         "apart from unlearnable data.")
    ap.add_argument("--part", default="all", choices=["all", "probe", "overfit", "control"])
    ap.add_argument("--probe_limit", type=int, default=4000)
    ap.add_argument("--overfit_images", type=int, default=256)
    ap.add_argument("--overfit_steps", type=int, default=300)
    ap.add_argument("--overfit_lr", type=float, default=3e-4)
    ap.add_argument("--overfit_image_size", type=int, default=None,
                    help="Defaults to data.image_size from the config.")
    ap.add_argument("--gate", action="store_true",
                    help="Exit non-zero if the overfit test fails. Used to gate the "
                         "long training job -- see scripts/train.sbatch.")
    ap.add_argument("--gate_min_acc", type=float, default=0.95)
    ap.add_argument("--control_epochs", type=int, default=3)
    ap.add_argument("--control_limit", type=int, default=40000)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")

    if args.datasets:
        print(f"overriding train_datasets: {cfg['data']['train_datasets']} -> {args.datasets}")
        cfg["data"]["train_datasets"] = args.datasets
    train_ds, val_ds = load_splits(cfg, args.data_root)
    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])

    if args.part in ("all", "probe"):
        print("\n" + "=" * 72)
        print("PART 1 -- what the trained checkpoints actually predict")
        print("=" * 72)
        for name in ("best.pt", "last.pt"):
            probe(ckpt_dir / name, val_ds, device, args.probe_limit, cfg["data"]["batch_size"])

    if args.part in ("all", "overfit"):
        print("\n" + "=" * 72)
        print("PART 2 -- can the model memorize a tiny set? (tests the CODE)")
        print("=" * 72)
        image_size = args.overfit_image_size or cfg["data"]["image_size"]
        acc = overfit(cfg, train_ds, device, args.overfit_images, args.overfit_steps,
                      args.overfit_lr, image_size)
        if args.gate and acc < args.gate_min_acc:
            print(
                f"\nGATE FAILED: the model reached only {acc:.4f} on {args.overfit_images} "
                f"images (need >= {args.gate_min_acc}).\nA model that cannot memorize a "
                f"tiny training set will not learn 100k images either -- refusing to start "
                f"the full run.",
                file=sys.stderr,
            )
            sys.exit(3)
        if args.gate:
            print(f"\nGATE PASSED: {acc:.4f} >= {args.gate_min_acc}. Safe to start the full run.")

    if args.part in ("all", "control"):
        print("\n" + "=" * 72)
        print("PART 3 -- independent control: is the DATA learnable?")
        print("=" * 72)
        control(train_ds, val_ds, device, args.control_epochs,
                cfg["data"]["batch_size"], args.control_limit)


if __name__ == "__main__":
    main()
