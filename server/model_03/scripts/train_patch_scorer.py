"""Fine-tune the patch scorer on patches, which is the project's real ceiling.

Every localisation number model_03 has produced is bounded by one mismatch: the
backend was trained to answer "is this *image* generated?" and the mapper asks it
"is this 64px *fragment* generated?" a thousand times per image. Measured, that
mismatch is not subtle -- the resulting heatmap has a per-pixel AUC of 0.460
against ground-truth masks, below chance, and correlates with local smoothness
(-0.25) about equally on tampered and authentic images. The map is not weakly
localising; it is not localising at all. No threshold, calibration or region
rule downstream can recover information the map never had, which two failed
attempts (per-scale calibration, coarse-to-fine cascade) demonstrated
empirically.

This script trains the missing component: a classifier whose training
distribution *is* the mapper's inference distribution.

**Labelling is the whole design, so it is worth being explicit.** A patch's
label comes from the tamper mask, not from its image:

    real image        every patch is authentic          -> 0
    synthetic image   every patch is generated          -> 1
    tampered image    label by mask coverage of the patch

For tampered images a patch counts as generated when at least `--positive_frac`
of its area falls inside the mask, and as authentic when at most
`--negative_frac` does. **Patches in between are discarded rather than assigned
to either side.** That band is the point: a patch straddling an edit boundary is
genuinely half-and-half, and forcing it to one label teaches the model that
boundary content looks like whichever side it was arbitrarily assigned. Dropping
it costs training data and buys a decision boundary that means something.

Sampling is balanced per image and re-drawn every epoch, so the model sees many
different crops of the same content rather than memorising a fixed set.

**What "working" means here is not patch accuracy.** A model can score patches
well and still produce a useless map, because what the mapper needs is that
patches *inside* an edit score higher than patches *outside* it in the same
image -- a within-image ranking, not a global one. So validation reports
per-pixel map AUC on held-out images alongside patch AUC, and it is the former
that decides whether this was worth doing.

    python scripts/train_patch_scorer.py \
        --data eval_data/sid_set_train --out checkpoints/patch_scorer

Then point a config at it:

    backend: {name: hf, model_id: server/model_03/checkpoints/patch_scorer}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_manifest(root: Path) -> list[dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    return manifest["items"]


class PatchSampler:
    """Draws labelled patches from SID-Set images, balanced and re-drawn each epoch."""

    def __init__(
        self,
        root: Path,
        items: list[dict],
        scales: list[int],
        per_image: int,
        positive_frac: float,
        negative_frac: float,
        seed: int = 0,
    ):
        self.root = root
        self.items = items
        self.scales = scales
        self.per_image = per_image
        self.positive_frac = positive_frac
        self.negative_frac = negative_frac
        self.rng = random.Random(seed)

    def _label_for(self, row: dict, mask: np.ndarray | None, box) -> int | None:
        if row["class"] == "real":
            return 0
        if row["class"] == "synthetic":
            return 1
        if mask is None:
            return None
        x0, y0, x1, y1 = box
        window = mask[y0:y1, x0:x1]
        if window.size == 0:
            return None
        coverage = float(window.mean())
        if coverage >= self.positive_frac:
            return 1
        if coverage <= self.negative_frac:
            return 0
        return None  # the ambiguous band: dropped on purpose

    def draw(self, row: dict) -> list[tuple[Image.Image, int]]:
        from mapper.windows import plan_windows

        path = self.root / row["image"]
        with Image.open(path) as handle:
            image = handle.convert("RGB")

        mask = None
        if row.get("mask"):
            with Image.open(self.root / row["mask"]) as handle:
                mask = np.asarray(handle.convert("L").resize(image.size, Image.NEAREST)) > 127

        plan = plan_windows(image.width, image.height, self.scales, overlap=0.5)
        windows = [w for group in plan.values() for w in group]
        self.rng.shuffle(windows)

        positives, negatives = [], []
        for window in windows:
            label = self._label_for(row, mask, window.box)
            if label is None:
                continue
            bucket = positives if label == 1 else negatives
            if len(bucket) < self.per_image:
                bucket.append((image.crop(window.box), label))
            if len(positives) >= self.per_image and len(negatives) >= self.per_image:
                break

        # Balance within the image, so a tampered image with a tiny edit cannot
        # flood the batch with negatives that happen to share its content.
        keep = min(len(positives), len(negatives)) if row["class"] == "tampered" else self.per_image
        return positives[:keep] + negatives[:keep] if row["class"] == "tampered" else (positives + negatives)[:keep]


def batched(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def evaluate_patches(model, processor, device, patches, labels, batch_size=64) -> float:
    """Patch-level AUC, reading P(AI) from the class the model *says* is AI.

    The index is resolved from the model's own id2label, never hardcoded. An
    earlier version of this function took logit index 1, which on a model whose
    AI class sits at index 0 reported a well-trained scorer as AUC 0.02 -- the
    exact inversion mapper/labels.py exists to prevent, reintroduced here by
    hand. Reading a *trained* model as broken is the benign direction of that
    mistake; the other direction ships an inverted detector.
    """
    import torch

    from eval.evaluate import auc
    from mapper.labels import resolve_positive_indices

    positive, _ = resolve_positive_indices(model.config.id2label)

    model.eval()
    scores = []
    with torch.no_grad():
        for chunk in batched(patches, batch_size):
            inputs = processor(images=list(chunk), return_tensors="pt").to(device)
            probs = model(**inputs).logits.float().softmax(-1)
            scores.extend(probs[:, positive].sum(dim=-1).cpu().tolist())
    scores = np.array(scores)
    labels = np.array(labels)
    return auc(scores[labels == 1], scores[labels == 0])


def evaluate_objective(model, processor, device, root: Path, items: list[dict],
                       scales: list[int], limit: int = 40, per_image: int = 12) -> dict:
    """AUC(real vs AI) at image level -- the actual objective.

    The system makes one binary call: real or AI. Patch AUC does not measure it,
    and per-pixel map AUC measures only *within-image* ranking, which a model can
    ace while every real photograph still floats above the decision threshold --
    exactly the failure observed after the first training run (real images
    averaging 0.537 against partially-AI images at 0.311).

    So this reduces each validation image to a single score the way the pipeline
    effectively does -- a high percentile over a patch grid -- and asks whether
    real images separate from AI ones. Both AI subsets are reported, because the
    partially-AI subset is the one a whole-image detector cannot see and the
    whole reason the region machinery exists.
    """
    import torch

    from eval.evaluate import auc
    from mapper.labels import resolve_positive_indices
    from mapper.windows import plan_windows

    positive, _ = resolve_positive_indices(model.config.id2label)
    model.eval()
    rng = random.Random(0)

    scores: dict[str, list[float]] = {"real": [], "synthetic": [], "tampered": []}
    picked: dict[str, int] = {}
    with torch.no_grad():
        for row in items:
            cls = row["class"]
            if picked.get(cls, 0) >= limit:
                continue
            with Image.open(root / row["image"]) as handle:
                image = handle.convert("RGB")
            plan = plan_windows(image.width, image.height, scales, overlap=0.5)
            windows = [w for group in plan.values() for w in group]
            rng.shuffle(windows)
            crops = [image.crop(w.box) for w in windows[:per_image]]
            if not crops:
                continue
            inputs = processor(images=crops, return_tensors="pt").to(device)
            probs = model(**inputs).logits.float().softmax(-1)
            patch_scores = probs[:, positive].sum(dim=-1).cpu().numpy()
            scores[cls].append(float(np.percentile(patch_scores, 90)))
            picked[cls] = picked.get(cls, 0) + 1

    real = np.array(scores["real"])
    full = np.array(scores["synthetic"])
    part = np.array(scores["tampered"])
    ai = np.concatenate([full, part]) if full.size or part.size else np.array([])
    if real.size == 0 or ai.size == 0:
        return {}
    return {
        "auc_real_vs_ai": auc(ai, real),
        "auc_real_vs_fully_ai": auc(full, real) if full.size else float("nan"),
        "auc_real_vs_partially_ai": auc(part, real) if part.size else float("nan"),
        "mean_real": float(real.mean()),
        "mean_partially_ai": float(part.mean()) if part.size else float("nan"),
    }


def mine_hard_negatives(model, processor, device, sampler, root: Path,
                        real_items: list[dict], want: int, batch_size: int = 64):
    """Real-image patches the model currently scores highest, as extra negatives.

    Every false positive the system produces is a patch of an authentic
    photograph scoring high. Random negatives mostly teach what the model
    already knows; the patches it gets wrong are the ones carrying information,
    and re-showing them is the cheapest available pressure on the false-positive
    rate -- the failing condition under the real-vs-AI objective.
    """
    import torch

    from mapper.labels import resolve_positive_indices

    if want <= 0 or not real_items:
        return []

    positive, _ = resolve_positive_indices(model.config.id2label)
    candidates: list[Image.Image] = []
    for row in real_items:
        candidates.extend(patch for patch, _ in sampler.draw(row))
        if len(candidates) >= want * 4:
            break
    if not candidates:
        return []

    model.eval()
    scored = []
    with torch.no_grad():
        for chunk in batched(candidates, batch_size):
            inputs = processor(images=list(chunk), return_tensors="pt").to(device)
            probs = model(**inputs).logits.float().softmax(-1)
            scored.extend(probs[:, positive].sum(dim=-1).cpu().tolist())

    order = np.argsort(scored)[::-1][:want]
    return [(candidates[i], 0) for i in order]


def evaluate_map(model_dir: Path, root: Path, items: list[dict], limit: int = 12) -> dict:
    """The metric that actually decides this: does the map localise?"""
    from analyze import load_config
    from eval.evaluate import auc
    from mapper.backends import build_backend
    from mapper.heatmap import AILikelihoodMapper

    config = load_config()
    scorer = build_backend(str(model_dir), batch_size=64)
    mapper = AILikelihoodMapper(
        scorer=scorer,
        scales=config["mapper"]["scales"],
        overlap=config["mapper"]["overlap"],
        max_side=config["mapper"]["max_side"],
        scale_combine=config["mapper"]["scale_combine"],
    )

    aucs = []
    rng = np.random.default_rng(0)
    for row in [r for r in items if r["class"] == "tampered" and r.get("mask")][:limit]:
        with Image.open(root / row["image"]) as handle:
            image = handle.convert("RGB")
        amap = mapper.run(image)
        with Image.open(root / row["mask"]) as handle:
            mask = np.asarray(handle.convert("L").resize(amap.working_size, Image.NEAREST)) > 127
        heat = amap.heat
        inside = heat[mask]
        outside = heat[~mask]
        inside = inside[~np.isnan(inside)]
        outside = outside[~np.isnan(outside)]
        if inside.size < 50 or outside.size < 50:
            continue
        aucs.append(
            auc(
                inside[rng.integers(0, inside.size, min(4000, inside.size))],
                outside[rng.integers(0, outside.size, min(4000, outside.size))],
            )
        )
    return {"n": len(aucs), "mean_pixel_auc": float(np.mean(aucs)) if aucs else float("nan")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="eval_data/sid_set_train")
    parser.add_argument("--out", default="checkpoints/patch_scorer")
    parser.add_argument("--base_model", default="Organika/sdxl-detector",
                        help="Starting weights. Fine-tuning an AI-detector beats training "
                             "from scratch on this much data.")
    parser.add_argument("--scales", type=int, nargs="+", default=[64, 128, 224])
    parser.add_argument("--per_image", type=int, default=4, help="Patches per class per image per epoch")
    parser.add_argument("--positive_frac", type=float, default=0.70,
                        help="Mask coverage at or above which a patch counts as generated")
    parser.add_argument("--negative_frac", type=float, default=0.05,
                        help="Mask coverage at or below which a patch counts as authentic; "
                             "patches in between are dropped, not guessed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit_images", type=int, default=None)
    parser.add_argument("--hard_negatives", type=int, default=512,
                        help="Real-image patches the model scores highest, re-shown as "
                             "negatives next epoch. Every false positive is one of these, "
                             "and false positives are the failing condition under the "
                             "real-vs-AI objective. 0 disables.")
    args = parser.parse_args()

    import torch
    from torch.optim import AdamW
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    root = Path(args.data)
    items = load_manifest(root)
    if args.limit_images:
        items = items[: args.limit_images]

    rng = random.Random(args.seed)
    rng.shuffle(items)
    split = max(1, int(len(items) * args.val_frac))
    val_items, train_items = items[:split], items[split:]
    print(f"{len(train_items)} training images, {len(val_items)} validation images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(args.base_model).to(device)

    # Keep the base model's label order so the resolver in mapper/labels.py
    # continues to identify the AI class by name rather than by index.
    id2label = dict(model.config.id2label)
    ai_index = [i for i, name in id2label.items() if "artificial" in str(name).lower() or "ai" == str(name).lower()]
    if ai_index and ai_index[0] != 1:
        # Our labels are 0 = authentic, 1 = generated; if the base model puts
        # the AI class at index 0, flip our target so we do not train against
        # the head's own convention.
        print(f"base model AI class is index {ai_index[0]}; targets flipped to match")
        flip = True
    else:
        flip = False

    sampler = PatchSampler(
        root, train_items, args.scales, args.per_image,
        args.positive_frac, args.negative_frac, seed=args.seed,
    )
    val_sampler = PatchSampler(
        root, val_items, args.scales, args.per_image,
        args.positive_frac, args.negative_frac, seed=args.seed + 1,
    )

    print("building the validation patch set ...")
    val_patches, val_labels = [], []
    for row in val_items:
        for patch, label in val_sampler.draw(row):
            val_patches.append(patch)
            val_labels.append(label)
    print(f"  {len(val_patches)} validation patches ({sum(val_labels)} positive)")

    optimiser = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    baseline = evaluate_patches(model, processor, device, val_patches, val_labels)
    print(f"patch AUC before training: {baseline:.4f}")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    real_items = [r for r in train_items if r["class"] == "real"]
    hard_negatives: list[tuple[Image.Image, int]] = []

    for epoch in range(args.epochs):
        model.train()
        order = list(train_items)
        rng.shuffle(order)

        # Carry the previous epoch's worst false positives back in.
        pool: list[tuple[Image.Image, int]] = list(hard_negatives)
        losses = []
        iterator = tqdm(order, desc=f"epoch {epoch + 1}/{args.epochs}", unit="img") if tqdm else order

        for row in iterator:
            pool.extend(sampler.draw(row))
            if len(pool) < args.batch_size:
                continue

            rng.shuffle(pool)
            batch, pool = pool[: args.batch_size], pool[args.batch_size :]
            images = [p for p, _ in batch]
            targets = torch.tensor(
                [(1 - l) if flip else l for _, l in batch], device=device, dtype=torch.long
            )

            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(**inputs).logits
                loss = torch.nn.functional.cross_entropy(logits, targets)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            losses.append(float(loss.detach()))
            if tqdm and losses:
                iterator.set_postfix({"loss": f"{np.mean(losses[-50:]):.4f}"})

        patch_auc = evaluate_patches(model, processor, device, val_patches, val_labels)
        objective = evaluate_objective(
            model, processor, device, root, val_items, args.scales
        )
        print(
            f"epoch {epoch + 1}: loss {np.mean(losses):.4f}  patch AUC {patch_auc:.4f}  "
            f"AUC(real vs AI) {objective.get('auc_real_vs_ai', float('nan')):.4f}  "
            f"[partial {objective.get('auc_real_vs_partially_ai', float('nan')):.4f}, "
            f"real {objective.get('mean_real', float('nan')):.3f} vs "
            f"partial {objective.get('mean_partially_ai', float('nan')):.3f}]",
            flush=True,
        )

        if args.hard_negatives > 0 and epoch + 1 < args.epochs:
            hard_negatives = mine_hard_negatives(
                model, processor, device, sampler, root, real_items, args.hard_negatives
            )
            print(f"  mined {len(hard_negatives)} hard negatives from real images", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    (out_dir / "training.json").write_text(json.dumps({
        "base_model": args.base_model,
        "scales": args.scales,
        "positive_frac": args.positive_frac,
        "negative_frac": args.negative_frac,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_images": len(train_items),
        "val_patches": len(val_patches),
        "patch_auc_before": baseline,
        "patch_auc_after": patch_auc,
        "flip": flip,
    }, indent=2))
    print(f"\nwrote {out_dir}")

    print("\nscoring the map, which is what this was for ...")
    before = {"mean_pixel_auc": 0.460, "note": "measured earlier for the base model"}
    after = evaluate_map(out_dir, root, val_items)
    print(f"  per-pixel map AUC: base {before['mean_pixel_auc']:.3f} -> tuned {after['mean_pixel_auc']:.3f}")
    if after["mean_pixel_auc"] <= 0.55:
        print("  This is still at or near chance. Patch accuracy without map AUC means the\n"
              "  model separates images, not regions within an image -- do not ship it as a\n"
              "  localiser on this evidence.")


if __name__ == "__main__":
    main()
