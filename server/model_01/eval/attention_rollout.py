"""Attention-rollout explainability for the CNN + Transformer hybrid.

Combines the Transformer encoder's per-layer self-attention maps into a single
saliency map showing which image regions most influenced the [CLS] token's
final representation -- i.e. which regions drove the "AI-generated" decision.
Implements the rollout method from Abnar & Zuidema, "Quantifying Attention Flow
in Transformers" (2020). This is the explainability angle named for this
architecture in the technical-approach notes (attention weights are already
exposed by the model, unlike Grad-CAM which needs gradient hooks into conv
layers).

Only applies to `model.architecture: cnn_transformer` checkpoints -- the
`clip_frozen` baseline doesn't expose attention weights through this path.

Usage:
    python eval/attention_rollout.py --checkpoint checkpoints/best.pt \\
        --image path/to/image.jpg --output rollout.png
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from model.detector import AIGCDetector

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def compute_attention_rollout(attn_maps: list[torch.Tensor]) -> torch.Tensor:
    """attn_maps: list of (B, N, N) head-averaged attention matrices, one per
    Transformer layer (N = 1 + num_patches, position 0 = [CLS]).

    Returns (B, N): how much each token contributes to the final [CLS]
    representation, propagated back through every layer.
    """
    b, n, _ = attn_maps[0].shape
    identity = torch.eye(n, device=attn_maps[0].device).unsqueeze(0).expand(b, -1, -1)
    rollout = identity
    for attn in attn_maps:
        # Account for the residual connection around attention: a token's next
        # representation is (attention-weighted sum) + itself, so mix in identity
        # before renormalizing (Abnar & Zuidema, Sec. 3).
        attn_with_residual = 0.5 * attn + 0.5 * identity
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        rollout = attn_with_residual @ rollout
    return rollout[:, 0, :]  # CLS row: contribution of every token to the CLS output


def explain_image(model: AIGCDetector, img: Image.Image, image_size: int, device: torch.device):
    """Returns (prob_fake, patch_grid_saliency) for a single PIL image."""
    if model.architecture != "cnn_transformer":
        raise ValueError("Attention rollout requires model.architecture == 'cnn_transformer'")

    preprocess = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature_map = model.cnn_stem(x)
        pooled, attn_maps = model.transformer(feature_map, return_attention=True)
        prob_fake = torch.sigmoid(model.head(pooled)).item()

    rollout = compute_attention_rollout(attn_maps)[0]  # (N,)
    patch_scores = rollout[1:]  # drop the CLS-to-CLS entry
    grid = int(round(patch_scores.numel() ** 0.5))
    heatmap = patch_scores.reshape(grid, grid).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return prob_fake, heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="Path to a single image to explain")
    parser.add_argument("--output", default="rollout.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = AIGCDetector.from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    prob_fake, heatmap = explain_image(model, img, cfg["data"]["image_size"], device)

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    heatmap_arr = np.asarray(heatmap_img) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[1].imshow(heatmap_arr, cmap="jet")
    axes[1].set_title("Attention rollout")
    axes[2].imshow(img)
    axes[2].imshow(heatmap_arr, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay -- P(AI-generated)={prob_fake:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"P(AI-generated) = {prob_fake:.3f}")
    print(f"Wrote visualization to {args.output}")


if __name__ == "__main__":
    main()
