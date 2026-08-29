"""Step 1a: DINOv2 features -- "numbers describing texture / structure".

DINOv2 is self-supervised: it was never trained to name what's in an image, only
to produce representations where visually/structurally similar patches agree.
That makes its embedding a good description of *how an image is built* -- surface
texture, material, edge and part structure -- which is where diffusion output
tends to differ from a photograph (over-smooth skin and fabric, mushy fine
texture, subtly inconsistent structure) even when the picture reads as plausible.

Pooling: the [CLS] token summarizes the whole image; the mean over patch tokens
keeps local texture evidence that [CLS] may average away. `cls_mean` (default)
concatenates both, giving 2 x hidden_size numbers.

Frozen -- no gradients, no fine-tuning. `facebook/dinov2-base` is ~86M params.
"""
from __future__ import annotations

import numpy as np
import torch

from features.base import FeatureExtractor, resize_and_normalize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# DINOv2 uses 14x14 patches, so the input side must be a multiple of 14.
PATCH_SIZE = 14


class DinoV2Features(FeatureExtractor):
    name = "dino"

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        image_size: int = 224,
        pooling: str = "cls_mean",
        l2_normalize: bool = True,
        device: str | torch.device = "cpu",
    ):
        if pooling not in ("cls", "mean", "cls_mean"):
            raise ValueError(f"pooling must be 'cls' | 'mean' | 'cls_mean', got {pooling!r}")
        if image_size % PATCH_SIZE != 0:
            raise ValueError(
                f"DINOv2 uses {PATCH_SIZE}x{PATCH_SIZE} patches, so features.dinov2.image_size "
                f"must be a multiple of {PATCH_SIZE} (got {image_size}; 224 or 238 are fine)."
            )
        self.model_name = model_name
        self.image_size = image_size
        self.pooling = pooling
        self.l2_normalize = l2_normalize
        self.device = torch.device(device)

        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - environment problem, not logic
            raise ImportError(
                "DINOv2 features need `transformers` (pip install -r requirements.txt), "
                "or set features.dinov2.enabled: false in the config."
            ) from exc

        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        hidden = self.model.config.hidden_size
        self.hidden = hidden
        self.dim = hidden * 2 if pooling == "cls_mean" else hidden

    @torch.no_grad()
    def __call__(self, canonical: torch.Tensor) -> np.ndarray:
        x = resize_and_normalize(
            canonical.to(self.device), self.image_size, IMAGENET_MEAN, IMAGENET_STD
        )
        tokens = self.model(pixel_values=x).last_hidden_state  # (B, 1 + n_patches, hidden)
        cls_token = tokens[:, 0]
        patch_mean = tokens[:, 1:].mean(dim=1)

        if self.pooling == "cls":
            feats = cls_token
        elif self.pooling == "mean":
            feats = patch_mean
        else:
            feats = torch.cat([cls_token, patch_mean], dim=-1)

        if self.l2_normalize:
            # Unit-norm each block so DINOv2's raw activation scale doesn't
            # dominate the concatenated vector relative to CLIP and the FFT stats.
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.float().cpu().numpy()

    def feature_names(self) -> list[str]:
        if self.pooling == "cls_mean":
            return [f"dino_cls_{i}" for i in range(self.hidden)] + [
                f"dino_patchmean_{i}" for i in range(self.hidden)
            ]
        return [f"dino_{self.pooling}_{i}" for i in range(self.hidden)]

    def signature(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "model_name": self.model_name,
            "image_size": self.image_size,
            "pooling": self.pooling,
            "l2_normalize": self.l2_normalize,
        }
