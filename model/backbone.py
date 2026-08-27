"""Frozen CLIP vision backbone wrapper.

Uses a pretrained CLIP vision encoder as a fixed feature extractor. Frozen large
vision-language backbones have been shown to transfer better across unseen
generator families than a detector trained end-to-end on one dataset (Ojha et al.,
"Towards Universal Fake Image Detectors that Generalize Across Generative Models",
CVPR 2023) — the intuition being that CLIP's broad pretraining captures general
image statistics rather than overfitting to one generator's specific artifacts.

Both ViT-B/16 (~86M params) and ViT-L/14 (~300M params) are comfortably under the
challenge's 2B-parameter limit; only the small head (and optional frequency branch)
in `detector.py` are actually trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClipVisionBackbone(nn.Module):
    """Wraps an open_clip vision encoder and exposes a fixed-size embedding.

    Falls back to `transformers.CLIPModel` if `open_clip` isn't installed, so this
    still runs in either environment.
    """

    def __init__(self, backbone_name: str = "ViT-B-16", pretrained: str = "openai", freeze: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self._impl = "open_clip"
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                backbone_name, pretrained=pretrained
            )
            self.model = model.visual
            self.preprocess = preprocess
            self.embed_dim = model.visual.output_dim
        except ImportError:
            self._impl = "transformers"
            from transformers import CLIPImageProcessor, CLIPVisionModel

            hf_name = "openai/clip-vit-base-patch16" if "B" in backbone_name else "openai/clip-vit-large-patch14"
            self.model = CLIPVisionModel.from_pretrained(hf_name)
            self.preprocess = CLIPImageProcessor.from_pretrained(hf_name)
            self.embed_dim = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        self.freeze = freeze

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: preprocessed image batch, shape (B, 3, H, W). Returns (B, embed_dim)."""
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            if self._impl == "open_clip":
                features = self.model(pixel_values)
            else:
                out = self.model(pixel_values=pixel_values)
                features = out.pooler_output
        return features

    def train(self, mode: bool = True):
        # Keep the backbone in eval mode even when the parent module is set to
        # train(), since it's frozen (BatchNorm/Dropout inside CLIP should stay
        # in inference mode regardless of the head's training state).
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self
