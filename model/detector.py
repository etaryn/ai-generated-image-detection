"""Combines the frozen CLIP backbone (+ optional frequency branch) with the
trainable classification head into a single detector model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.backbone import ClipVisionBackbone
from model.freq_branch import FrequencyBranch
from model.head import ClassificationHead


class AIGCDetector(nn.Module):
    def __init__(
        self,
        backbone_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_backbone: bool = True,
        use_freq_branch: bool = False,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = ClipVisionBackbone(backbone_name, pretrained, freeze=freeze_backbone)
        self.use_freq_branch = use_freq_branch
        feat_dim = self.backbone.embed_dim

        if use_freq_branch:
            self.freq_branch = FrequencyBranch(out_dim=128)
            feat_dim += 128
        else:
            self.freq_branch = None

        self.head = ClassificationHead(feat_dim, head_hidden_dim, head_dropout)

    @classmethod
    def from_config(cls, model_cfg: dict) -> "AIGCDetector":
        return cls(
            backbone_name=model_cfg["backbone_name"],
            pretrained=model_cfg["pretrained"],
            freeze_backbone=model_cfg["freeze_backbone"],
            use_freq_branch=model_cfg["use_freq_branch"],
            head_hidden_dim=model_cfg["head_hidden_dim"],
            head_dropout=model_cfg["head_dropout"],
        )

    def forward(self, clip_pixel_values: torch.Tensor, raw_pixel_values: torch.Tensor | None = None) -> torch.Tensor:
        """
        clip_pixel_values: CLIP-preprocessed batch (resized/normalized to what the
            backbone expects) — (B, 3, H_clip, W_clip).
        raw_pixel_values: only needed when use_freq_branch=True; a [0,1]-scaled
            batch at a consistent resolution (e.g. 224x224) for the DCT branch —
            (B, 3, H, W).
        Returns: (B,) logits (positive = more likely AI-generated).
        """
        feats = self.backbone(clip_pixel_values)
        if self.use_freq_branch:
            if raw_pixel_values is None:
                raise ValueError("use_freq_branch=True requires raw_pixel_values")
            freq_feats = self.freq_branch(raw_pixel_values)
            feats = torch.cat([feats, freq_feats], dim=-1)
        return self.head(feats)

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Convenience: forward() + sigmoid -> P(fake) in [0, 1]."""
        return torch.sigmoid(self.forward(*args, **kwargs))

    def trainable_parameters(self):
        """Parameters that actually get gradient updates (head + optional freq branch)."""
        params = list(self.head.parameters())
        if self.freq_branch is not None:
            params += list(self.freq_branch.parameters())
        return params
