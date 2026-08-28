"""Combines a chosen image-feature backbone with the trainable classification head.

Two architectures are supported via `model.architecture` in the config:

- "cnn_transformer" (default, current team decision): a from-scratch, fully
  trainable CNN stem (cnn_stem.py) that produces a grid of local feature tokens,
  fed into a Transformer encoder (transformer_encoder.py) that self-attends over
  those tokens. The CNN half is well suited to the local, low-level artifacts
  synthetic images tend to leave; the Transformer half adds global reasoning
  across regions on top of that. Fully trainable end-to-end -- no pretrained
  component to lean on, so expect to need more data/epochs than the frozen-CLIP
  path, but it can specialize directly on the AIGC-artifact task.
- "clip_frozen": a frozen CLIP vision encoder (backbone.py) + trainable head.
  Kept as an alternative/baseline -- cheap to train (only the head has
  gradients) and transfers well across unseen generators, at the cost of being
  capped by CLIP's own biases/resolution.

Both stay far under the challenge's 2B-parameter limit (cnn_transformer's default
channel/depth schedule is roughly 14M trainable parameters).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.backbone import ClipVisionBackbone
from model.cnn_stem import ConvStem
from model.freq_branch import FrequencyBranch
from model.head import ClassificationHead
from model.transformer_encoder import TokenTransformer


class AIGCDetector(nn.Module):
    def __init__(
        self,
        architecture: str = "cnn_transformer",
        # --- clip_frozen args ---
        backbone_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_backbone: bool = True,
        # --- cnn_transformer args ---
        cnn_channels: list[int] | tuple[int, ...] = (64, 128, 256, 384),
        transformer_dim: int = 384,
        transformer_depth: int = 6,
        transformer_heads: int = 6,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        input_image_size: int = 224,
        # --- shared ---
        use_freq_branch: bool = False,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.architecture = architecture

        if architecture == "clip_frozen":
            self.backbone = ClipVisionBackbone(backbone_name, pretrained, freeze=freeze_backbone)
            self.cnn_stem = None
            self.transformer = None
            feat_dim = self.backbone.embed_dim

        elif architecture == "cnn_transformer":
            self.backbone = None
            self.cnn_stem = ConvStem(in_channels=3, channels=list(cnn_channels))
            stride = 2 ** len(cnn_channels)
            if input_image_size % stride != 0:
                raise ValueError(
                    f"input_image_size ({input_image_size}) must be divisible by the "
                    f"CNN stem's total stride (2**{len(cnn_channels)}={stride})"
                )
            grid = input_image_size // stride
            self.transformer = TokenTransformer(
                in_channels=self.cnn_stem.out_dim,
                num_patches=grid * grid,
                dim=transformer_dim,
                depth=transformer_depth,
                heads=transformer_heads,
                mlp_ratio=transformer_mlp_ratio,
                dropout=transformer_dropout,
            )
            feat_dim = self.transformer.out_dim

        else:
            raise ValueError(f"Unknown architecture '{architecture}' (expected 'cnn_transformer' or 'clip_frozen')")

        self.use_freq_branch = use_freq_branch
        if use_freq_branch:
            self.freq_branch = FrequencyBranch(out_dim=128)
            feat_dim += 128
        else:
            self.freq_branch = None

        self.head = ClassificationHead(feat_dim, head_hidden_dim, head_dropout)

    @classmethod
    def from_config(cls, model_cfg: dict) -> "AIGCDetector":
        architecture = model_cfg.get("architecture", "cnn_transformer")
        common = dict(
            architecture=architecture,
            use_freq_branch=model_cfg["use_freq_branch"],
            head_hidden_dim=model_cfg["head_hidden_dim"],
            head_dropout=model_cfg["head_dropout"],
        )
        if architecture == "clip_frozen":
            clip_cfg = model_cfg["clip_frozen"]
            return cls(
                backbone_name=clip_cfg["backbone_name"],
                pretrained=clip_cfg["pretrained"],
                freeze_backbone=clip_cfg["freeze_backbone"],
                **common,
            )
        ct_cfg = model_cfg["cnn_transformer"]
        return cls(
            cnn_channels=ct_cfg["stem_channels"],
            transformer_dim=ct_cfg["transformer_dim"],
            transformer_depth=ct_cfg["transformer_depth"],
            transformer_heads=ct_cfg["transformer_heads"],
            transformer_mlp_ratio=ct_cfg["transformer_mlp_ratio"],
            transformer_dropout=ct_cfg["dropout"],
            input_image_size=model_cfg.get("input_image_size", 224),
            **common,
        )

    def forward(self, pixel_values: torch.Tensor, raw_pixel_values: torch.Tensor | None = None) -> torch.Tensor:
        """
        pixel_values: normalized batch at the resolution the chosen architecture
            expects -- (B, 3, H, W).
        raw_pixel_values: only needed when use_freq_branch=True; a [0,1]-scaled
            batch at a consistent resolution for the DCT branch -- (B, 3, H, W).
        Returns: (B,) logits (positive = more likely AI-generated).
        """
        if self.architecture == "clip_frozen":
            feats = self.backbone(pixel_values)
        else:
            feature_map = self.cnn_stem(pixel_values)
            feats = self.transformer(feature_map)

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
        """Parameters that actually get gradient updates.

        "clip_frozen": just the head (+ optional freq branch) -- the backbone is
        frozen. "cnn_transformer": the whole model, since there's no pretrained
        component -- everything trains end-to-end.
        """
        if self.architecture == "clip_frozen":
            params = list(self.head.parameters())
            if self.freq_branch is not None:
                params += list(self.freq_branch.parameters())
            return params
        return list(self.parameters())
