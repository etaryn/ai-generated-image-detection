"""Transformer encoder stage that sits on top of the CNN stem's feature grid.

Turns the CNN stem's (B, C, H', W') feature map into a sequence of tokens, adds a
learnable [CLS] token and learnable positional embeddings, and runs a standard
multi-head self-attention encoder over it. Self-attention lets every local patch
"compare notes" with every other patch, which is the mechanism for catching global
inconsistency cues a purely convolutional model would miss -- e.g. a
diffusion-inpainted region that looks locally plausible but is statistically
inconsistent with the rest of the image, or artifacts that only show up when you
look at the *relationship* between two regions (a hallmark of many AIGC pipelines).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_patches: int,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, dim) if in_channels != dim else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """feature_map: (B, C, H, W) from the CNN stem.

        Returns (B, out_dim): the encoded [CLS] token, used as the pooled image
        representation fed to the classification head.
        """
        b, c, h, w = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = self.proj(tokens)

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1 + H*W, dim)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]

        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])  # pooled [CLS] representation
