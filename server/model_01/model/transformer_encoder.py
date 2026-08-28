"""Transformer encoder stage that sits on top of the CNN stem's feature grid.

Turns the CNN stem's (B, C, H', W') feature map into a sequence of tokens, adds a
learnable [CLS] token and learnable positional embeddings, and runs a standard
multi-head self-attention encoder over it. Self-attention lets every local patch
"compare notes" with every other patch, which is the mechanism for catching global
inconsistency cues a purely convolutional model would miss -- e.g. a
diffusion-inpainted region that looks locally plausible but is statistically
inconsistent with the rest of the image, or artifacts that only show up when you
look at the *relationship* between two regions.

Implemented as an explicit stack of pre-norm encoder blocks (functionally
equivalent to `nn.TransformerEncoderLayer(norm_first=True, activation="gelu")`)
rather than using `nn.TransformerEncoder` directly, so that self-attention
weights can be captured on demand -- torch's built-in encoder layer doesn't
reliably expose attention weights once its fused/flash-attention fast path
kicks in. Capturing them is what powers the attention-rollout explainability
pass in `eval/attention_rollout.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """One pre-norm self-attention + MLP block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.last_attn_weights: torch.Tensor | None = None  # set when return_attention=True

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed,
            need_weights=return_attention,
            average_attn_weights=True,  # average over heads -> (B, N, N)
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            self.last_attn_weights = attn_weights.detach()
        return x


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

        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim
        self.depth = depth

    def forward(self, feature_map: torch.Tensor, return_attention: bool = False):
        """feature_map: (B, C, H, W) from the CNN stem.

        Returns the encoded [CLS] token (B, out_dim) by default -- the pooled
        image representation fed to the classification head. When
        `return_attention=True`, also returns a list of per-layer (B, N, N)
        head-averaged attention matrices (N = 1 + H*W, including [CLS]), for
        `eval/attention_rollout.py`.
        """
        b, c, h, w = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = self.proj(tokens)

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1 + H*W, dim)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]

        attn_maps = [] if return_attention else None
        for block in self.blocks:
            tokens = block(tokens, return_attention=return_attention)
            if return_attention:
                attn_maps.append(block.last_attn_weights)

        pooled = self.norm(tokens[:, 0])  # pooled [CLS] representation
        if return_attention:
            return pooled, attn_maps
        return pooled
