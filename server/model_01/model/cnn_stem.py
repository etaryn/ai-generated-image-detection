"""Convolutional stem for the CNN + Transformer hybrid detector.

Produces a grid of local feature vectors ("patch tokens") from the raw image,
analogous to the hybrid-ViT design (Dosovitskiy et al., ICLR 2021) where a
ResNet-style CNN replaces the raw linear patch embedding. The motivation here is
specific to AIGC detection: convolutional layers with a limited receptive field
are well suited to picking up the *local*, low-level artifacts synthetic images
tend to leave (upsampling checkerboard patterns, GAN/diffusion decoder texture,
blending seams), while the transformer stage that follows (transformer_encoder.py)
reasons globally over those local signals to catch spatially inconsistent regions
-- e.g. one patch whose artifact statistics don't match its neighbors.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvStem(nn.Module):
    """4-stage CNN that downsamples a 224x224 RGB image to a 14x14 grid of
    `channels[-1]`-dim feature vectors (stride 16 total: 2*2*2*2 across 4 stages).

    Fully trainable, no pretraining -- parameter count for the default channel
    schedule (64, 128, 256, 384) is roughly 4-6M, small enough to train end-to-end
    on hackathon-scale compute and far under the challenge's 2B-parameter limit.
    """

    def __init__(self, in_channels: int = 3, channels: list[int] | tuple[int, ...] = (64, 128, 256, 384)):
        super().__init__()
        self.channels = list(channels)
        stages = []
        prev = in_channels
        for ch in channels:
            stages.append(conv_block(prev, ch, stride=2))
            prev = ch
        self.stages = nn.ModuleList(stages)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W). Returns (B, out_dim, H / 2**num_stages, W / 2**num_stages)."""
        for stage in self.stages:
            x = stage(x)
        return x
