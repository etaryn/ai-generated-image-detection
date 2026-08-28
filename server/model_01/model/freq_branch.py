"""Optional frequency-domain feature branch.

AI-generated images (particularly GAN/diffusion outputs whose decoders use
transposed convolutions or repeated upsampling) tend to leave characteristic
spectral artifacts — periodic peaks in the frequency domain that aren't present in
natural photos. This branch extracts a DCT-magnitude representation and runs it
through a small CNN, producing a feature vector that can be fused with the CLIP
embedding in `detector.py`.

This is an ablation/stretch component (see configs/default.yaml's
`model.use_freq_branch`): it adds a complementary, differently-behaved signal, but
isn't required for the baseline CLIP-features-only pipeline to work.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dctn


def rgb_to_dct_magnitude(img_tensor: torch.Tensor) -> torch.Tensor:
    """Compute a log-magnitude 2D DCT per channel.

    img_tensor: (B, 3, H, W) float tensor in [0, 1].
    Returns: (B, 3, H, W) tensor of log-magnitude DCT coefficients.
    """
    batch = img_tensor.detach().cpu().numpy()
    out = np.empty_like(batch)
    for b in range(batch.shape[0]):
        for c in range(batch.shape[1]):
            coeffs = dctn(batch[b, c], type=2, norm="ortho")
            out[b, c] = np.log1p(np.abs(coeffs))
    return torch.from_numpy(out).to(img_tensor.device, dtype=img_tensor.dtype)


class FrequencyBranch(nn.Module):
    """Small CNN over the DCT-magnitude representation of the input image."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """img_tensor: (B, 3, H, W) float in [0, 1]. Returns (B, out_dim)."""
        dct_mag = rgb_to_dct_magnitude(img_tensor)
        feats = self.net(dct_mag).flatten(1)
        return self.proj(feats)
