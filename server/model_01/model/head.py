"""Small trainable classification head.

This is the only part of the model (besides the optional frequency branch) that
gets gradient updates — the CLIP backbone stays frozen. Keeping the trainable
footprint small is what makes training feasible on hackathon-scale compute and
keeps the total parameter count far under the challenge's 2B limit.
"""
from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # single logit: P(fake)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) logits
