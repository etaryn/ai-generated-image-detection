"""The Step-1 stack: build the enabled extractors, run them, concatenate.

One image in, one flat vector out:

    [ ---- DINOv2 (texture/structure) ---- | -- CLIP (semantics) -- | - FFT stats - ]

The stack also records a *block spec* -- the name, width and column range of each
extractor's contribution. That's what makes the design auditable: eval/ablation.py
retrains on individual blocks to report what each branch is actually worth, and
the checkpoint stores the spec so infer.py can verify it's feeding the classifier
the same layout it was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from features.base import FeatureExtractor


@dataclass
class FeatureBlock:
    name: str
    dim: int
    start: int
    stop: int

    def slice(self) -> slice:
        return slice(self.start, self.stop)

    def to_dict(self) -> dict:
        return {"name": self.name, "dim": self.dim, "start": self.start, "stop": self.stop}


class FeatureStack:
    """Holds the enabled extractors and concatenates their outputs."""

    def __init__(self, extractors: list[FeatureExtractor]):
        if not extractors:
            raise ValueError(
                "No feature extractors are enabled -- set at least one of "
                "features.dinov2/clip/fft `enabled: true` in the config."
            )
        self.extractors = extractors
        self.blocks: list[FeatureBlock] = []
        offset = 0
        for ex in extractors:
            self.blocks.append(FeatureBlock(ex.name, ex.dim, offset, offset + ex.dim))
            offset += ex.dim
        self.dim = offset

    @classmethod
    def from_config(cls, features_cfg: dict, device: str | torch.device = "cpu") -> "FeatureStack":
        """Build from the `features` block of a loaded YAML config.

        Extractors are imported lazily so that disabling a branch also removes its
        dependency -- an FFT-only run needs neither transformers nor open_clip.
        """
        extractors: list[FeatureExtractor] = []

        dino_cfg = features_cfg.get("dinov2", {})
        if dino_cfg.get("enabled", False):
            from features.dino import DinoV2Features

            extractors.append(
                DinoV2Features(
                    model_name=dino_cfg.get("model_name", "facebook/dinov2-base"),
                    image_size=dino_cfg.get("image_size", 224),
                    pooling=dino_cfg.get("pooling", "cls_mean"),
                    l2_normalize=dino_cfg.get("l2_normalize", True),
                    device=device,
                )
            )

        clip_cfg = features_cfg.get("clip", {})
        if clip_cfg.get("enabled", False):
            from features.clip import ClipFeatures

            extractors.append(
                ClipFeatures(
                    backbone_name=clip_cfg.get("backbone_name", "ViT-B-16"),
                    pretrained=clip_cfg.get("pretrained", "openai"),
                    image_size=clip_cfg.get("image_size", 224),
                    l2_normalize=clip_cfg.get("l2_normalize", True),
                    device=device,
                )
            )

        fft_cfg = features_cfg.get("fft", {})
        if fft_cfg.get("enabled", False):
            from features.fft import FFTStatsFeatures

            extractors.append(
                FFTStatsFeatures(
                    work_size=fft_cfg.get("work_size", 256),
                    n_radial_bins=fft_cfg.get("n_radial_bins", 32),
                    n_angular_bins=fft_cfg.get("n_angular_bins", 16),
                    blur_sigma=fft_cfg.get("blur_sigma", 1.0),
                    n_threads=fft_cfg.get("n_threads", 4),
                )
            )

        return cls(extractors)

    def __call__(self, canonical: torch.Tensor) -> np.ndarray:
        """canonical: (B, 3, S, S) float in [0, 1]. Returns (B, self.dim) float32."""
        parts = [ex(canonical) for ex in self.extractors]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for ex in self.extractors:
            names.extend(ex.feature_names())
        return names

    def block_spec(self) -> list[dict]:
        return [b.to_dict() for b in self.blocks]

    def signature(self) -> dict:
        return {
            "dim": self.dim,
            "blocks": self.block_spec(),
            "extractors": [ex.signature() for ex in self.extractors],
        }

    def resolved_backbones(self) -> dict:
        """Which concrete backbone each extractor actually ran, by block name.

        Kept as its own top-level cache field rather than read back out of
        signature()["extractors"], because train.py stores the *pruned* block spec
        (--blocks fft drops the clip entry) and infer.py still has to be able to
        ask what the clip features were extracted with.
        """
        return {
            ex.name: ex.resolved_backbone_name
            for ex in self.extractors
            if getattr(ex, "resolved_backbone_name", None)
        }
