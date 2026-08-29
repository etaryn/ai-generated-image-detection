"""Step 1b: CLIP features -- "numbers describing semantic meaning / logic".

CLIP was trained to align images with the text that describes them, so its
embedding encodes what the picture *means*: objects, scene, style, and how
coherently those go together. That's a complementary axis to DINOv2's texture
evidence -- it's the branch that can react to a semantically implausible scene
(text that isn't language, an implausible object combination, the characteristic
"prompt-shaped" composition of generated imagery) rather than to pixel statistics.

There's also a practical reason to keep CLIP in the stack: frozen CLIP features
are the strongest known baseline for *cross-generator* generalization (Ojha et al.,
CVPR 2023) -- they degrade less on generator families that weren't in the training
set, which is the failure mode this challenge's held-out demo set probes.

Uses `open_clip` when available (same implementation model_01 uses, so both models
see identical features), falling back to `transformers`' CLIP. Frozen; ViT-B-16 is
~86M params.
"""
from __future__ import annotations

import numpy as np
import torch

from features.base import FeatureExtractor, resize_and_normalize

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# open_clip name -> HF name, for the fallback path.
_HF_EQUIVALENT = {
    "ViT-B-16": "openai/clip-vit-base-patch16",
    "ViT-B-32": "openai/clip-vit-base-patch32",
    "ViT-L-14": "openai/clip-vit-large-patch14",
}


class ClipFeatures(FeatureExtractor):
    name = "clip"

    def __init__(
        self,
        backbone_name: str = "ViT-B-16",
        pretrained: str = "openai",
        image_size: int = 224,
        l2_normalize: bool = True,
        device: str | torch.device = "cpu",
    ):
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.image_size = image_size
        self.l2_normalize = l2_normalize
        self.device = torch.device(device)

        try:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(backbone_name, pretrained=pretrained)
            # Keep only the vision tower (as model_01/model/backbone.py does): the
            # text encoder is ~63M parameters this pipeline never calls, and there
            # is no reason to carry it into GPU memory during extraction.
            self.model = model.visual.eval().to(self.device)
            self.impl = "open_clip"
            self.dim = self.model.output_dim
        except ImportError:
            from transformers import CLIPVisionModelWithProjection

            hf_name = _HF_EQUIVALENT.get(backbone_name)
            if hf_name is None:
                raise ImportError(
                    f"open_clip isn't installed and there's no HF fallback mapped for "
                    f"backbone_name={backbone_name!r}. Install open_clip_torch, or use one of "
                    f"{sorted(_HF_EQUIVALENT)}."
                )
            self.model = CLIPVisionModelWithProjection.from_pretrained(hf_name).eval().to(self.device)
            self.impl = "transformers"
            self.dim = self.model.config.projection_dim

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, canonical: torch.Tensor) -> np.ndarray:
        x = resize_and_normalize(canonical.to(self.device), self.image_size, CLIP_MEAN, CLIP_STD)
        if self.impl == "open_clip":
            feats = self.model(x)
        else:
            feats = self.model(pixel_values=x).image_embeds

        if self.l2_normalize:
            # CLIP's own similarity space is the unit sphere, so unit-norming is
            # the natural scale for these features (and matches how CLIP is used
            # downstream everywhere else).
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.float().cpu().numpy()

    def signature(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "backbone_name": self.backbone_name,
            "pretrained": self.pretrained,
            "image_size": self.image_size,
            "l2_normalize": self.l2_normalize,
            "impl": self.impl,
        }
