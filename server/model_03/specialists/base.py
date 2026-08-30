"""What a specialist is, and what it is handed.

A specialist looks at *one* region and answers four things:

    probability   P(this region was AI-generated or AI-edited), in [0, 1]
    confidence    how much this answer should be allowed to matter, in [0, 1]
    evidence      short human-readable strings naming what it actually saw
    refined_mask  optional tighter mask than the one the mapper proposed

`confidence` is the load-bearing field. It is what lets fusion be conservative
without being deaf: a specialist that found strong, mutually-corroborating
evidence says so and moves the verdict; one that ran on a JPEG-free image where
half its measurements are meaningless returns the same probability with low
confidence and barely moves it. A specialist that returns high confidence on
thin evidence is a bug, not an optimisation.

`evidence` is not decoration either -- it is what the explanation in the final
report is built from, so each string should name a measurement and its
direction ("residual noise 41% below the surrounding ring"), never a conclusion
("this is inpainted").

Specialists share `SpecialistContext`, which caches the two expensive things --
the forensic measurement set and any re-scoring of the region through the
detector -- so routing a region to two specialists costs barely more than one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image

from forensics import region_report
from mapper.heatmap import AILikelihoodMap
from regions.components import ring
from regions.proposals import Region


@dataclass
class SpecialistResult:
    """One specialist's finding for one region."""

    specialist: str
    probability: float
    confidence: float
    evidence: list[str] = field(default_factory=list)
    refined_mask: np.ndarray | None = None
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.probability = float(np.clip(self.probability, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "specialist": self.specialist,
            "probability": self.probability,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "details": self.details,
        }


class SpecialistContext:
    """Everything a specialist may look at, with the expensive parts cached."""

    def __init__(
        self,
        image: Image.Image,
        amap: AILikelihoodMap,
        region: Region,
        scorer,
        ring_radius: int = 8,
        faces: list | None = None,
    ):
        self.image = image
        self.amap = amap
        self.region = region
        self.scorer = scorer
        self.ring_radius = int(ring_radius)
        self.faces = faces or []
        self._ring: np.ndarray | None = None
        self._forensics: dict | None = None
        self._crop_scores: dict[int, float] = {}

    @property
    def ring(self) -> np.ndarray:
        """The band just outside the region -- the local reference for every contrast."""
        if self._ring is None:
            self._ring = ring(self.region.mask, self.ring_radius)
        return self._ring

    def forensics(self) -> dict:
        if self._forensics is None:
            self._forensics = region_report(self.image, self.region.mask, self.ring)
        return self._forensics

    def score_crop(self, pad: int = 8) -> float:
        """Re-score the region's bounding box through the detector as one image.

        Distinct evidence from the map: the map is an average of overlapping
        patch verdicts, this is a single verdict on the region as a whole, which
        is closer to the question the detector was actually trained on.
        """
        if pad not in self._crop_scores:
            box = self.region.crop_box(pad=pad, bounds=self.image.size)
            crop = self.image.crop(box)
            if min(crop.size) < 8:
                self._crop_scores[pad] = float(self.region.mean_score)
            else:
                self._crop_scores[pad] = float(self.scorer.score_patches([crop])[0])
        return self._crop_scores[pad]


class Specialist(Protocol):
    name: str

    def analyse(self, ctx: SpecialistContext) -> SpecialistResult:
        ...


def blend_probability(*terms: tuple[float, float]) -> float:
    """Weighted mean of (value, weight) pairs, with a 0.5 prior when unweighted.

    Used by every specialist so they combine their inputs the same way, and so a
    specialist whose measurements were all invalid lands on 0.5 ("no opinion")
    rather than on 0.0 ("definitely clean"), which fusion would read as evidence
    of authenticity that nobody actually produced.
    """
    weighted = [(float(v), float(w)) for v, w in terms if w > 0]
    if not weighted:
        return 0.5
    total = sum(w for _, w in weighted)
    return float(np.clip(sum(v * w for v, w in weighted) / total, 0.0, 1.0))
