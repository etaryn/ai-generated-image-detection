"""Layer 3: choosing which specialist looks at which region.

The router is a short list of readable rules, not a model, and that is a
deliberate choice for the prototype. Routing is where a wrong answer is most
expensive -- send an inpainted area to the synthesis specialist and the evidence
is measured against the wrong hypothesis -- and a rule you can read is a rule you
can check against a heatmap you are looking at. A learned router is the obvious
next step once there is routing-labelled data to learn from; there isn't yet, and
a router fit on guesses would be worse than the rules while looking better.

It decides on **cheap descriptors only**: region geometry, the map's own shape,
and face boxes. Nothing here computes a forensic statistic -- that is the
specialist's job, and paying for it before choosing a specialist would defeat
the point of routing at all.

The rules, in priority order:

1. **Overlaps a detected face** -> `face_edit`. Faces are the highest-stakes
   case; when a face detector is installed and a region sits on a face, it goes
   to the face route even if another rule would also fire.
2. **Covers most of the frame with the scales agreeing** -> `synthesis`. Not a
   tampered region: an image that was generated.
3. **Weak, unstable, thinly-covered, or tiny** -> `fallback`. Anything the other
   specialists could not honestly measure.
4. **Fine-scale-dominant with a sharp rim** -> `inpainting`. A localised edit
   that the fine windows see and the coarse ones dilute.
5. **Box-shaped rim, or a sharp rim without fine-scale dominance** ->
   `compositing`. Machine-selected boundaries and content with a foreign
   history, which may not be AI at all.
6. Anything else -> `inpainting`, the most common local case, with
   `compositing` as its alternate.

Every decision carries `alternates` and a `reason` string. The reason goes into
the report next to the finding, so the choice is auditable after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from regions.proposals import Region
from specialists.faces import FaceBox, overlap_fraction


@dataclass
class RouteDecision:
    region_id: int
    primary: str
    alternates: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "primary": self.primary,
            "alternates": list(self.alternates),
            "reason": self.reason,
        }


class Router:
    def __init__(
        self,
        whole_image_frac: float = 0.55,
        scale_agreement_max: float = 0.12,
        face_overlap_min: float = 0.35,
        sharp_boundary: float = 0.030,
        fine_scale_margin: float = 0.06,
        rectangular_fill: float = 0.88,
        weak_score: float = 0.60,
        min_support: float = 0.35,
        min_area_frac: float = 0.008,
        unstable_scales: float = 0.22,
    ):
        self.whole_image_frac = whole_image_frac
        self.scale_agreement_max = scale_agreement_max
        self.face_overlap_min = face_overlap_min
        self.sharp_boundary = sharp_boundary
        self.fine_scale_margin = fine_scale_margin
        self.rectangular_fill = rectangular_fill
        self.weak_score = weak_score
        self.min_support = min_support
        self.min_area_frac = min_area_frac
        self.unstable_scales = unstable_scales

    @staticmethod
    def _fine_minus_coarse(region: Region) -> float:
        """Fine-scale mean minus coarse-scale mean over the region.

        Positive means the small windows are more suspicious than the large
        ones -- the signature of evidence confined to a small area, since a
        coarse window averages the edit together with the authentic content
        around it.
        """
        profile = region.scale_profile
        if len(profile) < 2:
            return 0.0
        scales = sorted(profile)
        fine, coarse = profile[scales[0]], profile[scales[-1]]
        if np.isnan(fine) or np.isnan(coarse):
            return 0.0
        return float(fine - coarse)

    def _matching_face(self, region: Region, faces: list[FaceBox]) -> FaceBox | None:
        best, best_overlap = None, 0.0
        for face in faces:
            overlap = overlap_fraction(region.mask, face)
            if overlap > best_overlap:
                best, best_overlap = face, overlap
        return best if best_overlap >= self.face_overlap_min else None

    def route(self, region: Region, faces: list[FaceBox] | None = None) -> RouteDecision:
        faces = faces or []
        fine_lead = self._fine_minus_coarse(region)
        sharp = region.boundary_sharpness >= self.sharp_boundary

        face = self._matching_face(region, faces)
        if face is not None:
            # Stash it so the face specialist can widen the mask to the whole face.
            region.meta["face"] = face
            return RouteDecision(
                region.region_id,
                "face_edit",
                ["inpainting"],
                f"region overlaps a detected face box by "
                f"{overlap_fraction(region.mask, face) * 100:.0f}%",
            )

        if (
            region.area_frac >= self.whole_image_frac
            and region.scale_disagreement <= self.scale_agreement_max
        ):
            return RouteDecision(
                region.region_id,
                "synthesis",
                [],
                f"covers {region.area_frac * 100:.0f}% of the frame with the window scales "
                f"agreeing (spread {region.scale_disagreement:.2f}) -- reads as a generated "
                f"image, not a local edit",
            )

        weak_reasons = []
        if region.mean_score < self.weak_score:
            weak_reasons.append(f"mean score {region.mean_score:.2f} below {self.weak_score:.2f}")
        if region.support < self.min_support:
            weak_reasons.append(f"window support {region.support:.2f} is thin")
        if region.area_frac < self.min_area_frac:
            weak_reasons.append(f"only {region.area_frac * 100:.1f}% of the frame")
        # Scale disagreement only counts as instability when it is *unexplained*.
        # A local edit makes the fine scale lead the coarse one by construction
        # (a coarse window averages the edit together with authentic pixels), so
        # subtracting the fine-scale lead is the difference between "the scales
        # are noisy" and "the scales are telling us this evidence is local".
        unexplained = region.scale_disagreement - max(0.0, fine_lead)
        if unexplained > self.unstable_scales:
            weak_reasons.append(
                f"scales disagree by {region.scale_disagreement:.2f}, of which "
                f"{unexplained:.2f} is not explained by the evidence being localised"
            )
        if weak_reasons:
            return RouteDecision(
                region.region_id,
                "fallback",
                [],
                "evidence too weak or unstable to characterise: " + "; ".join(weak_reasons),
            )

        if fine_lead >= self.fine_scale_margin and sharp:
            return RouteDecision(
                region.region_id,
                "inpainting",
                ["compositing"],
                f"localised evidence -- fine windows lead the coarse ones by {fine_lead:.2f} "
                f"and the map falls off sharply at the rim "
                f"({region.boundary_sharpness:.3f}/px)",
            )

        if region.fill_ratio >= self.rectangular_fill:
            return RouteDecision(
                region.region_id,
                "compositing",
                ["inpainting"],
                f"the region fills {region.fill_ratio * 100:.0f}% of its bounding box -- a "
                f"box-shaped selection rather than an object outline",
            )

        if sharp:
            return RouteDecision(
                region.region_id,
                "compositing",
                ["inpainting"],
                f"sharp boundary ({region.boundary_sharpness:.3f}/px) without fine-scale "
                f"dominance (lead {fine_lead:.2f}) -- inserted content rather than "
                f"regenerated content",
            )

        return RouteDecision(
            region.region_id,
            "inpainting",
            ["compositing"],
            f"no distinguishing geometry (fine-scale lead {fine_lead:.2f}, boundary "
            f"{region.boundary_sharpness:.3f}/px); defaulting to the most common local case",
        )
