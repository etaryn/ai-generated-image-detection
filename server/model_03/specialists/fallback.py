"""Specialist: the general fallback for uncertain regions.

Where the other specialists have a hypothesis to test, this one has none. It is
what runs when the map found something it cannot characterise -- weak evidence,
conflicting scales, a region too small or too thin for the forensic contrasts to
mean anything, or a heavily degraded upload where compression has erased the
statistics everything else reads.

Its job is not to produce a verdict. Its job is to *keep the region visible*
while refusing to inflate it: it reports the map's own evidence plus a detector
re-score, at deliberately low confidence, so fusion carries the region into the
report as "worth a look" without letting it drive the image-level answer.

This is the behaviour the report asks for in the "uncertain" branch, and it is
the honest answer to a real situation. A pipeline that quietly dropped
uncertain regions would look more decisive and be less useful; one that routed
them to a confident specialist anyway would be worse than either.
"""
from __future__ import annotations

import numpy as np

from specialists.base import SpecialistContext, SpecialistResult, blend_probability

MAX_CONFIDENCE = 0.35


class FallbackSpecialist:
    name = "fallback"

    def analyse(self, ctx: SpecialistContext) -> SpecialistResult:
        region = ctx.region
        crop_score = ctx.score_crop()

        evidence = [
            f"region flagged by the likelihood map (mean {region.mean_score:.2f}, "
            f"peak {region.max_score:.2f}) but not characterised by any specialist",
            f"detector score on the region crop {crop_score:.3f}",
        ]

        if region.scale_disagreement > 0.15:
            evidence.append(
                f"window scales disagree about this region by {region.scale_disagreement:.2f} -- "
                f"the evidence is not stable across resolutions"
            )
        if region.support < 0.4:
            evidence.append(
                f"thin window coverage here ({region.support:.2f}), typically because the region "
                f"runs to the frame edge -- less evidence than elsewhere in the image"
            )
        if region.area_frac < 0.01:
            evidence.append(
                f"small region ({region.area_frac * 100:.1f}% of the frame); forensic contrasts "
                f"need more pixels than this to be reliable"
            )

        probability = blend_probability(
            (float(region.mean_score), 1.0),
            (crop_score, 1.0),
        )

        # Confidence rises a little with size and scale agreement but is capped
        # well below the characterising specialists by construction.
        confidence = float(
            np.clip(
                0.10
                + 0.10 * min(1.0, region.area_frac / 0.05)
                + 0.10 * float(np.clip(1.0 - region.scale_disagreement / 0.2, 0.0, 1.0)),
                0.0,
                MAX_CONFIDENCE,
            )
        )

        return SpecialistResult(
            specialist=self.name,
            probability=probability,
            confidence=confidence,
            evidence=evidence,
            details={
                "crop_score": crop_score,
                "scale_disagreement": region.scale_disagreement,
                "support": region.support,
                "note": "no hypothesis tested; reported so the region stays visible",
            },
        )
