"""Specialist: whole-image synthesis / generator artefacts.

The case this is for: the image is not a photograph that was edited, it is a
picture a generator made. This is the one case where the region-first framing
inverts -- there is no "surrounding authentic content" to compare against,
because there is no authentic content. Every local-contrast measurement in
forensics.py is therefore useless here by construction: inside and outside look
the same because they came from the same decoder.

So this specialist deliberately ignores the local contrasts and reads global
evidence instead:

* **Coverage.** How much of the frame the suspicious region occupies. A region
  covering most of the image is not a tampered area, it is the image.
* **Scale agreement.** A generator's fingerprint is present at every scale, so
  the 128px and 224px maps agree. A local paste lights the fine scale and gets
  diluted at the coarse one. Agreement across scales is the signature that
  separates "all of this was generated" from "something in here was".
* **Map uniformity.** Synthesis produces a flat, high map. A tampered photo
  produces a peaked one. Measured as the spread of scores inside the region.
* **A whole-image detector pass.** The unmodified question both sibling models
  were actually trained to answer, asked on the whole frame rather than on
  patches -- the strongest single piece of evidence available for this case, and
  the one that needs no patch-level extrapolation to interpret.

Because that last input is in-distribution for the detector, this is the only
specialist allowed to reach high confidence on its own.
"""
from __future__ import annotations

import numpy as np

from specialists.base import SpecialistContext, SpecialistResult, blend_probability


class SynthesisSpecialist:
    name = "synthesis"

    def __init__(self, coverage_full: float = 0.55, agreement_scale: float = 0.15):
        self.coverage_full = coverage_full
        self.agreement_scale = agreement_scale

    def analyse(self, ctx: SpecialistContext) -> SpecialistResult:
        region = ctx.region
        amap = ctx.amap

        # The whole frame through the detector, as one image. This is the
        # sibling models' native question.
        whole_image_score = float(ctx.scorer.score_patches([ctx.image])[0])

        coverage = float(np.clip(region.area_frac / self.coverage_full, 0.0, 1.0))
        agreement = float(np.clip(1.0 - region.scale_disagreement / self.agreement_scale, 0.0, 1.0))

        vals = amap.heat[region.mask]
        vals = vals[~np.isnan(vals)]
        uniformity = float(np.clip(1.0 - (vals.std() / 0.20), 0.0, 1.0)) if vals.size else 0.0

        evidence: list[str] = []
        if region.area_frac > 0.4:
            evidence.append(
                f"the suspicious area covers {region.area_frac * 100:.0f}% of the frame -- "
                f"this reads as a generated image rather than an edited one"
            )
        if agreement > 0.6:
            profile = ", ".join(
                f"{scale}px {value:.2f}" for scale, value in sorted(region.scale_profile.items())
            )
            evidence.append(
                f"the evidence holds at every window scale ({profile}), which is how a "
                f"generator fingerprint behaves; a local paste fades at the coarse scale"
            )
        if uniformity > 0.6:
            evidence.append(
                f"the likelihood is flat across the region (sd {vals.std():.3f}) rather than "
                f"peaked, as whole-image synthesis is"
            )
        evidence.append(f"whole-image detector score {whole_image_score:.3f}")

        probability = blend_probability(
            (whole_image_score, 2.0),
            (float(region.mean_score), 1.0),
            (whole_image_score, coverage * 1.0),   # coverage amplifies the global read
            (0.5 + 0.5 * agreement * (float(region.mean_score) - 0.5) * 2.0, 0.6),
        )

        confidence = float(
            np.clip(0.30 + 0.30 * coverage + 0.25 * agreement + 0.15 * uniformity, 0.0, 0.95)
        )

        return SpecialistResult(
            specialist=self.name,
            probability=probability,
            confidence=confidence,
            evidence=evidence,
            details={
                "whole_image_score": whole_image_score,
                "coverage": coverage,
                "scale_agreement": agreement,
                "uniformity": uniformity,
                "scale_profile": {str(k): v for k, v in sorted(region.scale_profile.items())},
            },
        )
