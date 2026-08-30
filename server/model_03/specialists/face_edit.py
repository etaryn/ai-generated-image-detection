"""Specialist: face edits.

**Read this before reading its output.** This is not a deepfake detector. A real
one is trained on face swaps and face reenactment, and this project has no such
model; substituting one would mean shipping a component whose name promises far
more than it does, on the finding users are most likely to act on. So what this
specialist actually does is narrower and stated plainly in its own evidence
strings:

  it checks whether a *face-shaped* region shows the same blending and
  noise-floor discontinuities that any local edit shows, and it re-scores the
  face crop through the general detector.

That is genuinely useful -- a swapped face is a paste, and pastes leave seams --
but it will miss a well-blended swap that a face-specific model would catch, and
it cannot distinguish a swapped face from a face that was merely retouched or
beautified. Its probability is therefore capped: it is not permitted to be the
sole basis for a confident verdict (see `MAX_CONFIDENCE`), and the report labels
its findings "face-region edit evidence", never "deepfake".

The face-specific parts that *are* real:

* the region is scored against the face box rather than its own extent, since a
  swap covers the whole face and the map may only light up part of it;
* the comparison ring is the face's immediate surround (forehead, neck, hair),
  which is where a swap's blending boundary is;
* asymmetry between the eye halves of the box is reported when present, since
  generated faces still commonly fail left/right consistency.
"""
from __future__ import annotations

import numpy as np

from forensics import noise_level, region_report, to_luma
from regions.components import ring
from specialists.base import SpecialistContext, SpecialistResult, blend_probability

# Hard ceiling on what a non-face-specific model is allowed to claim about a face.
MAX_CONFIDENCE = 0.55


class FaceEditSpecialist:
    name = "face_edit"

    def analyse(self, ctx: SpecialistContext) -> SpecialistResult:
        region = ctx.region
        face = (region.meta or {}).get("face")

        evidence: list[str] = [
            "checked for edit evidence in a face region; this is a general forensic "
            "check on a face-shaped area, not a face-swap model"
        ]

        # Prefer the face box over the region's own extent: a swap covers the
        # whole face even when the map only lit part of it.
        if face is not None:
            mask = face.mask(region.mask.shape) | region.mask
        else:
            mask = region.mask
        surround = ring(mask, 10)

        f = region_report(ctx.image, mask, surround)
        crop_score = ctx.score_crop(pad=12)

        terms: list[tuple[float, float]] = [
            (float(region.mean_score), 1.0),
            (crop_score, 1.2),
        ]

        if f.get("valid"):
            noise_c = f["noise_contrast"]
            hf_c = f["high_freq_contrast"]
            terms.append((float(0.5 * (1.0 + np.tanh(-noise_c / 0.25))), 1.0))
            terms.append((float(0.5 * (1.0 + np.tanh(-hf_c / 0.30))), 0.8))
            if abs(noise_c) > 0.10:
                direction = "lower" if noise_c < 0 else "higher"
                evidence.append(
                    f"noise floor {abs(noise_c) * 100:.0f}% {direction} across the face than in "
                    f"the surrounding hair/neck band -- consistent with a blended paste"
                )
            if hf_c < -0.10:
                evidence.append(
                    f"face carries {abs(hf_c) * 100:.0f}% less fine detail than its surround -- "
                    f"the smoothing both a swap and a beautify filter produce"
                )
        else:
            evidence.append(f"forensic comparison unavailable: {f.get('reason', 'unknown')}")

        asymmetry = self._eye_line_asymmetry(ctx, mask)
        if asymmetry is not None:
            terms.append((float(np.clip(0.5 + asymmetry, 0.0, 1.0)), 0.5))
            if asymmetry > 0.12:
                evidence.append(
                    f"left/right halves of the face differ in noise floor by "
                    f"{asymmetry * 100:.0f}% -- generated faces commonly fail this symmetry"
                )

        evidence.append(f"detector score on the face crop {crop_score:.3f}")

        probability = blend_probability(*terms)
        confidence = float(
            np.clip(0.20 + (0.25 if f.get("valid") else 0.0) + 0.15 * min(1.0, region.area_frac / 0.03), 0.0, MAX_CONFIDENCE)
        )

        return SpecialistResult(
            specialist=self.name,
            probability=probability,
            confidence=confidence,
            evidence=evidence,
            refined_mask=mask if face is not None else None,
            details={
                **f,
                "crop_score": crop_score,
                "eye_line_asymmetry": asymmetry,
                "face": face.to_dict() if face is not None else None,
                "caveat": "general forensic check on a face region, not a face-swap model",
            },
        )

    @staticmethod
    def _eye_line_asymmetry(ctx: SpecialistContext, mask: np.ndarray) -> float | None:
        """|left - right| noise-floor difference across the face's vertical midline."""
        ys, xs = np.nonzero(mask)
        if xs.size < 256:
            return None
        gray = to_luma(ctx.image)
        mid = int(xs.mean())
        left = mask.copy()
        left[:, mid:] = False
        right = mask.copy()
        right[:, :mid] = False
        if left.sum() < 128 or right.sum() < 128:
            return None
        nl, nr = noise_level(gray, left), noise_level(gray, right)
        if nl + nr < 1e-6:
            return None
        return float(abs(nl - nr) / (nl + nr))
