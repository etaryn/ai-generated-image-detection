"""Specialist: conventional compositing (splices, pastes, non-AI edits).

Not every tampered region is AI. Someone pasting a person from one photograph
into another, cloning out a logo, or dropping in a screenshot produces a region
that the AI-likelihood map may well flag -- the map keys on "these pixels have a
different history from their surroundings", and a splice satisfies that without
a generator being involved anywhere.

This specialist exists so the system can say so. It reports the probability that
the region is *manipulated*, and separately reports how much of that evidence
points at a generator versus at a conventional edit -- fusion uses the split to
avoid labelling a plain Photoshop splice "AI-generated", which would be a false
positive of exactly the kind the challenge brief cares most about.

Its evidence is classical image forensics:

* **Foreign noise floor.** A splice from another camera (or another exposure)
  carries a *different*, often stronger, noise level -- the opposite sign to
  inpainting's over-smoothing, and the main thing separating the two.
* **JPEG ghost / grid mismatch.** Spliced content has a different compression
  history: its 8x8 lattice is misaligned, weaker, or absent.
* **ELA discontinuity.** The spliced area has been through fewer compression
  generations, so it gives up more error on a fresh recompression.
* **Straight rim.** A rectangular or machine-selected boundary. Almost never
  true of a real object, common for a paste.
* **Channel-correlation mismatch.** Demosaicing ties a camera's channel noise
  together; content from a different pipeline does not match.

ELA and JPEG-grid evidence are worthless on a PNG or a re-saved image, so both
are weighted by how much compression structure the image actually has, and the
specialist reports low confidence rather than a confident 0.5 when the image
gives it nothing to work with.
"""
from __future__ import annotations

import numpy as np

from specialists.base import SpecialistContext, SpecialistResult, blend_probability


def _squash(value: float, scale: float) -> float:
    return float(0.5 * (1.0 + np.tanh(value / max(scale, 1e-6))))


class CompositingSpecialist:
    name = "compositing"

    def analyse(self, ctx: SpecialistContext) -> SpecialistResult:
        f = ctx.forensics()
        region = ctx.region

        if not f.get("valid"):
            return SpecialistResult(
                specialist=self.name,
                probability=float(region.mean_score),
                confidence=0.15,
                evidence=[f"forensic measurement unavailable: {f.get('reason', 'unknown')}"],
                details=dict(f),
            )

        evidence: list[str] = []

        # How much compression structure exists to reason about at all.
        grid_strength = max(f["jpeg_grid_inside"], f["jpeg_grid_outside"]) - 1.0
        compression_weight = float(np.clip(grid_strength / 0.15, 0.0, 1.0))

        noise_c = f["noise_contrast"]
        grid_c = f["jpeg_grid_contrast"]
        ela_c = f["ela_contrast"]
        corr_c = f["channel_corr_contrast"]
        straightness = f["edge_straightness"]

        # Any noise mismatch is manipulation evidence here -- unlike inpainting,
        # the sign is not the point, the discontinuity is.
        terms: list[tuple[float, float]] = [
            (_squash(abs(noise_c), 0.25), 1.0),
            (_squash(abs(grid_c), 0.20), compression_weight * 1.2),
            (_squash(ela_c, 0.25), compression_weight * 1.0),
            (_squash(abs(corr_c), 0.30), 0.6),
            (float(np.clip(0.5 + straightness * 0.5, 0.0, 1.0)), 0.8),
            (float(region.mean_score), 0.8),
        ]

        if abs(noise_c) > 0.10:
            direction = "higher" if noise_c > 0 else "lower"
            evidence.append(
                f"noise floor {abs(noise_c) * 100:.0f}% {direction} inside the region -- "
                f"content with a different capture history"
            )
        if compression_weight > 0.2 and abs(grid_c) > 0.08:
            evidence.append(
                f"JPEG lattice mismatch across the boundary (inside {f['jpeg_grid_inside']:.2f}x, "
                f"outside {f['jpeg_grid_outside']:.2f}x) -- different compression generations"
            )
        if compression_weight > 0.2 and ela_c > 0.10:
            evidence.append(
                f"error-level analysis {ela_c * 100:.0f}% higher inside the region on "
                f"recompression, as less-compressed inserted content behaves"
            )
        if straightness > 0.35:
            evidence.append(
                f"{straightness * 100:.0f}% of the region rim runs in straight axis-aligned "
                f"lines -- a machine selection boundary, not an object outline"
            )
        if abs(corr_c) > 0.15:
            evidence.append(
                f"cross-channel noise correlation differs by {abs(corr_c) * 100:.0f}% from the "
                f"surround -- a different demosaicing or decoding pipeline"
            )

        probability = blend_probability(*terms)

        # Which way does the evidence lean -- generator, or conventional edit?
        # Over-smoothing points at a decoder; foreign/stronger noise and
        # compression-history mismatch point at a splice.
        ai_lean = max(0.0, -noise_c) + max(0.0, -f["high_freq_contrast"])
        conventional_lean = max(0.0, noise_c) + compression_weight * (abs(grid_c) + max(0.0, ela_c))
        total_lean = ai_lean + conventional_lean
        ai_share = float(ai_lean / total_lean) if total_lean > 1e-6 else 0.5

        if total_lean > 0.15:
            if ai_share < 0.35:
                evidence.append(
                    "the discontinuity looks like conventional compositing rather than "
                    "generated content -- flagged as manipulated, not as AI"
                )
            elif ai_share > 0.65:
                evidence.append(
                    "the discontinuity carries decoder-like smoothing, so a generative "
                    "edit is the better explanation than a plain splice"
                )

        confidence = float(
            np.clip(
                0.20 + 0.30 * compression_weight + 0.25 * min(1.0, abs(noise_c) / 0.25) + 0.15 * straightness,
                0.0,
                0.85,
            )
        )

        if not evidence:
            evidence.append("no compositing signature found at the region boundary")
            confidence = min(confidence, 0.25)

        return SpecialistResult(
            specialist=self.name,
            probability=probability,
            confidence=confidence,
            evidence=evidence,
            details={
                **f,
                "compression_weight": compression_weight,
                "ai_share": ai_share,
                "manipulation_kind": (
                    "generative" if ai_share > 0.65 else "conventional" if ai_share < 0.35 else "ambiguous"
                ),
            },
        )
