"""Layer 4: fusing regional findings into one verdict.

This stage exists to be *conservative*, and the report says why: "an uncertain
mapper prediction must not be converted into a definitive conclusion merely
because one specialist returns a high score." Three mechanisms enforce that, and
they are the substance of this module:

1. **Shrinkage toward the map.** A specialist's probability moves the region's
   score in proportion to that specialist's own confidence. At confidence 0 the
   region keeps the map's opinion; at confidence 1 the specialist replaces it.
   Nothing in between is free.

2. **A ceiling set by map support.** A region the map was never confident about
   -- weak scores, thin window coverage -- has a hard cap on how certain the
   fused result may become, no matter what a specialist says. A specialist's job
   is to characterise evidence the map found, not to manufacture evidence the
   map did not.

3. **Two hypotheses, not two votes.** "This whole image was generated" and
   "this photograph was locally edited" are competing explanations of the same
   pixels, so they are combined with `max`, not with a noisy-OR. OR-ing them
   would let a moderate global score and a moderate local score multiply into a
   confident verdict that neither piece of evidence supports.

Within the local hypothesis, separate regions *are* independent evidence -- two
different edited areas genuinely do make tampering more likely than one -- so
those combine with a noisy-OR, each region weighted by its own confidence.

The reported `confidence` is about the *system's* epistemic state, not the
score's distance from 0.5, and it is capped when the map is uncalibrated: a
threshold on an uncalibrated score is an arbitrary cut, so a pipeline running on
`Calibrator.identity()` should not be claiming high confidence. That cap is the
main reason to bother fitting a calibrator at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mapper.heatmap import AILikelihoodMap
from regions.proposals import Region
from router import RouteDecision
from specialists.base import SpecialistResult

UNCALIBRATED_CONFIDENCE_CAP = 0.60


@dataclass
class RegionFinding:
    """One region after its specialists have reported and shrinkage applied."""

    region: Region
    route: RouteDecision
    results: list[SpecialistResult]
    probability: float
    confidence: float
    label: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self, scale_factor: float = 1.0) -> dict:
        return {
            "region_id": self.region.region_id,
            "label": self.label,
            "probability": self.probability,
            "confidence": self.confidence,
            "bbox": list(self.region.bbox),
            "bbox_original": self.region.bbox_in_original(scale_factor),
            "area_frac": self.region.area_frac,
            "routed_to": self.route.primary,
            "routing_reason": self.route.reason,
            "descriptors": self.region.descriptors(),
            "evidence": list(self.evidence),
            "specialists": [r.to_dict() for r in self.results],
        }


@dataclass
class FusedVerdict:
    verdict: str
    score: float
    confidence: float
    explanation: str
    findings: list[RegionFinding]
    details: dict = field(default_factory=dict)


def _map_confidence(region: Region, thresholds: tuple[float, float]) -> float:
    """How much the map itself backed this region, in [0, 1].

    Distance above the "likely AI" threshold, discounted by thin window support
    and by disagreement between scales.
    """
    lo, hi = thresholds
    headroom = max(1e-6, 1.0 - hi)
    strength = float(np.clip((region.mean_score - hi) / headroom, 0.0, 1.0))
    support = float(np.clip(region.support / 0.6, 0.0, 1.0))
    stability = float(np.clip(1.0 - region.scale_disagreement / 0.25, 0.0, 1.0))
    return float(np.clip(0.35 * strength + 0.35 * support + 0.30 * stability, 0.0, 1.0))


def fuse_region(
    region: Region,
    route: RouteDecision,
    results: list[SpecialistResult],
    thresholds: tuple[float, float],
) -> RegionFinding:
    """Shrink the specialists' opinion toward the map, then cap it by map support."""
    if not results:
        raise ValueError(f"region {region.region_id} has no specialist results")

    # Confidence-weighted consensus across whichever specialists ran.
    weights = np.array([r.confidence for r in results], dtype=np.float64)
    probs = np.array([r.probability for r in results], dtype=np.float64)
    if weights.sum() <= 1e-6:
        specialist_p, specialist_conf = 0.5, 0.0
    else:
        specialist_p = float((probs * weights).sum() / weights.sum())
        # Two specialists agreeing is worth more than either alone; two
        # disagreeing is worth less than either.
        spread = float(probs.max() - probs.min()) if probs.size > 1 else 0.0
        specialist_conf = float(np.clip(weights.max() * (1.0 - 0.5 * spread), 0.0, 1.0))

    # 1. Shrinkage: the specialist moves the map's score by its own confidence.
    fused = specialist_conf * specialist_p + (1.0 - specialist_conf) * float(region.mean_score)

    # 2. Ceiling: a region the map was unsure about cannot become a certainty.
    map_conf = _map_confidence(region, thresholds)
    ceiling = 0.5 + 0.5 * (0.35 + 0.65 * map_conf)
    probability = float(min(fused, ceiling))

    confidence = float(np.clip(0.5 * map_conf + 0.5 * specialist_conf, 0.0, 1.0))

    primary = next((r for r in results if r.specialist == route.primary), results[0])
    kind = primary.details.get("manipulation_kind")
    if route.primary == "synthesis":
        label = "generated_content"
    elif route.primary == "face_edit":
        label = "face_region_edit_evidence"
    elif kind == "conventional":
        label = "conventional_manipulation"
    elif route.primary == "fallback":
        label = "uncharacterised_suspicion"
    else:
        label = "generative_edit"

    evidence: list[str] = []
    for result in results:
        evidence.extend(result.evidence)
    if probability < fused - 1e-6:
        evidence.append(
            f"score capped at {probability:.2f}: the likelihood map's own support for this "
            f"region ({map_conf:.2f}) does not justify the specialist's confidence"
        )

    return RegionFinding(
        region=region,
        route=route,
        results=results,
        probability=probability,
        confidence=confidence,
        label=label,
        evidence=evidence,
    )


def _noisy_or(contributions: list[float]) -> float:
    survival = 1.0
    for c in contributions:
        survival *= 1.0 - float(np.clip(c, 0.0, 0.999))
    return float(1.0 - survival)


def fuse(
    amap: AILikelihoodMap,
    findings: list[RegionFinding],
    whole_image_score: float,
    min_confidence: float = 0.35,
    decide_at: float = 0.5,
) -> FusedVerdict:
    """Combine regional findings and the whole-image score into a verdict."""
    thresholds = amap.thresholds

    # Local hypothesis: independent regions, each weighted by its confidence.
    contributions = []
    for finding in findings:
        excess = max(0.0, finding.probability - decide_at) / max(1e-6, 1.0 - decide_at)
        contributions.append(excess * (0.4 + 0.6 * finding.confidence))
    local_evidence = _noisy_or(contributions)
    local_score = decide_at + (1.0 - decide_at) * local_evidence

    # Global hypothesis vs. local hypothesis: alternatives, so max, not OR.
    score = float(max(whole_image_score, local_score if findings else 0.0))

    synthesis_findings = [f for f in findings if f.route.primary == "synthesis"]
    conventional = [f for f in findings if f.label == "conventional_manipulation"]
    coverage = float(sum(f.region.area_frac for f in findings))

    # Confidence: how decisive the map was, how well the specialists did, and
    # how much of the frame was left ambiguous.
    map_decisiveness = 1.0 - amap.fraction(1)  # 1 == LABEL_UNCERTAIN
    finding_conf = max((f.confidence for f in findings), default=0.0)
    global_decisiveness = float(min(1.0, abs(whole_image_score - decide_at) * 2.0))
    confidence = float(
        np.clip(
            0.35 * map_decisiveness + 0.40 * max(finding_conf, global_decisiveness) + 0.25 * float(amap.support.mean()),
            0.0,
            1.0,
        )
    )
    capped_by_calibration = False
    if not amap.calibrated:
        capped_by_calibration = confidence > UNCALIBRATED_CONFIDENCE_CAP
        confidence = min(confidence, UNCALIBRATED_CONFIDENCE_CAP)

    verdict = _verdict(
        score=score,
        confidence=confidence,
        findings=findings,
        synthesis_findings=synthesis_findings,
        conventional=conventional,
        coverage=coverage,
        min_confidence=min_confidence,
        decide_at=decide_at,
    )

    explanation = _explain(
        verdict=verdict,
        score=score,
        confidence=confidence,
        findings=findings,
        amap=amap,
        whole_image_score=whole_image_score,
        capped_by_calibration=capped_by_calibration,
    )

    return FusedVerdict(
        verdict=verdict,
        score=score,
        confidence=confidence,
        explanation=explanation,
        findings=findings,
        details={
            "whole_image_score": whole_image_score,
            "local_score": local_score if findings else None,
            "suspicious_coverage": coverage,
            "regions": len(findings),
            "map": amap.summary(),
            "confidence_capped_by_calibration": capped_by_calibration,
        },
    )


def _verdict(
    score: float,
    confidence: float,
    findings: list[RegionFinding],
    synthesis_findings: list[RegionFinding],
    conventional: list[RegionFinding],
    coverage: float,
    min_confidence: float,
    decide_at: float,
) -> str:
    if confidence < min_confidence:
        return "uncertain"
    if score < decide_at:
        return "likely_authentic" if not findings else "likely_authentic_with_open_questions"
    if synthesis_findings and coverage >= 0.5:
        return "ai_generated"
    if conventional and len(conventional) >= max(1, len(findings) // 2 + len(findings) % 2):
        # The manipulation evidence points at conventional editing, not a
        # generator. Saying "AI" here would be the expensive false positive.
        return "manipulated_not_necessarily_ai"
    if findings:
        return "ai_edited"
    return "ai_generated"


def _explain(
    verdict: str,
    score: float,
    confidence: float,
    findings: list[RegionFinding],
    amap: AILikelihoodMap,
    whole_image_score: float,
    capped_by_calibration: bool,
) -> str:
    """A short paragraph a person can check against the overlay."""
    headline = {
        "ai_generated": "This image reads as generated rather than photographed.",
        "ai_edited": "This looks like a photograph with AI-edited region(s).",
        "manipulated_not_necessarily_ai": (
            "This image shows manipulation, but the evidence points at conventional "
            "editing rather than a generator."
        ),
        "likely_authentic": "No AI-generation or editing evidence was found.",
        "likely_authentic_with_open_questions": (
            "Probably authentic, though one or more areas did not fully clear."
        ),
        "uncertain": "The evidence is too weak or conflicting to call.",
    }.get(verdict, "Inconclusive.")

    parts = [headline]

    if findings:
        top = max(findings, key=lambda f: f.probability * f.confidence)
        x0, y0, x1, y1 = top.region.bbox
        parts.append(
            f"The strongest finding covers {top.region.area_frac * 100:.1f}% of the frame at "
            f"({x0}, {y0})-({x1}, {y1}), routed to the {top.route.primary} specialist because "
            f"{top.route.reason}."
        )
        if top.evidence:
            parts.append("It found: " + "; ".join(top.evidence[:3]) + ".")
        if len(findings) > 1:
            parts.append(
                f"{len(findings) - 1} further region(s) were examined; see the per-region findings."
            )
    else:
        parts.append(
            f"No region cleared the map's 'likely AI' threshold "
            f"({amap.thresholds[1]:.2f}); the score comes from the whole-image detector "
            f"({whole_image_score:.2f})."
        )

    uncertain_frac = amap.fraction(1)
    if uncertain_frac > 0.25:
        parts.append(
            f"{uncertain_frac * 100:.0f}% of the image sits in the map's uncertain band, so "
            f"this verdict rests on less evidence than the score alone suggests."
        )
    if capped_by_calibration:
        parts.append(
            "Confidence is capped because the likelihood map is running uncalibrated "
            "(no fitted calibrator), which makes its thresholds arbitrary."
        )

    parts.append(f"Score {score:.2f}, confidence {confidence:.2f}.")
    return " ".join(parts)
