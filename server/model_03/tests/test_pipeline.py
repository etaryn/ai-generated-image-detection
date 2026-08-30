"""End-to-end: map -> regions -> routing -> specialists -> fusion.

Runs the whole pipeline on synthetic images with a stub patch scorer, so it
needs no checkpoint and no torch -- but the stub is not a lookup table. It
measures the patch's actual high-frequency content and scores smooth patches as
suspicious, which is a real (if crude) version of what a detector does to an
over-smoothed generative fill. Nothing in the test tells the pipeline where the
tampered square is; it has to find it.

Three cases, matching the three the design has to distinguish:

* a photograph with one smoothed square    -> a localised region, `ai_edited`
* a wholly smooth image                    -> the synthesis route, `ai_generated`
* an untouched noisy photograph            -> no regions, `likely_authentic`

Plus direct unit tests of the two fusion rules that keep the system honest:
shrinkage toward the map, and the map-support ceiling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import RegionAwareAnalyzer  # noqa: E402
from mapper.backends import CallableBackend  # noqa: E402
from mapper.heatmap import LABEL_AI  # noqa: E402
from regions.proposals import Region  # noqa: E402
from router import RouteDecision  # noqa: E402
from fusion import fuse_region  # noqa: E402
from specialists.base import SpecialistResult  # noqa: E402

SIZE = 512
# The "inpainted" area, 192px square == 14% of the frame. It is deliberately
# larger than the mapper's finest window (128px): a patch that straddles the
# edit averages tampered and authentic content together, so an edit smaller
# than the finest scale gets diluted below the threshold and is not found. That
# is a real property of the sliding-window MVP, documented in the README under
# "Known weaknesses" -- the fix is a finer scale, at proportional cost.
SQUARE = (160, 160, 352, 352)  # x0, y0, x1, y1


def make_image(seed: int = 0, tampered: bool = True, all_smooth: bool = False) -> Image.Image:
    """A gradient 'photograph' with sensor-like noise, optionally with a smooth patch."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    base = 90 + 60 * (xx / SIZE) + 40 * (yy / SIZE)
    base = np.stack([base, base * 0.95 + 10, base * 0.9 + 20], axis=-1)

    noise = rng.normal(0.0, 9.0, size=base.shape)
    if all_smooth:
        arr = base  # no sensor noise anywhere: a wholly synthetic image
    else:
        arr = base + noise
        if tampered:
            x0, y0, x1, y1 = SQUARE
            arr[y0:y1, x0:x1] = base[y0:y1, x0:x1] + noise[y0:y1, x0:x1] * 0.12

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def smoothness_scorer(patches):
    """Score a patch by how *little* fine detail it has.

    A stand-in for the over-smoothing an upsampling decoder leaves. Measures the
    patch's own pixels -- it has no idea where the tampered square is.
    """
    scores = []
    for patch in patches:
        gray = np.asarray(patch.convert("L"), dtype=np.float64)
        detail = float(np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean())
        scores.append(float(np.clip(1.0 - detail / 12.0, 0.0, 1.0)))
    return scores


def _analyzer(**overrides):
    # Scales are left at the shipped default so these tests exercise the real
    # configuration; only the thresholds are pinned, so a later change to the
    # defaults doesn't silently rewrite what the tests assert.
    config = {
        "mapper": {"threshold_hi": 0.7, "threshold_lo": 0.4},
        "routing": {"face_detection": False},
    }
    for key, value in overrides.items():
        config.setdefault(key, {}).update(value)
    return RegionAwareAnalyzer(config, scorer=CallableBackend(smoothness_scorer, name="stub"))


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def test_locates_the_tampered_square():
    report = _analyzer().analyse(make_image(tampered=True))

    x0, y0, x1, y1 = SQUARE
    inside = report.amap.heat[y0:y1, x0:x1]
    outside_mask = np.ones(report.amap.heat.shape, dtype=bool)
    outside_mask[y0:y1, x0:x1] = False
    outside = report.amap.heat[outside_mask]

    assert np.nanmean(inside) > np.nanmean(outside) + 0.3, (
        f"map did not separate the edit: inside {np.nanmean(inside):.3f} "
        f"vs outside {np.nanmean(outside):.3f}"
    )
    assert (report.amap.labels[y0:y1, x0:x1] == LABEL_AI).mean() > 0.3

    findings = report.verdict.findings
    assert findings, "no region proposed for a clearly-marked edit"
    best = max(findings, key=lambda f: _iou(f.region.bbox, SQUARE))
    assert _iou(best.region.bbox, SQUARE) > 0.45, f"region {best.region.bbox} vs truth {SQUARE}"


def test_the_finest_scale_sets_how_well_extent_is_recovered():
    """Pins the measured cost of dropping the fine scale.

    A region is only confidently mapped where a window fits inside it, so the
    finest scale bounds how small an edit the map can *outline* (as opposed to
    merely centre on). Without the 64px scale, a 192px edit is still found and
    still centred correctly -- the recovered box just shrinks toward the middle.
    This is asserted rather than merely documented so that a future change to
    the default scales has to confront the trade rather than absorb it silently.
    """
    fine = _analyzer().analyse(make_image(tampered=True))
    coarse = _analyzer(mapper={"scales": [128, 224]}).analyse(make_image(tampered=True))

    fine_iou = max(_iou(f.region.bbox, SQUARE) for f in fine.verdict.findings)
    coarse_iou = max(_iou(f.region.bbox, SQUARE) for f in coarse.verdict.findings)

    assert fine_iou > 0.55, f"three-scale localisation regressed to {fine_iou:.2f}"
    assert coarse_iou < fine_iou, "the fine scale should improve extent recovery"

    # Both still find the edit and both still centre on it -- only extent differs.
    cx, cy = (SQUARE[0] + SQUARE[2]) / 2, (SQUARE[1] + SQUARE[3]) / 2
    for report in (fine, coarse):
        box = max(report.verdict.findings, key=lambda f: f.region.area_frac).region.bbox
        assert abs((box[0] + box[2]) / 2 - cx) < 24 and abs((box[1] + box[3]) / 2 - cy) < 24


def test_localised_edit_reads_as_edited_not_generated():
    report = _analyzer().analyse(make_image(tampered=True))
    assert report.score > 0.5, f"score {report.score:.3f}"
    assert report.verdict.verdict in {"ai_edited", "manipulated_not_necessarily_ai", "uncertain"}, (
        report.verdict.verdict
    )
    routed = {f.route.primary for f in report.verdict.findings}
    assert "synthesis" not in routed, "a 7%-of-frame edit must not route to whole-image synthesis"


def test_wholly_synthetic_image_takes_the_synthesis_route():
    report = _analyzer().analyse(make_image(all_smooth=True))
    assert report.verdict.findings, "a fully smooth image should raise a region"
    top = max(report.verdict.findings, key=lambda f: f.region.area_frac)
    assert top.region.area_frac > 0.55, f"coverage {top.region.area_frac:.2f}"
    assert top.route.primary == "synthesis", f"routed to {top.route.primary}"
    assert report.verdict.verdict in {"ai_generated", "uncertain"}


def test_clean_photograph_raises_nothing():
    report = _analyzer().analyse(make_image(tampered=False))
    assert not report.verdict.findings, (
        f"false positives: {[f.region.bbox for f in report.verdict.findings]}"
    )
    assert report.score < 0.5, f"score {report.score:.3f}"
    assert report.verdict.verdict == "likely_authentic"


def test_report_serialises_and_explains_itself():
    report = _analyzer().analyse(make_image(tampered=True))
    payload = report.to_dict()

    import json

    json.loads(json.dumps(payload, default=str))  # must be JSON-clean

    assert payload["explanation"], "every verdict needs an explanation"
    assert 0.0 <= payload["confidence"] <= 1.0
    for region in payload["regions"]:
        assert region["evidence"], "a finding with no evidence is unreviewable"
        assert region["routing_reason"], "a routing decision must say why"
    # Uncalibrated runs must admit it rather than claim confidence they haven't earned.
    assert any("uncalibrated" in note for note in payload["notes"])
    assert payload["confidence"] <= 0.60 + 1e-9


def _region(mean_score: float, support: float = 0.9, disagreement: float = 0.02) -> Region:
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    return Region(
        region_id=1, mask=mask, bbox=(16, 16, 48, 48), area_px=1024, area_frac=0.25,
        centroid=(32.0, 32.0), mean_score=mean_score, max_score=mean_score + 0.05,
        p90_score=mean_score + 0.03, scale_profile={128: mean_score, 224: mean_score},
        scale_disagreement=disagreement, boundary_sharpness=0.05, compactness=0.8,
        fill_ratio=1.0, uncertain_halo_frac=0.2, touches_border=False, support=support,
    )


def test_fusion_shrinks_a_low_confidence_specialist_toward_the_map():
    region = _region(mean_score=0.55)
    route = RouteDecision(1, "inpainting", [], "test")
    shouty = SpecialistResult("inpainting", probability=0.99, confidence=0.10)
    finding = fuse_region(region, route, [shouty], thresholds=(0.45, 0.75))
    assert finding.probability < 0.65, (
        f"a 0.10-confidence specialist moved a 0.55 map to {finding.probability:.3f}"
    )


def test_fusion_caps_by_map_support():
    # Map barely backed this region (weak score, thin support), so no specialist
    # may push it to near-certainty.
    region = _region(mean_score=0.50, support=0.10, disagreement=0.30)
    route = RouteDecision(1, "inpainting", [], "test")
    confident = SpecialistResult("inpainting", probability=1.0, confidence=1.0)
    finding = fuse_region(region, route, [confident], thresholds=(0.45, 0.75))
    assert finding.probability <= 0.80, f"uncapped at {finding.probability:.3f}"
    assert any("capped" in e for e in finding.evidence)


def test_disagreeing_specialists_lower_confidence():
    region = _region(mean_score=0.80)
    route = RouteDecision(1, "inpainting", ["compositing"], "test")
    agree = [
        SpecialistResult("inpainting", 0.85, 0.8),
        SpecialistResult("compositing", 0.83, 0.8),
    ]
    disagree = [
        SpecialistResult("inpainting", 0.95, 0.8),
        SpecialistResult("compositing", 0.25, 0.8),
    ]
    thresholds = (0.45, 0.75)
    assert (
        fuse_region(region, route, agree, thresholds).confidence
        > fuse_region(region, route, disagree, thresholds).confidence
    )


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} pipeline tests passed")


if __name__ == "__main__":
    run()
