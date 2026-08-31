"""The dual-backend gate: trust, alignment, and what a fallback report may claim.

The gate exists because localisation is both the gain and the fragility, so the
properties worth pinning are the ones that make a wrong decision safe rather
than the ones that make a right decision good:

* an unfitted aligner must not pretend the two scales agree
* alignment must be monotone, or it reorders an arm it was only meant to rescale
* a distrusted image must not report the regions it just declined to believe
* the fallback backend must not be loaded by a run that never distrusts

Torch-free: the analyser and both scorers are stubbed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dual_backend import DualBackendAnalyzer, ScoreAligner  # noqa: E402


class _Verdict:
    def __init__(self, score, conf_uncapped, findings):
        self.score = score
        self.verdict = "ai_edited"
        self.confidence = min(score, 0.6)
        self.explanation = ""
        self.findings = findings
        self.details = {"confidence_uncapped": conf_uncapped, "whole_image_score": 0.5}


class _Map:
    working_image = Image.new("RGB", (16, 16))
    scale_factor = 1.0

    def summary(self):
        return {"frac_likely_ai": 0.1}


class _Report:
    def __init__(self, score, conf_uncapped, n_findings=2):
        self.verdict = _Verdict(score, conf_uncapped, [object()] * n_findings)
        self.amap = _Map()
        self.image = self.amap.working_image
        self.timings = {}
        self.notes = []
        self.backend = {"backend": "stub-primary"}

    def to_dict(self):
        return {
            "score": self.verdict.score,
            "regions": [{"id": i} for i, _ in enumerate(self.verdict.findings)],
            "notes": list(self.notes),
            "timings_sec": dict(self.timings),
        }


class _Primary:
    def __init__(self, score=0.9, conf=0.95):
        self.score, self.conf = score, conf

    def analyse(self, image):
        return _Report(self.score, self.conf)


class _Fallback:
    name = "stub-fallback"

    def __init__(self):
        self.calls = 0

    def score_patches(self, images):
        self.calls += 1
        return [0.42]

    def describe(self):
        return {"backend": self.name}


def _analyzer(conf, threshold=0.8577, backend="stub-fallback-spec", **fb):
    """Separate-backend mode by default; pass backend="self" for the shipped default."""
    cfg = {"trust": {"threshold": threshold}, "fallback": {"backend": backend, **fb}}
    return DualBackendAnalyzer(cfg, primary=_Primary(conf=conf), fallback_scorer=_Fallback())


def test_identity_aligner_is_a_no_op_and_admits_it():
    aligner = ScoreAligner.identity()
    assert aligner.fitted is False
    for x in (0.0, 0.37, 1.0):
        assert aligner(x) == x


def test_fit_is_monotone_and_moves_one_scale_onto_the_other():
    rng = np.random.default_rng(0)
    fallback = rng.beta(2, 5, size=500)      # crowded low
    primary = rng.beta(5, 2, size=500)       # crowded high
    aligner = ScoreAligner.fit(fallback, primary)
    assert aligner.fitted is True

    probe = np.linspace(fallback.min(), fallback.max(), 50)
    mapped = np.array([aligner(x) for x in probe])
    assert np.all(np.diff(mapped) >= -1e-12), "alignment reordered the fallback arm"
    # The point of fitting: the fallback's median lands near the primary's.
    assert abs(aligner(float(np.median(fallback))) - float(np.median(primary))) < 0.1


def test_trusted_image_keeps_its_regions_and_never_builds_the_fallback():
    analyzer = _analyzer(conf=0.95)
    analyzer._fallback_scorer = None          # so a build would be observable
    analyzer._fallback_spec = "nonexistent-backend-would-raise"

    out = analyzer.analyse(Image.new("RGB", (16, 16)))
    assert out.trusted is True
    assert out.source == "region_aware"
    assert out.score == 0.9
    assert len(out.to_dict()["regions"]) == 2


def test_distrusted_image_uses_the_fallback_and_drops_its_regions():
    analyzer = _analyzer(conf=0.10)
    out = analyzer.analyse(Image.new("RGB", (16, 16)))

    assert out.trusted is False
    assert out.source == "fallback"
    assert out.score == 0.42
    # The regions were just judged unreliable; reporting them beside a score
    # that ignored them is what invites a reader to believe both.
    assert out.to_dict()["regions"] == []
    assert any("not trusted" in n for n in out.to_dict()["notes"])


def test_unfitted_alignment_is_declared_in_the_notes():
    out = _analyzer(conf=0.10).analyse(Image.new("RGB", (16, 16)))
    assert out.aligned is False
    assert any("no alignment was fitted" in n for n in out.to_dict()["notes"])


def test_eager_scores_the_fallback_even_when_trusted():
    analyzer = _analyzer(conf=0.95, eager=True)
    out = analyzer.analyse(Image.new("RGB", (16, 16)))
    assert out.trusted is True
    assert out.source == "region_aware"        # trusted still wins the score
    assert out.fallback_score == 0.42          # but both arms exist for comparison


def test_distrust_direction_can_be_inverted_for_a_larger_is_worse_signal():
    cfg = {"trust": {"signal": "frac_likely_ai", "threshold": 0.25, "distrust_below": False}}
    analyzer = DualBackendAnalyzer(cfg, primary=_Primary(), fallback_scorer=_Fallback())
    out = analyzer.analyse(Image.new("RGB", (16, 16)))
    assert out.signal_value == 0.1             # read off the map summary, not details
    assert out.trusted is True                 # 0.1 <= 0.25 -> fine


def test_self_fallback_is_the_default_and_reuses_the_existing_whole_image_pass():
    """The shipped default: no second model, and no second forward pass.

    Measured on shard 4, a separate detector was the worse fallback (mean AUC
    0.745 against 0.806 for the primary's own global view), and analyze.py has
    already computed that number -- so the better arm is also the free one.
    """
    analyzer = DualBackendAnalyzer(
        {"trust": {"threshold": 0.8577}}, primary=_Primary(conf=0.10)
    )
    assert analyzer.uses_separate_backend is False
    assert analyzer.fallback_scorer is None, "self mode must not build a second backend"

    out = analyzer.analyse(Image.new("RGB", (16, 16)))
    assert out.trusted is False
    assert out.source == "fallback"
    assert out.score == 0.5                     # _Verdict.details["whole_image_score"]
    assert "fallback" not in out.timings, "self mode must not cost a forward pass"
    # Same backend, same scale -- an alignment note here would be noise.
    assert not any("alignment" in n for n in out.to_dict()["notes"])
    assert out.to_dict()["regions"] == []


def test_self_fallback_still_reports_which_arm_decided():
    out = DualBackendAnalyzer({"trust": {"threshold": 0.0}}, primary=_Primary(conf=0.9)).analyse(
        Image.new("RGB", (16, 16))
    )
    meta = out.to_dict()["dual_backend"]
    assert meta["source"] == "region_aware"
    assert meta["trusted_localisation"] is True
    assert meta["signal"] == "confidence_uncapped"
