"""Two detectors, one verdict: use localisation only where it can be trusted.

Measured on SID-Set shard 4 (job 779811), the region-aware pipeline is both the
whole gain and the whole fragility. Against a plain whole-image pass it is worth
+0.28 AUC on clean tampered images and -0.24 at jpeg_q30, where heavy
compression makes the map fire on almost every frame it sees. Two follow-ups
narrowed what can be done about that:

* Re-placing the map's thresholds does not fix it. Adaptive cuts (quantile,
  median_shift in mapper/heatmap.py) removed the distribution shift and
  recovered almost nothing, because at those severities the per-patch signal is
  gone rather than mis-cut -- the whole-image pass still separates at 0.605
  while every fused variant sits at 0.37-0.45.
* Choosing *per image* whether to trust the map does work. The best single
  signal was the pipeline's own pre-cap confidence, which lifted mean AUC across
  14 conditions from 0.693 to 0.806 and the worst condition from 0.368 to 0.666.

So this module does not try to make localisation robust. It decides when to
listen to it, and falls back to a whole-image detector when it should not.

    primary   region-aware pipeline (patch scorer + map + specialists)
    fallback  a plain whole-image detector, scored once
    trust     a signal read off the primary's own report

Three things this owns that a bare `if` would get wrong:

**The two scores are not on the same scale.** Substituting one for the other
changes the ranking for reasons that have nothing to do with the image. On
shard 4 that inflated the apparent gain from +0.038 to +0.112 AUC. `ScoreAligner`
maps the fallback's distribution onto the primary's before either is used, and
an unfitted aligner says so rather than pretending the scales match.

**Falling back means the regions are not trusted either.** Reporting a fallback
score beside the regions that were just judged unreliable invites a reader to
believe both. When the gate fires the findings are dropped and the note says why.

**The trust threshold is not a universal constant.** It was tuned against the
patch scorer under absolute cuts; carried unchanged onto median_shift cuts it
made results worse than either arm alone. It belongs in config, per deployment.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from analyze import AnalysisReport, RegionAwareAnalyzer, _merge, _resolve_path

# Tuned by leave-one-condition-out on SID-Set shard 4 against the patch-scorer
# backend under absolute cuts. Valid for that combination only -- see the
# module docstring. Held-out validation on shard 3 is job 780095.
DEFAULT_TRUST_THRESHOLD = 0.8577

DEFAULTS: dict = {
    "trust": {
        # Which field of the primary report decides. "confidence_uncapped" is
        # the pre-cap value from fusion.py: the reported `confidence` saturates
        # at UNCALIBRATED_CONFIDENCE_CAP on an uncalibrated map and cannot
        # discriminate at all.
        "signal": "confidence_uncapped",
        "threshold": DEFAULT_TRUST_THRESHOLD,
        # Below the threshold -> distrust. Set False for signals that mean the
        # opposite (e.g. frac_likely_ai, where a *large* value is the bad case).
        "distrust_below": True,
    },
    "fallback": {
        # "self" reuses the whole-image pass the primary already computed.
        #
        # This is the default because a second *model* measured worse than the
        # primary's own global view. On SID-Set shard 4, mean AUC across 14
        # conditions: primary alone 0.693, self-fallback 0.806, a separate
        # sdxl-detector as fallback 0.745 (0.728 with the fitted alignment).
        # The useful split is between two *pathways* -- local evidence versus
        # global -- not between two detectors, and the fine-tuned backend's own
        # global pass is simply the better of the two global views.
        #
        # It is also free: analyze.py already scores the frame once and passes
        # that number down, so this arm costs no extra forward pass. Set a Hub
        # id here to route to a genuinely separate model instead.
        "backend": "self",
        # Only meaningful for a separate backend; "self" is already on scale.
        "alignment_path": None,
        # Score the fallback on every image rather than only on distrust. Costs
        # one extra whole-image pass per image and is what evaluation needs, so
        # both arms exist for the same frame; production wants it off.
        "eager": False,
    },
}


class ScoreAligner:
    """Monotone map from fallback scores onto the primary's score scale.

    Quantile mapping: push a score through the fallback's CDF and back through
    the primary's inverse CDF, so the two arms produce comparable numbers and a
    downstream threshold means one thing rather than two. Monotone by
    construction, so it can never reorder either arm's own ranking -- it only
    makes them commensurable.

    `identity()` is the honest default. An unfitted aligner leaves scores
    untouched and reports `fitted=False`, which callers surface as a note; it
    does not silently pretend the scales agree.
    """

    def __init__(self, src: np.ndarray | None = None, dst: np.ndarray | None = None):
        self.src = None if src is None else np.asarray(src, dtype=np.float64)
        self.dst = None if dst is None else np.asarray(dst, dtype=np.float64)

    @property
    def fitted(self) -> bool:
        return self.src is not None and self.dst is not None and self.src.size > 1

    @classmethod
    def identity(cls) -> "ScoreAligner":
        return cls()

    @classmethod
    def fit(cls, fallback_scores, primary_scores, knots: int = 257) -> "ScoreAligner":
        """Fit on paired-or-unpaired samples of each arm over the same corpus.

        The samples need not be paired image-for-image -- this matches
        distributions, not individual images -- but they must come from the same
        population, or the map encodes the difference between two corpora.
        """
        fb = np.asarray([s for s in fallback_scores], dtype=np.float64)
        pr = np.asarray([s for s in primary_scores], dtype=np.float64)
        if fb.size < 2 or pr.size < 2:
            raise ValueError("need at least two samples per arm to fit an aligner")
        q = np.linspace(0.0, 1.0, knots)
        return cls(np.quantile(fb, q), np.quantile(pr, q))

    def __call__(self, score: float) -> float:
        if not self.fitted:
            return float(score)
        return float(np.interp(float(score), self.src, self.dst))

    def to_dict(self) -> dict:
        return {
            "kind": "quantile",
            "fitted": self.fitted,
            "src": None if self.src is None else self.src.tolist(),
            "dst": None if self.dst is None else self.dst.tolist(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path | None) -> "ScoreAligner":
        if not path:
            return cls.identity()
        resolved = _resolve_path(path)
        if resolved is None or not Path(resolved).exists():
            return cls.identity()
        blob = json.loads(Path(resolved).read_text())
        if not blob.get("fitted"):
            return cls.identity()
        return cls(blob["src"], blob["dst"])


@dataclass
class DualReport:
    """The primary report, plus which arm was believed and why."""

    report: AnalysisReport
    trusted: bool
    signal_name: str
    signal_value: float
    threshold: float
    score: float
    source: str                      # "region_aware" | "fallback"
    fallback_score: float | None = None
    fallback_backend: dict = field(default_factory=dict)
    aligned: bool = False
    timings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self):
        return self.report.verdict

    def to_dict(self) -> dict:
        out = self.report.to_dict()
        out["score"] = self.score
        out["dual_backend"] = {
            "source": self.source,
            "trusted_localisation": self.trusted,
            "signal": self.signal_name,
            "signal_value": self.signal_value,
            "threshold": self.threshold,
            "fallback_score": self.fallback_score,
            "fallback_backend": self.fallback_backend,
            "fallback_score_aligned": self.aligned,
        }
        out["notes"] = list(self.report.notes) + self.notes
        out["timings_sec"] = {**self.report.timings, **self.timings}
        if not self.trusted:
            # The regions were judged unreliable for this frame. Keeping them in
            # the payload beside a score that deliberately ignored them is how a
            # reader ends up trusting both.
            out["regions"] = []
        return out


class DualBackendAnalyzer:
    """Region-aware analysis, with a whole-image detector to fall back on."""

    def __init__(self, config: dict | None = None, primary: RegionAwareAnalyzer | None = None,
                 fallback_scorer=None):
        cfg = _merge(DEFAULTS, {k: v for k, v in (config or {}).items() if k in DEFAULTS})
        self.trust_cfg = cfg["trust"]
        self.fallback_cfg = cfg["fallback"]

        self.primary = primary or RegionAwareAnalyzer(
            {k: v for k, v in (config or {}).items() if k not in DEFAULTS}
        )

        self._fallback_scorer = fallback_scorer
        self._fallback_spec = self.fallback_cfg["backend"]
        self.aligner = ScoreAligner.load(self.fallback_cfg.get("alignment_path"))

    @property
    def uses_separate_backend(self) -> bool:
        return str(self._fallback_spec) != "self"

    @property
    def fallback_scorer(self):
        """Built on first use, so a run that never distrusts never loads it."""
        if not self.uses_separate_backend:
            return None
        if self._fallback_scorer is None:
            from mapper.backends import build_backend

            self._fallback_scorer = build_backend(self._fallback_spec)
        return self._fallback_scorer

    def _signal(self, report: AnalysisReport) -> float:
        name = self.trust_cfg["signal"]
        details = report.verdict.details
        if name in details:
            return float(details[name])
        if name in report.amap.summary():
            return float(report.amap.summary()[name])
        raise KeyError(
            f"trust signal {name!r} is not in the report's details or map summary; "
            f"available details: {sorted(details)}"
        )

    def analyse(self, image: Image.Image) -> DualReport:
        report = self.primary.analyse(image)
        value = self._signal(report)

        threshold = float(self.trust_cfg["threshold"])
        if self.trust_cfg.get("distrust_below", True):
            trusted = value >= threshold
        else:
            trusted = value <= threshold

        notes: list[str] = []
        timings: dict[str, float] = {}
        fallback_score = None

        if not trusted or self.fallback_cfg.get("eager"):
            if self.uses_separate_backend:
                t0 = time.perf_counter()
                raw = float(self.fallback_scorer.score_patches([report.amap.working_image])[0])
                timings["fallback"] = time.perf_counter() - t0
                fallback_score = self.aligner(raw)
                if not self.aligner.fitted:
                    notes.append(
                        "The fallback score is on its own backend's scale -- no alignment was "
                        "fitted, so it is not directly comparable with the region-aware score. "
                        "Fit one with scripts/fit_score_alignment.py."
                    )
            else:
                # Already computed by analyze.py and shared with fusion, so this
                # arm is free and needs no alignment: same backend, same scale.
                fallback_score = float(report.verdict.details["whole_image_score"])

        if trusted:
            score, source = float(report.verdict.score), "region_aware"
        else:
            score, source = float(fallback_score), "fallback"
            notes.append(
                f"Localisation was not trusted for this image ({self.trust_cfg['signal']} "
                f"{value:.3f} < {threshold:.3f}), so the score comes from the whole-image "
                f"detector and the regions are not reported."
            )

        if self.uses_separate_backend and self._fallback_scorer is not None:
            describe = getattr(self._fallback_scorer, "describe", None)
        else:
            describe = (lambda: {"backend": "self (primary's whole-image pass)"}) if not self.uses_separate_backend else None
        return DualReport(
            report=report,
            trusted=trusted,
            signal_name=self.trust_cfg["signal"],
            signal_value=value,
            threshold=threshold,
            score=score,
            source=source,
            fallback_score=fallback_score,
            fallback_backend=(describe() if describe else {}),
            aligned=self.aligner.fitted,
            timings=timings,
            notes=notes,
        )
