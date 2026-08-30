"""The pipeline: image in, region-aware report out.

    [1] AI-likelihood mapper        mapper/
    [2] region proposals            regions/
    [3] routing -> specialists      router.py, specialists/
    [4] evidence fusion             fusion.py
        -> tamper map, regional labels, verdict, confidence, explanation

Two things this stage owns, beyond wiring:

**Short-circuiting.** If the map finds no region above the "likely AI"
threshold, no specialist runs. That is the point of building the map first --
specialist compute is spent only where there is something to characterise --
and it is what makes the pipeline's cost scale with how suspicious an image is
rather than with its size. The whole-image detector pass still happens, so a
clean image still gets a real score, just cheaply.

**One whole-image score, computed once.** Fusion needs it, the synthesis
specialist needs it, and `predict_image()` returns it when nothing else fires.
Scoring the frame once and passing it down keeps those three from disagreeing
about the same number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from fusion import FusedVerdict, RegionFinding, fuse, fuse_region
from mapper.calibration import ScaleCalibrators
from mapper.heatmap import AILikelihoodMap, AILikelihoodMapper
from regions.proposals import extract_regions
from router import Router
from specialists import build_specialists, detect_faces, faces_available
from specialists.base import SpecialistContext

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


DEFAULTS: dict = {
    "backend": {
        "name": "hf",
        "model_id": None,        # None -> mapper.backends.DEFAULT_HF_MODEL
        "positive_label": None,  # only set when a model's labels are unrecognised
        "positive_index": None,
        "batch_size": 32,
        "device": None,
        "fp16": None,            # None -> True on CUDA, False on CPU
        "checkpoint": None,      # model_01 / model_02 only
    },
    "mapper": {
        "scales": [64, 128, 224],
        "overlap": 0.5,
        "scale_combine": "max",
        "threshold_hi": 0.75,
        "threshold_lo": 0.45,
        "max_side": 1024,
        "smoothing": "guided",
        "smooth_radius": 8,
        "smooth_eps": 0.001,
        "min_support": 0.15,
        "calibration_path": None,
    },
    "regions": {
        "min_area_frac": 0.004,
        "grow_into_uncertain": 3,
        "close_radius": 2,
        "halo_radius": 6,
        "max_regions": 8,
    },
    "routing": {
        "run_alternates": True,
        "alternate_min_area_frac": 0.02,
        "face_detection": True,
        "whole_image_frac": 0.55,
        "scale_agreement_max": 0.12,
        "face_overlap_min": 0.35,
        "sharp_boundary": 0.03,
        "fine_scale_margin": 0.06,
        "rectangular_fill": 0.88,
        "weak_score": 0.6,
        "min_support": 0.35,
        "min_area_frac": 0.008,
        "unstable_scales": 0.22,
    },
    "fusion": {"min_confidence": 0.35, "decide_at": 0.5},
}


def _resolve_path(path: str | Path | None) -> Path | None:
    """Resolve a config path relative to model_03/, not to the caller's cwd.

    The config names `configs/calibration_*.json`, and the Streamlit client
    imports this package from the repository root while the CLI runs from
    inside model_03. Resolving against cwd would make the same config load a
    calibrator in one case and silently fail in the other.
    """
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    local = Path(__file__).resolve().parent / candidate
    return local if local.exists() else candidate


def _merge(base: dict, override: dict | None) -> dict:
    """Recursive dict merge -- a config file may set one key without restating the rest."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config on top of DEFAULTS. Missing file -> defaults, no error."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return _merge(DEFAULTS, None)
    import yaml

    return _merge(DEFAULTS, yaml.safe_load(path.read_text()) or {})


@dataclass
class AnalysisReport:
    """Everything the pipeline concluded, serialisable and renderable."""

    verdict: FusedVerdict
    amap: AILikelihoodMap
    image: Image.Image
    timings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    backend: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.verdict.score

    def to_dict(self) -> dict:
        sf = self.amap.scale_factor
        return {
            "verdict": self.verdict.verdict,
            "score": self.verdict.score,
            "confidence": self.verdict.confidence,
            "explanation": self.verdict.explanation,
            # Provenance: which detector scored the patches, and which of its
            # outputs was read as "AI". Every number in this report descends
            # from that choice, so a report that does not record it cannot be
            # audited after the fact.
            "backend": self.backend,
            "regions": [f.to_dict(sf) for f in self.verdict.findings],
            "map": self.amap.summary(),
            "details": self.verdict.details,
            "notes": list(self.notes),
            "timings_sec": self.timings,
        }


class RegionAwareAnalyzer:
    """Holds the loaded backend and specialists; analyses one image per call."""

    def __init__(self, config: dict | None = None, scorer=None):
        self.config = _merge(DEFAULTS, config)

        if scorer is None:
            from mapper.backends import build_backend

            backend_cfg = dict(self.config["backend"])
            name = backend_cfg.pop("name", None)
            scorer = build_backend(name, **{k: v for k, v in backend_cfg.items() if v is not None})
        self.scorer = scorer

        mapper_cfg = dict(self.config["mapper"])
        calibrator = ScaleCalibrators.load(_resolve_path(mapper_cfg.pop("calibration_path", None)))
        self.mapper = AILikelihoodMapper(scorer=self.scorer, calibrator=calibrator, **mapper_cfg)

        routing_cfg = dict(self.config["routing"])
        self.run_alternates = bool(routing_cfg.pop("run_alternates", True))
        self.alternate_min_area_frac = float(routing_cfg.pop("alternate_min_area_frac", 0.02))
        self.face_detection = bool(routing_cfg.pop("face_detection", True))
        self.router = Router(**routing_cfg)

        self.specialists = build_specialists()

    def analyse(self, image: Image.Image) -> AnalysisReport:
        timings: dict[str, float] = {}
        notes: list[str] = []

        t0 = time.perf_counter()
        amap = self.mapper.run(image)
        timings["mapping"] = time.perf_counter() - t0

        if not amap.calibrated:
            notes.append(
                "The likelihood map is uncalibrated (no fitted calibrator), so its thresholds "
                "are nominal and the reported confidence is capped. Fit one with "
                "scripts/calibrate_mapper.py."
            )

        t0 = time.perf_counter()
        regions = extract_regions(amap, **self.config["regions"])
        timings["regions"] = time.perf_counter() - t0

        faces = []
        if self.face_detection and regions:
            if faces_available():
                faces = detect_faces(amap.working_image)
            else:
                notes.append(
                    "No face detector installed (OpenCV absent), so the face-edit route was "
                    "never taken; face regions fall through to the general specialists."
                )

        t0 = time.perf_counter()
        findings: list[RegionFinding] = []
        for region in regions:
            route = self.router.route(region, faces)
            ctx = SpecialistContext(amap.working_image, amap, region, self.scorer, faces=faces)

            names = [route.primary]
            if self.run_alternates and region.area_frac >= self.alternate_min_area_frac:
                names += [n for n in route.alternates if n not in names]

            results = [self.specialists[name].analyse(ctx) for name in names if name in self.specialists]
            findings.append(fuse_region(region, route, results, amap.thresholds))
        timings["specialists"] = time.perf_counter() - t0

        # One whole-image pass, shared by fusion and by predict_image().
        t0 = time.perf_counter()
        whole_image_score = float(self.scorer.score_patches([amap.working_image])[0])
        timings["whole_image"] = time.perf_counter() - t0

        verdict = fuse(
            amap=amap,
            findings=findings,
            whole_image_score=whole_image_score,
            **self.config["fusion"],
        )

        if not regions:
            notes.append(
                "No region cleared the map's 'likely AI' threshold, so no specialist ran; "
                "the score is the whole-image detector's."
            )

        describe = getattr(self.scorer, "describe", None)
        backend = describe() if describe else {"backend": getattr(self.scorer, "name", "unknown")}

        return AnalysisReport(
            verdict=verdict,
            amap=amap,
            image=image,
            timings=timings,
            notes=notes,
            backend=backend,
        )
