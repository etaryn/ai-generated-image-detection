"""Required deliverable script: image directory in -> JSON confidence per image.

Output format is identical to model_01/infer.py and model_02/infer.py, so all
three models are drop-in comparable on the same scoring harness:

    [
      {"image_path": "/path/to/images/img001.jpg", "pred": 0.93},
      {"image_path": "/path/to/images/img002.jpg", "pred": 0.07}
    ]

model_03 knows a great deal more than one number -- where the evidence is, what
kind it is, and how much to trust it -- so `--report` writes the full region
report alongside, and `--render_dir` writes the overlays. The flat `pred` list
stays the default because the deliverable contract is the deliverable contract.

Also exposes the same single-image API the siblings do:

    load_model()                    -> warm the backend up front, not on first upload
    predict_image(pil_image)        -> float, P(AI-generated) in [0, 1]
    analyze_image(pil_image)        -> AnalysisReport, the region-aware output

so client/app.py can call model_03 exactly as it calls the other two, and use
the richer entry point when it wants the map.

Usage:
    python infer.py --input_dir /path/to/images --output predictions.json
    python infer.py --input_dir imgs --report report.json --render_dir overlays/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from analyze import AnalysisReport, RegionAwareAnalyzer, load_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Building the analyzer loads a sibling detector's checkpoint (and, for the
# model_02 backend, the frozen DINOv2/CLIP towers), which is far too slow to
# redo per upload -- so it is cached and reused across calls.
_LOADED: dict = {}


def load_model(config_path: str | Path | None = None, backend: str | None = None):
    """Build (or return the cached) analyzer. Call once at startup."""
    config_path = config_path or os.environ.get("AIGC_MODEL03_CONFIG")
    backend = backend or os.environ.get("AIGC_MODEL03_BACKEND")
    key = (str(config_path), str(backend))
    if key not in _LOADED:
        config = load_config(config_path)
        if backend:
            config["backend"]["name"] = backend
        _LOADED[key] = RegionAwareAnalyzer(config)
    return _LOADED[key]


def analyze_image(image: Image.Image, config_path: str | Path | None = None) -> AnalysisReport:
    """Full region-aware analysis of one PIL image."""
    return load_model(config_path).analyse(image)


def predict_image(image: Image.Image, config_path: str | Path | None = None) -> float:
    """Score one PIL image. Returns P(AI-generated) in [0, 1].

    This is the entry point the Streamlit client calls, and it deliberately
    matches the siblings' signature. Note that the number it returns is the
    *fused* score -- it already accounts for localised evidence a whole-image
    pass would dilute, which is the entire reason model_03 exists. The region
    detail behind it is one call away via `analyze_image`.
    """
    return float(analyze_image(image, config_path).score)


def render_overlay(report: AnalysisReport) -> Image.Image:
    """The likelihood map painted over the image. For UI callers."""
    from render import overlay_heatmap

    return overlay_heatmap(report.amap.working_image, report.amap)


def render_regions(report: AnalysisReport) -> Image.Image:
    """The overlay with each finding outlined and captioned. For UI callers."""
    from render import draw_regions, overlay_heatmap

    base = overlay_heatmap(report.amap.working_image, report.amap)
    return draw_regions(base, report.verdict.findings)


def iter_images(input_dir: str | Path):
    for path in sorted(Path(input_dir).rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--output", default="predictions.json", help="Where to write the JSON output")
    parser.add_argument("--config", default=None, help="Pipeline config YAML (default: configs/default.yaml)")
    parser.add_argument("--backend", default=None, choices=["model_01", "model_02"],
                        help="Which sibling detector scores the patches (default: from config)")
    parser.add_argument("--report", default=None,
                        help="Also write the full region-aware report (regions, routing, evidence) here")
    parser.add_argument("--render_dir", default=None,
                        help="Also write overlay/region/panel PNGs per image into this directory")
    args = parser.parse_args()

    analyzer = load_model(args.config, args.backend)

    paths = list(iter_images(args.input_dir))
    if not paths:
        raise SystemExit(f"No images found under {args.input_dir}")

    render_dir = Path(args.render_dir) if args.render_dir else None
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
        from render import draw_regions, overlay_heatmap, render_panel

    predictions, reports = [], []
    for i, path in enumerate(paths, start=1):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        report = analyzer.analyse(image)

        predictions.append({"image_path": str(path), "pred": float(report.score)})
        reports.append({"image_path": str(path), **report.to_dict()})

        if render_dir:
            stem = path.stem
            base = report.amap.working_image
            overlay_heatmap(base, report.amap).save(render_dir / f"{stem}_heatmap.png")
            draw_regions(base, report.verdict.findings).save(render_dir / f"{stem}_regions.png")
            render_panel(image, report.amap, report.verdict.findings).save(render_dir / f"{stem}_panel.png")

        print(
            f"[{i}/{len(paths)}] {path.name}: {report.verdict.verdict} "
            f"score={report.score:.3f} confidence={report.verdict.confidence:.2f} "
            f"regions={len(report.verdict.findings)}"
        )

    Path(args.output).write_text(json.dumps(predictions, indent=2))
    print(f"Wrote {len(predictions)} predictions to {args.output}")

    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=2, default=str))
        print(f"Wrote the full region report to {args.report}")
    if render_dir:
        print(f"Wrote overlays to {render_dir}/")


if __name__ == "__main__":
    main()
