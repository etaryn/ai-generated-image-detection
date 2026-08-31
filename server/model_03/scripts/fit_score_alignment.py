"""Fit the quantile map that puts the fallback backend on the primary's scale.

dual_backend.py substitutes a whole-image score for the region-aware score when
localisation is not trusted. Those two numbers come from different detectors and
are not on the same scale, so substituting one for the other shifts the ranking
for reasons unrelated to the image. On SID-Set shard 4 that accounted for most
of the gate's apparent benefit: +0.112 AUC raw against +0.038 once both arms
were put on a common scale.

Fit on CLEAN images only, and on a corpus that looks like production traffic.
Fitting across degraded conditions would bake the degradation into the mapping
and quietly undo the thing the gate exists to handle.

Usage:
    python scripts/fit_score_alignment.py \
        --primary  eval_results/diag_shard4_ps_absolute.json \
        --fallback eval_results/diag_shard4_base_absolute.json \
        --out configs/score_alignment_patch_scorer_vs_sdxl.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dual_backend import ScoreAligner  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary", type=Path, required=True,
                    help="instrumented robustness run for the region-aware arm")
    ap.add_argument("--fallback", type=Path, required=True,
                    help="instrumented robustness run for the fallback backend")
    ap.add_argument("--condition", default="clean",
                    help="which condition to fit on (default: clean -- see module docstring)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    primary = json.loads(args.primary.read_text())
    fallback = json.loads(args.fallback.read_text())
    for name, blob, path in (("primary", primary, args.primary), ("fallback", fallback, args.fallback)):
        if "per_image" not in blob:
            raise SystemExit(f"{path} has no per_image rows -- rerun with the instrumented harness")
        if args.condition not in blob["per_image"]:
            raise SystemExit(f"{path} has no condition {args.condition!r}")

    prim_rows = {r["stem"]: r for r in primary["per_image"][args.condition]}
    fall_rows = {r["stem"]: r for r in fallback["per_image"][args.condition]}
    shared = sorted(set(prim_rows) & set(fall_rows))
    if len(shared) < 2:
        raise SystemExit(
            f"only {len(shared)} images in common -- the two runs must cover the same corpus"
        )

    # The fallback arm contributes its whole-image score, because that is the
    # number dual_backend.py actually substitutes; the primary contributes its
    # fused score, which is what that substitution has to be comparable with.
    src = [fall_rows[s]["whole_image_score"] for s in shared]
    dst = [prim_rows[s]["score"] for s in shared]

    aligner = ScoreAligner.fit(src, dst)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    aligner.save(args.out)

    lo_s, hi_s = min(src), max(src)
    print(f"fitted on {len(shared)} paired images from '{args.condition}'")
    print(f"  fallback range {lo_s:.3f}..{hi_s:.3f}  ->  "
          f"{aligner(lo_s):.3f}..{aligner(hi_s):.3f} on the primary's scale")
    print(f"  primary  range {min(dst):.3f}..{max(dst):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
