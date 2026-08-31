"""Turns a pair of `eval/robustness.py` CIFAKE runs into the comparison table
`EVALUATION_RESULTS.md` S7 does for SID-Set, adapted for a dataset with no
`tampered` class.

CIFAKE is two-class only (see `eval/fetch_cifake.py`'s docstring), so
`auc_real_vs_all_ai` *is* AUC(real vs AI) here -- there is no separate
`tampered` row to also report, and `auc_real_vs_tampered` in the raw JSON is
NaN by construction (empty tampered set). This script reads the two numbers
`eval/robustness.py` already computes per condition per run --
`whole_image_score`-based AUC (the backend alone, no localisation) and the
fused-score AUC (with localisation) -- so no extra evaluation run is needed
beyond the two `robustness.py` invocations that produced the input files.

Usage:
    python eval/report_cifake.py \\
        eval_results/robustness_cifake_base.json \\
        eval_results/robustness_cifake_patch_scorer.json \\
        --out CIFAKE_RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value, spec: str = ".3f") -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and value != value:  # NaN
        return "--"
    return format(value, spec)


def bundled_table(base: dict, patch: dict) -> str:
    """The headline comparison: plain baseline (base backend, no localisation)
    vs the shipped configuration (patch scorer + localisation) -- the two
    arms a user of the system actually chooses between."""
    lines = [
        "| Condition | baseline: base backend, no localisation | shipped: patch scorer + localisation | delta |",
        "|---|---|---|---|",
    ]
    for cond in base["conditions"]:
        b = base["conditions"][cond]["auc_real_vs_all_ai_whole_image"]
        s = patch["conditions"].get(cond, {}).get("auc_real_vs_all_ai")
        delta = (s - b) if (b == b and s == s) else None  # NaN-safe
        lines.append(f"| {cond} | {fmt(b)} | {fmt(s)} | {fmt(delta, '+.3f') if delta is not None else '--'} |")
    return "\n".join(lines)


def unbundled_table(payload: dict, label: str) -> str:
    """Isolates the localisation effect alone: same backend, same images, same
    condition, with vs without localisation."""
    lines = [
        f"**{label}** (`AUC(real vs AI)`, with localisation -> without):",
        "",
        "| Condition | with | without | delta |",
        "|---|---|---|---|",
    ]
    for cond, row in payload["conditions"].items():
        w = row["auc_real_vs_all_ai"]
        wo = row["auc_real_vs_all_ai_whole_image"]
        delta = (w - wo) if (w == w and wo == wo) else None
        lines.append(f"| {cond} | {fmt(w)} | {fmt(wo)} | {fmt(delta, '+.3f') if delta is not None else '--'} |")
    return "\n".join(lines)


def confidence_stability_table(payload: dict, label: str) -> str:
    lines = [
        f"**{label}**:",
        "",
        "| Condition | mean confidence | verdict stability vs clean | mean regions/img |",
        "|---|---|---|---|",
    ]
    for cond, row in payload["conditions"].items():
        lines.append(
            f"| {cond} | {fmt(row['mean_confidence'], '.2f')} | "
            f"{fmt(row['verdict_stability_vs_clean'], '.2f')} | {fmt(row['mean_regions'], '.2f')} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_json", help="robustness.py output for the base backend (no patch scorer)")
    parser.add_argument("patch_json", help="robustness.py output for --backend checkpoints/patch_scorer")
    parser.add_argument("--out", default="CIFAKE_RESULTS.md")
    args = parser.parse_args()

    base = json.loads(Path(args.base_json).read_text())
    patch = json.loads(Path(args.patch_json).read_text())

    doc = f"""# Results — model_03 on CIFAKE (real vs synthetic, no partially-AI subset)

**Dataset:** {base['dataset']['counts']} · source `{base['dataset']['source']}`
**Base run:** backend `{base['backend'].get('backend', '?')}` · {base['images_per_class']}/class · max_side {base['max_side']}
**Patch-scorer run:** backend `{patch['backend'].get('backend', '?')}` · {patch['images_per_class']}/class · max_side {patch['max_side']}

> CIFAKE ships at native 32x32. `mapper/windows.py` clamps the [64, 128, 224]
> window scales down to the image's own short side and drops the resulting
> duplicates, so on these images the multi-scale region machinery collapses to
> a single whole-image window (see `eval/fetch_cifake.py`). Any gap between
> "with" and "without localisation" below comes from `fuse()`'s
> max()-of-hypotheses logic on that one region, not from spatial evidence --
> read it as a check on whether the fusion step itself ever hurts on
> wholly-generated images, not as a localisation result.

## 1. Headline: plain baseline vs the full upgrade

{bundled_table(base, patch)}

## 2. Isolating localisation, per backend and condition

{unbundled_table(base, "Base backend")}

{unbundled_table(patch, "Patch scorer")}

## 3. Confidence and verdict stability

{confidence_stability_table(base, "Base backend")}

{confidence_stability_table(patch, "Patch scorer")}

---

Regenerate from the two `eval/robustness.py` runs this was built from:

```bash
python eval/report_cifake.py {args.base_json} {args.patch_json} --out {args.out}
```
"""
    Path(args.out).write_text(doc)
    print(doc)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
