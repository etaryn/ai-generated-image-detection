"""Turns a pair of `eval/robustness.py` runs (base backend vs
`checkpoints/patch_scorer`, same images, same conditions) into the headline /
unbundled comparison tables `EVALUATION_RESULTS.md` S7 builds by hand for
SID-Set. Dataset-agnostic: reports the `tampered`-specific AUC columns
whenever the input JSON has any (SID-Set), and falls back to `all_ai` only
when it doesn't (CIFAKE, and `eval/report_cifake.py` predates this -- kept as
its CIFAKE-specific wrapper since its module docstring carries the
CIFAKE-only windowing caveat this script doesn't know to state).

Usage:
    python eval/report_robustness.py \\
        eval_results/robustness_base.json \\
        eval_results/robustness_patch_scorer.json \\
        --out ROBUSTNESS_RESULTS.md
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


def has_tampered(payload: dict) -> bool:
    return any(
        row.get("auc_real_vs_tampered") == row.get("auc_real_vs_tampered")  # not NaN
        for row in payload["conditions"].values()
    )


def bundled_table(base: dict, patch: dict, metric: str, whole_metric: str, label: str) -> str:
    lines = [
        f"### {label}",
        "",
        "| Condition | baseline: base backend, no localisation | shipped: patch scorer + localisation | delta |",
        "|---|---|---|---|",
    ]
    for cond in base["conditions"]:
        b = base["conditions"][cond][whole_metric]
        s = patch["conditions"].get(cond, {}).get(metric)
        delta = (s - b) if (b == b and s == s) else None
        lines.append(f"| {cond} | {fmt(b)} | {fmt(s)} | {fmt(delta, '+.3f') if delta is not None else '--'} |")
    return "\n".join(lines)


def unbundled_table(payload: dict, label: str, metric: str, whole_metric: str, metric_label: str) -> str:
    lines = [
        f"**{label}** (`{metric_label}`, with localisation -> without):",
        "",
        "| Condition | with | without | delta |",
        "|---|---|---|---|",
    ]
    for cond, row in payload["conditions"].items():
        w, wo = row[metric], row[whole_metric]
        delta = (w - wo) if (w == w and wo == wo) else None
        lines.append(f"| {cond} | {fmt(w)} | {fmt(wo)} | {fmt(delta, '+.3f') if delta is not None else '--'} |")
    return "\n".join(lines)


def confidence_stability_table(payload: dict, label: str) -> str:
    lines = [
        f"**{label}**:",
        "",
        "| Condition | mean confidence | verdict stability vs clean | mean regions/img | loc IoU | loc recall |",
        "|---|---|---|---|---|---|",
    ]
    for cond, row in payload["conditions"].items():
        lines.append(
            f"| {cond} | {fmt(row['mean_confidence'], '.2f')} | "
            f"{fmt(row['verdict_stability_vs_clean'], '.2f')} | {fmt(row['mean_regions'], '.2f')} | "
            f"{fmt(row.get('mean_localisation_iou'))} | {fmt(row.get('mean_localisation_recall'))} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_json", help="robustness.py output for the base backend (no patch scorer)")
    parser.add_argument("patch_json", help="robustness.py output for --backend checkpoints/patch_scorer")
    parser.add_argument("--out", default="ROBUSTNESS_RESULTS.md")
    parser.add_argument("--dataset_name", default=None, help="Overrides the dataset summary's own name in the title")
    args = parser.parse_args()

    base = json.loads(Path(args.base_json).read_text())
    patch = json.loads(Path(args.patch_json).read_text())
    tampered = has_tampered(base) or has_tampered(patch)

    name = args.dataset_name or base["dataset"].get("source", "?")
    sections = [bundled_table(base, patch, "auc_real_vs_all_ai", "auc_real_vs_all_ai_whole_image", "All AI (real vs synthetic+tampered)")]
    if tampered:
        sections.append(
            bundled_table(base, patch, "auc_real_vs_tampered", "auc_real_vs_tampered_whole_image", "Tampered only (real vs partially-AI)")
        )

    unbundled = [unbundled_table(base, "Base backend", "auc_real_vs_all_ai", "auc_real_vs_all_ai_whole_image", "AUC(real vs all AI)")]
    unbundled.append(unbundled_table(patch, "Patch scorer", "auc_real_vs_all_ai", "auc_real_vs_all_ai_whole_image", "AUC(real vs all AI)"))
    if tampered:
        unbundled.append(unbundled_table(base, "Base backend", "auc_real_vs_tampered", "auc_real_vs_tampered_whole_image", "AUC(real vs tampered)"))
        unbundled.append(unbundled_table(patch, "Patch scorer", "auc_real_vs_tampered", "auc_real_vs_tampered_whole_image", "AUC(real vs tampered)"))

    doc = f"""# Results — model_03 robustness on {name}

**Dataset:** {base['dataset']['counts']}
**Base run:** backend `{base['backend'].get('backend', '?')}` · {base['images_per_class']}/class · max_side {base['max_side']}
**Patch-scorer run:** backend `{patch['backend'].get('backend', '?')}` · {patch['images_per_class']}/class · max_side {patch['max_side']}

Both arms are paired: same images, same conditions, only the backend (and
whether localisation is applied) differs. `whole_image_score`/`*_whole_image`
metrics are the backend's own single-pass score on the full image (the "no
localisation" arm); the plain metrics are the fused, region-aware score.

## 1. Headline: plain baseline vs the full upgrade

{chr(10).join(f"{s}{chr(10)}" for s in sections)}

## 2. Isolating localisation, per backend and condition

{chr(10).join(f"{s}{chr(10)}" for s in unbundled)}

## 3. Confidence, verdict stability, localisation

{confidence_stability_table(base, "Base backend")}

{confidence_stability_table(patch, "Patch scorer")}

---

Regenerate from the two `eval/robustness.py` runs this was built from:

```bash
python eval/report_robustness.py {args.base_json} {args.patch_json} --out {args.out}
```
"""
    Path(args.out).write_text(doc)
    print(doc)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
