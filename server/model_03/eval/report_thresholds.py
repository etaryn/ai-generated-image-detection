"""Compare threshold modes, and test whether localisation trust can be routed.

Two questions, both answered from the artefacts eval/robustness.py now writes:

1. Do the adaptive cuts fix the two common-mode failures (jpeg_q30 over-firing,
   noise switch-off) and what do they cost on clean data? The cost is the point:
   a mode that survives degradation by never firing has not helped.

2. Can the pipeline tell, per image, when to trust localisation? Job 779368
   showed localisation is both the entire gain (+0.28 AUC on clean tampered) and
   the entire fragility (-0.24 at jpeg_q30). If a cheap per-image signal
   separates those cases, the fused score can fall back to the whole-image score
   when the map is not trustworthy -- which is the dual-backend idea in its
   cheapest form, one backend and a gate.

Usage:
    python eval/report_thresholds.py eval_results/diag_shard4_*.json --out THRESHOLD_RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEY_CONDITIONS = ("clean", "jpeg_q30", "noise_s0.05", "downscale_0.25")

# Share of the frame the map may label AI before localisation stops being
# trusted. Tuned on shard 4 against the absolute arm -- see routing_section.
GATE_TRUST_MAX = 0.25


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC, ties at half credit. Mirrors eval/evaluate.py."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    r = ranks[: pos.size].sum()
    return float((r - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def load(paths: list[Path]) -> dict[str, dict]:
    runs = {}
    for p in paths:
        d = json.loads(p.read_text())
        backend = d.get("backend", {}).get("backend", "?")
        mode = d.get("threshold_mode", "absolute")
        short = "patch_scorer" if "patch_scorer" in str(backend) else "base"
        runs[f"{short}/{mode}"] = d
    return runs


def mode_comparison(runs: dict[str, dict]) -> list[str]:
    out = ["## 1. Threshold modes: does removing the common-mode shift help?", ""]
    out.append("`AUC(real vs tampered)`, and the share of *real* images that grew a")
    out.append("region -- localisation's false-positive rate, which is what the")
    out.append("aggregate AUC hides.")
    out.append("")
    names = [n for n in runs if n.startswith("patch_scorer/")]
    if not names:
        return out + ["_no patch-scorer arms found_", ""]
    conds = list(runs[names[0]]["conditions"])

    header = "| condition | " + " | ".join(f"{n.split('/')[1]}" for n in names) + " |"
    out += [header, "|" + "---|" * (len(names) + 1)]
    for c in conds:
        cells = []
        for n in names:
            r = runs[n]["conditions"].get(c)
            if r is None:
                cells.append("-")
                continue
            real = (r.get("by_class") or {}).get("real") or {}
            fp = real.get("frac_regions_fired")
            fp_s = f"{fp*100:.0f}%" if fp is not None else "?"
            cells.append(f"{r['auc_real_vs_tampered']:.3f} (fp {fp_s})")
        mark = " **" if c in KEY_CONDITIONS else " "
        out.append(f"|{mark}{c}{mark.strip()} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def confidence_section(runs: dict[str, dict]) -> list[str]:
    out = ["## 2. Confidence, before and after the uncalibrated cap", ""]
    out.append("`mean_confidence` saturates at UNCALIBRATED_CONFIDENCE_CAP (0.60), so it")
    out.append("cannot show the system losing certainty. The pre-cap value can.")
    out.append("")
    out += ["| run | condition | reported | pre-cap | capped |", "|---|---|---|---|---|"]
    for n, d in runs.items():
        for c in KEY_CONDITIONS:
            r = d["conditions"].get(c)
            if not r:
                continue
            unc = r.get("mean_confidence_uncapped")
            frac = r.get("frac_confidence_capped")
            out.append(
                f"| {n} | {c} | {r['mean_confidence']:.3f} | "
                f"{unc:.3f} | {frac*100:.0f}% |" if unc is not None else
                f"| {n} | {c} | {r['mean_confidence']:.3f} | - | - |"
            )
    out.append("")
    return out


def routing_section(runs: dict[str, dict]) -> list[str]:
    """Can a per-image signal say when localisation should be trusted?

    The oracle bound comes first. A gate cannot beat picking the better arm per
    image with hindsight, so if the oracle is not much above always-fuse there
    is nothing worth routing for and the honest answer is to say so.
    """
    out = ["## 3. Routing localisation trust, per image", ""]
    name = "patch_scorer/absolute"
    d = runs.get(name)
    if not d or "per_image" not in d:
        return out + ["_needs per-image rows from the instrumented run_", ""]

    # "best pure" is the better of the two fixed strategies for that condition,
    # which is not an upper bound on the gate: the gate chooses per image, so it
    # can and does beat both (see blur_s2.0). It is a reference, not a ceiling.
    out += ["| condition | always fuse | always whole-image | best pure | gate | gate picks no-loc |",
            "|---|---|---|---|---|---|"]
    gains = []
    for c, rows in d["per_image"].items():
        real = [r for r in rows if r["class"] == "real"]
        tamp = [r for r in rows if r["class"] == "tampered"]
        if not real or not tamp:
            continue

        fused_t = np.array([r["score"] for r in tamp])
        fused_r = np.array([r["score"] for r in real])
        whole_t = np.array([r["whole_image_score"] for r in tamp])
        whole_r = np.array([r["whole_image_score"] for r in real])

        a_fuse = auc(fused_t, fused_r)
        a_whole = auc(whole_t, whole_r)
        a_best_pure = max(a_fuse, a_whole)

        # The gate: distrust the map when it lights up an implausible share of
        # the frame. One-sided, because a grid over both bounds on shard 4 put
        # the lower one at zero -- an under-firing map contributes nothing to
        # the fused score anyway, so gating it changes nothing and only the
        # over-firing side is worth a rule.
        #
        # GATE_TRUST_MAX is distribution-specific, not universal: it was tuned
        # against the absolute-threshold arm, and applying it unchanged to
        # median_shift (which fires more) makes that arm worse rather than
        # better. Retune per threshold_mode and per backend.
        def gated(rs):
            return np.array([
                r["score"] if r["map"].get("frac_likely_ai", 0.0) <= GATE_TRUST_MAX
                else r["whole_image_score"]
                for r in rs
            ])

        a_gate = auc(gated(tamp), gated(real))
        picks = np.mean([
            r["map"].get("frac_likely_ai", 0.0) > GATE_TRUST_MAX for r in rows
        ])
        gains.append((c, a_gate - a_fuse))
        out.append(
            f"| {c} | {a_fuse:.3f} | {a_whole:.3f} | {a_best_pure:.3f} | "
            f"{a_gate:.3f} | {picks*100:.0f}% |"
        )
    out.append("")
    if gains:
        mean_gain = float(np.mean([g for _, g in gains]))
        worst = min(gains, key=lambda kv: kv[1])
        out.append(f"Mean gate gain over always-fuse: **{mean_gain:+.3f}** AUC.")
        out.append(f"Worst condition for the gate: `{worst[0]}` at {worst[1]:+.3f}.")
        out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("THRESHOLD_RESULTS.md"))
    args = ap.parse_args()

    runs = load(args.results)
    lines = ["# model_03 - threshold modes and localisation routing", ""]
    lines.append(f"Arms: {', '.join(sorted(runs))}")
    lines.append("")
    lines += mode_comparison(runs)
    lines += confidence_section(runs)
    lines += routing_section(runs)

    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
