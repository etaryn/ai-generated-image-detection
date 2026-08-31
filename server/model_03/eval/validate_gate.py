"""Held-out test of the confidence gate, with the shard-4 constants frozen.

The gate and its threshold were chosen on shard 4 (job 779811) by
leave-one-condition-out over ten candidate features. That procedure can only
report how well it did on the data it searched. This applies the winning rule
unchanged to a disjoint shard and asks whether the effect survives.

Nothing here is fitted. GATE_THRESHOLD is a literal, and `--retune` exists only
to quantify the optimism of having tuned at all: it reports what the threshold
*would* have been on this shard, as a diagnostic, never to score with.

Two estimates are reported because the gate substitutes one score source for
another, and those sources are not on a common scale:

* raw          -- substitute directly. What a deployment doing exactly this gets.
* rank-normal  -- rank-normalise both sources within a condition first, so the
                  gate cannot profit from the two distributions merely sitting
                  apart. The conservative reading.

Usage:
    python eval/validate_gate.py eval_results/val3_shard3_ps_absolute.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Frozen on shard 4. Do not refit against a validation shard.
GATE_THRESHOLD = 0.8577
GATE_FEATURE = "confidence_uncapped"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def strategies(rows: list[dict], threshold: float) -> dict[str, float]:
    real = [r for r in rows if r["class"] == "real"]
    tamp = [r for r in rows if r["class"] == "tampered"]
    if not real or not tamp:
        return {}

    distrust = {id(r): r[GATE_FEATURE] < threshold for r in rows}

    fused = np.argsort(np.argsort([r["score"] for r in rows])) / max(1, len(rows) - 1)
    whole = np.argsort(np.argsort([r["whole_image_score"] for r in rows])) / max(1, len(rows) - 1)
    index = {id(r): i for i, r in enumerate(rows)}

    def pick(subset, mode):
        out = []
        for r in subset:
            if mode == "fuse":
                out.append(r["score"])
            elif mode == "whole":
                out.append(r["whole_image_score"])
            elif mode == "gate":
                out.append(r["whole_image_score"] if distrust[id(r)] else r["score"])
            else:  # gate_rank
                i = index[id(r)]
                out.append(whole[i] if distrust[id(r)] else fused[i])
        return np.array(out)

    return {
        "fuse": auc(pick(tamp, "fuse"), pick(real, "fuse")),
        "whole": auc(pick(tamp, "whole"), pick(real, "whole")),
        "gate": auc(pick(tamp, "gate"), pick(real, "gate")),
        "gate_rank": auc(pick(tamp, "gate_rank"), pick(real, "gate_rank")),
        "distrust_frac": float(np.mean(list(distrust.values()))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--threshold", type=float, default=GATE_THRESHOLD)
    ap.add_argument("--retune", action="store_true",
                    help="Report the threshold this shard would have chosen. Diagnostic only.")
    args = ap.parse_args()

    d = json.loads(args.results.read_text())
    if "per_image" not in d:
        raise SystemExit(f"{args.results} has no per_image rows -- rerun with the instrumented harness")

    print(f"{args.results.name}  (threshold {args.threshold} frozen from shard 4)\n")
    head = f"{'condition':16s}{'fuse':>9s}{'whole':>9s}{'gate':>9s}{'gate-rank':>11s}{'distrust%':>11s}"
    print(head)
    print("-" * len(head))

    cols = {k: [] for k in ("fuse", "whole", "gate", "gate_rank")}
    for cond, rows in d["per_image"].items():
        s = strategies(rows, args.threshold)
        if not s:
            continue
        for k in cols:
            cols[k].append(s[k])
        print(f"{cond:16s}{s['fuse']:9.3f}{s['whole']:9.3f}{s['gate']:9.3f}"
              f"{s['gate_rank']:11.3f}{s['distrust_frac']*100:11.0f}")

    print("-" * len(head))
    for label, fn in (("MEAN", np.mean), ("WORST", np.min)):
        print(f"{label:16s}" + "".join(f"{fn(cols[k]):9.3f}" if k != "gate_rank"
                                       else f"{fn(cols[k]):11.3f}" for k in cols))
    print(f"{'# below 0.5':16s}" + "".join(
        f"{sum(v < 0.5 for v in cols[k]):9d}" if k != "gate_rank"
        else f"{sum(v < 0.5 for v in cols[k]):11d}" for k in cols))

    print()
    print(f"gain over always-fuse:  raw {np.mean(cols['gate']) - np.mean(cols['fuse']):+.3f} mean, "
          f"{np.min(cols['gate']) - np.min(cols['fuse']):+.3f} worst")
    print(f"                       rank {np.mean(cols['gate_rank']) - np.mean(cols['fuse']):+.3f} mean, "
          f"{np.min(cols['gate_rank']) - np.min(cols['fuse']):+.3f} worst")

    if args.retune:
        vals = np.array([r[GATE_FEATURE] for rows in d["per_image"].values() for r in rows])
        grid = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, 40)))
        best = max(grid, key=lambda t: np.mean(
            [strategies(rows, t)["gate"] for rows in d["per_image"].values() if strategies(rows, t)]))
        got = np.mean([strategies(rows, best)["gate"] for rows in d["per_image"].values()
                       if strategies(rows, best)])
        print(f"\n[diagnostic] this shard's own best threshold {best:.4f} -> {got:.3f} mean; "
              f"the frozen {args.threshold} gives {np.mean(cols['gate']):.3f}. "
              f"The gap is the optimism in the shard-4 number.")


if __name__ == "__main__":
    main()
