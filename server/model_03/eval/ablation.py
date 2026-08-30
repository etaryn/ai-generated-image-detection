"""Does the region-aware idea actually work? The controlled ablation.

This is the experiment the project stands or falls on, so it is worth being
precise about what is compared.

    WITH localisation      the full model_03 pipeline: multi-scale map ->
                           regions -> specialist routing -> fusion, producing
                           `score`
    WITHOUT localisation   the same backend, on the same image, scored once as
                           a whole -- `whole_image_score`. This is exactly what
                           model_01 and model_02 do, and what every whole-image
                           detector does.

The comparison is **paired**: both arms see the identical image, through the
identical backend, in the same run. So the difference isolates the region
machinery and nothing else -- not a different model, not a different sample, not
a different preprocessing path.

Three kinds of result come out, answering different questions:

1. **Detection.** Does localisation make the image-level verdict *better*? AUC
   for both arms with a paired bootstrap confidence interval on the difference.
   Reported separately for tampered images (where the thesis predicts a gain --
   a small edit is diluted in a whole-image score) and for fully-synthetic ones
   (where it predicts nothing, since there is no localisation to do). If the
   idea works, the gain shows up in the tampered row specifically.

2. **False positives on authentic photographs.** A detection gain bought by
   flagging everything is not a gain. Both arms are thresholded so they flag the
   same fraction of AI images, then compared on how often they flag a real
   photograph. Comparing raw flag rates would be meaningless: the two arms'
   scores are on different scales, and any detector looks safer by being less
   sensitive.

3. **Capabilities the baseline does not have at all.** Localisation IoU against
   ground-truth masks, and whether a tampered photo is called `ai_edited` while
   a generated one is called `ai_generated`. A single whole-image score cannot
   do either at any threshold. These are not "wins" in a head-to-head sense --
   they are the reason the system exists -- so they are reported separately
   rather than mixed into the detection comparison.

An honest possible outcome, which this script states plainly if the data says
so: the region machinery may *not* improve detection AUC while still being worth
having for (3). And if it degrades detection *and* localises poorly, then the
idea does not work in its current form. Both conclusions are printed as such,
because an ablation that can only return "it works" is not an ablation.

    python eval/ablation.py eval_results/sid_set_calibrated.json
    python eval/ablation.py eval_results/*.json --out EVALUATION_RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.evaluate import auc  # noqa: E402

BOOTSTRAP = 2000
SEED = 0


def paired_auc_delta(
    rows: list[dict],
    positive_classes: tuple[str, ...],
    n_boot: int = BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """AUC for both arms, plus a bootstrap CI on (with - without).

    Images are resampled with replacement, stratified by class, and *both* arms
    are recomputed on each resample -- so the interval accounts for the two arms
    having seen the same images. Treating them as independent would overstate
    the uncertainty and could hide a real effect.
    """
    pos = [r for r in rows if r["class"] in positive_classes]
    neg = [r for r in rows if r["class"] == "real"]
    if not pos or not neg:
        return {}

    def arms(pos_rows, neg_rows):
        with_auc = auc(
            np.array([r["score"] for r in pos_rows]),
            np.array([r["score"] for r in neg_rows]),
        )
        without_auc = auc(
            np.array([r["whole_image_score"] for r in pos_rows]),
            np.array([r["whole_image_score"] for r in neg_rows]),
        )
        return with_auc, without_auc

    with_auc, without_auc = arms(pos, neg)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        p_idx = rng.integers(0, len(pos), len(pos))
        n_idx = rng.integers(0, len(neg), len(neg))
        a, b = arms([pos[j] for j in p_idx], [neg[j] for j in n_idx])
        deltas[i] = a - b

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "auc_with": with_auc,
        "auc_without": without_auc,
        "delta": with_auc - without_auc,
        "ci95": [float(lo), float(hi)],
        # "Significant" means the 95% interval excludes zero. At ~120 images per
        # class this is a coarse instrument; it is reported so a delta of 0.01
        # is not read as a finding.
        "significant": bool(lo > 0 or hi < 0),
    }


def false_positives_at_matched_recall(rows: list[dict], target_recall: float = 0.80) -> dict:
    """Compare the arms at an operating point where both flag the same share of AI."""
    ai = [r for r in rows if r["class"] in ("tampered", "synthetic")]
    real = [r for r in rows if r["class"] == "real"]
    if not ai or not real:
        return {}

    out = {"target_recall": target_recall, "n_ai": len(ai), "n_real": len(real)}
    for arm, key in (("with", "score"), ("without", "whole_image_score")):
        ai_scores = np.array([r[key] for r in ai])
        real_scores = np.array([r[key] for r in real])
        threshold = float(np.quantile(ai_scores, 1.0 - target_recall))
        out[arm] = {
            "threshold": threshold,
            "recall_on_ai": float((ai_scores >= threshold).mean()),
            "false_positive_rate": float((real_scores >= threshold).mean()),
        }
    out["fpr_delta"] = out["with"]["false_positive_rate"] - out["without"]["false_positive_rate"]
    return out


def baseline_only_capabilities(rows: list[dict]) -> dict:
    """What the with-localisation arm can do that a whole-image score cannot."""
    tampered = [r for r in rows if r["class"] == "tampered"]
    synthetic = [r for r in rows if r["class"] == "synthetic"]

    loc = [r["localisation"] for r in tampered if "localisation" in r]
    localisation = {}
    if loc:
        ious = np.array([m["iou"] for m in loc])
        localisation = {
            "n": len(loc),
            "mean_iou": float(ious.mean()),
            "median_iou": float(np.median(ious)),
            "mean_recall": float(np.mean([m["recall"] for m in loc])),
            "mean_precision": float(np.mean([m["precision"] for m in loc])),
            "touch_rate": float(np.mean([m["recall"] > 0.05 for m in loc])),
            "hit_rate_iou_0.10": float((ious >= 0.10).mean()),
        }

    # Can it tell an edited photograph from a generated image? One score cannot,
    # at any threshold: both are simply "AI".
    edited_right = sum(1 for r in tampered if r["verdict"] == "ai_edited")
    generated_right = sum(1 for r in synthetic if r["verdict"] == "ai_generated")
    discrimination = {
        "tampered_called_ai_edited": edited_right / len(tampered) if tampered else None,
        "synthetic_called_ai_generated": generated_right / len(synthetic) if synthetic else None,
        "note": "undefined for the without-localisation arm: a single score cannot "
                "express 'edited' vs 'generated'",
    }
    return {"localisation": localisation, "kind_discrimination": discrimination}


def conclude(detection: dict, matched: dict, capabilities: dict) -> list[str]:
    """State what the numbers support -- including when that is 'it does not work'."""
    lines = []

    tampered = detection.get("tampered", {})
    if tampered:
        delta = tampered["delta"]
        lo, hi = tampered["ci95"]
        if tampered["significant"] and delta > 0:
            lines.append(
                f"DETECTION: localisation HELPS on tampered images. AUC "
                f"{tampered['auc_without']:.3f} -> {tampered['auc_with']:.3f} "
                f"(delta {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}])."
            )
        elif tampered["significant"] and delta < 0:
            lines.append(
                f"DETECTION: localisation HURTS on tampered images. AUC "
                f"{tampered['auc_without']:.3f} -> {tampered['auc_with']:.3f} "
                f"(delta {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]). The region "
                f"machinery is discarding signal the whole-image score had."
            )
        else:
            lines.append(
                f"DETECTION: no measurable difference on tampered images. AUC "
                f"{tampered['auc_without']:.3f} vs {tampered['auc_with']:.3f} "
                f"(delta {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] includes zero). "
                f"On this sample the region machinery neither helps nor harms the "
                f"image-level verdict."
            )

    if matched:
        with_fpr = matched["with"]["false_positive_rate"]
        without_fpr = matched["without"]["false_positive_rate"]
        direction = "fewer" if with_fpr < without_fpr else "more" if with_fpr > without_fpr else "the same"
        lines.append(
            f"FALSE POSITIVES: at {matched['target_recall']:.0%} recall on AI images, the "
            f"region-aware arm flags {direction} authentic photographs "
            f"({with_fpr:.3f} vs {without_fpr:.3f})."
        )

    loc = capabilities.get("localisation") or {}
    if loc:
        lines.append(
            f"LOCALISATION (no baseline equivalent): mean IoU {loc['mean_iou']:.3f}, median "
            f"{loc['median_iou']:.3f}; overlaps the true edit at all in "
            f"{loc['touch_rate']:.0%} of tampered images."
        )

    # Each class is reported only if it was present: a tampered-only or
    # synthetic-only evaluation is a legitimate thing to run, and must not
    # crash the summary it is being run to produce.
    kind = capabilities.get("kind_discrimination") or {}
    kind_parts = []
    if kind.get("tampered_called_ai_edited") is not None:
        kind_parts.append(f"{kind['tampered_called_ai_edited']:.0%} of tampered photos called `ai_edited`")
    if kind.get("synthetic_called_ai_generated") is not None:
        kind_parts.append(
            f"{kind['synthetic_called_ai_generated']:.0%} of generated images called `ai_generated`"
        )
    if kind_parts:
        lines.append("KIND (no baseline equivalent): " + ", ".join(kind_parts) + ".")

    # A regression elsewhere is part of the answer. The region machinery can buy
    # tampered detection by spending synthetic detection, and reporting only the
    # column that improved would be the oldest trick in benchmarking.
    synthetic = detection.get("synthetic", {})
    if synthetic and synthetic["significant"]:
        direction = "gains" if synthetic["delta"] > 0 else "LOSES"
        lines.append(
            f"SYNTHETIC (where the thesis predicts no benefit -- nothing to localise): "
            f"the region-aware arm {direction} {abs(synthetic['delta']):.3f} AUC "
            f"({synthetic['auc_without']:.3f} -> {synthetic['auc_with']:.3f})."
        )

    # The overall read, stated once, so the answer to "does our idea work?" is
    # not left to the reader to assemble from five separate numbers.
    #
    # The bar is deliberately not just "did the delta clear zero". A
    # statistically significant +0.08 on top of a chance-level baseline is a
    # real directional result and still a weak detector, and a conclusion rule
    # that called that "works" would be flattering its own project. So absolute
    # performance has to clear a floor too, and a regression elsewhere is named
    # rather than netted out.
    helped = bool(tampered) and tampered["significant"] and tampered["delta"] > 0
    hurt = bool(tampered) and tampered["significant"] and tampered["delta"] < 0
    strong_detection = bool(tampered) and tampered["auc_with"] >= 0.75
    localises = bool(loc) and loc["touch_rate"] >= 0.5
    localises_well = bool(loc) and loc["mean_iou"] >= 0.25
    regressed = bool(synthetic) and synthetic["significant"] and synthetic["delta"] < -0.02

    fpr = (matched or {}).get("with", {}).get("false_positive_rate")
    unusable_fpr = fpr is not None and fpr > 0.25

    if helped and strong_detection and localises_well:
        verdict = "the idea works on this evidence -- it detects better AND localises well."
    elif helped and (localises or localises_well):
        verdict = (
            "the idea is DIRECTIONALLY VALIDATED but not yet usable. The gain over "
            "whole-image scoring is real and significant, and the baseline is at or near "
            "chance on exactly the case this was built for -- but the absolute numbers "
            "(AUC {auc:.3f}, mean IoU {iou:.3f}) are too weak to act on."
        ).format(auc=tampered["auc_with"], iou=(loc or {}).get("mean_iou", float("nan")))
    elif helped:
        verdict = (
            "detection improves but localisation misses more often than it lands, so the "
            "regions are acting as a scoring trick rather than as an explanation. Do not "
            "present the maps to users on this evidence."
        )
    elif hurt:
        verdict = (
            "the idea does NOT work in its current form -- the region machinery is "
            "DISCARDING signal the whole-image score already had."
        )
    elif localises_well:
        # No detection gain, but it reliably finds the edit. That is still worth
        # having: the baseline cannot do it at any threshold, and "where" is
        # what a reviewer actually needs to act on a flag.
        verdict = (
            "the idea earns its place on localisation, not on detection. It finds where "
            "the edit is -- which the baseline cannot do at all -- while scoring about the "
            "same at the image level."
        )
    else:
        verdict = (
            "on this evidence the idea does NOT yet work -- no measurable detection gain, "
            "and localisation misses more often than it lands."
        )

    lines.append("OVERALL: " + verdict)
    if regressed:
        lines.append(
            "  ...and it is a TRADE, not a free win: tampered detection was bought at a "
            "significant cost to synthetic detection. Whether that trade is worth making "
            "depends on which error is more expensive for the use case."
        )
    if unusable_fpr:
        lines.append(
            f"  ...and the operating point is not deployable regardless: {fpr:.0%} of "
            f"authentic photographs are flagged at the recall measured above."
        )
    return lines


def analyse(payload: dict) -> dict:
    rows = payload["per_image"]
    detection = {
        "all_ai": paired_auc_delta(rows, ("tampered", "synthetic")),
        "tampered": paired_auc_delta(rows, ("tampered",)),
        "synthetic": paired_auc_delta(rows, ("synthetic",)),
    }
    matched = false_positives_at_matched_recall(rows)
    capabilities = baseline_only_capabilities(rows)
    return {
        "backend": payload.get("backend", {}),
        "n_images": payload.get("n_images"),
        "detection": detection,
        "matched_operating_point": matched,
        "capabilities": capabilities,
        "conclusions": conclude(detection, matched, capabilities),
    }


def markdown(name: str, result: dict) -> str:
    backend = result["backend"].get("backend", "?")
    lines = [
        f"### {name}",
        "",
        f"Backend `{backend}`, {result['n_images']} images. Both arms scored on the "
        f"same images in the same run, so the comparison is paired.",
        "",
        "**Detection: with vs without localisation**",
        "",
        "| Positive class | without (whole-image) | with (region-aware) | delta | 95% CI | significant |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in (("all_ai", "all AI"), ("tampered", "**tampered**"), ("synthetic", "synthetic")):
        d = result["detection"].get(key) or {}
        if not d:
            continue
        lo, hi = d["ci95"]
        lines.append(
            f"| {label} | {d['auc_without']:.3f} | {d['auc_with']:.3f} | {d['delta']:+.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] | {'yes' if d['significant'] else 'no'} |"
        )

    matched = result.get("matched_operating_point") or {}
    if matched:
        lines += ["", f"**False positives at {matched['target_recall']:.0%} recall on AI images**", ""]
        lines += ["| Arm | threshold | recall on AI | false-positive rate on real |", "|---|---|---|---|"]
        for arm in ("without", "with"):
            m = matched[arm]
            lines.append(
                f"| {arm} | {m['threshold']:.3f} | {m['recall_on_ai']:.3f} | "
                f"{m['false_positive_rate']:.3f} |"
            )

    caps = result["capabilities"]
    loc = caps.get("localisation") or {}
    kind = caps.get("kind_discrimination") or {}
    if loc or kind.get("tampered_called_ai_edited") is not None:
        lines += ["", "**Capabilities with no baseline equivalent**", ""]
    if loc:
        lines.append(
            f"- Localisation over {loc['n']} masked images: mean IoU {loc['mean_iou']:.3f}, "
            f"median {loc['median_iou']:.3f}, recall {loc['mean_recall']:.3f}, "
            f"precision {loc['mean_precision']:.3f}, touch rate {loc['touch_rate']:.3f}."
        )
    kind_parts = []
    if kind.get("tampered_called_ai_edited") is not None:
        kind_parts.append(f"{kind['tampered_called_ai_edited']:.0%} of tampered called `ai_edited`")
    if kind.get("synthetic_called_ai_generated") is not None:
        kind_parts.append(f"{kind['synthetic_called_ai_generated']:.0%} of synthetic called `ai_generated`")
    if kind_parts:
        lines.append("- Kind: " + "; ".join(kind_parts) + ".")

    lines += ["", "**Conclusion**", ""]
    lines += [f"- {line}" for line in result["conclusions"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="evaluate.py JSON outputs")
    parser.add_argument("--out", default=None, help="Write markdown here (default: stdout only)")
    parser.add_argument("--json_out", default=None, help="Also write the raw numbers here")
    args = parser.parse_args()

    sections, raw = [], {}
    for path in args.results:
        payload = json.loads(Path(path).read_text())
        name = Path(path).stem
        result = analyse(payload)
        raw[name] = result
        sections.append(markdown(name, result))

        print(f"=== {name} ===")
        for line in result["conclusions"]:
            print(f"  {line}")
        print()

    document = "# Ablation: with vs without localisation\n\n" + "\n".join(sections)
    if args.out:
        Path(args.out).write_text(document, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(raw, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
