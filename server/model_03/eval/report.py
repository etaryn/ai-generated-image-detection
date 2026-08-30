"""Turn the evaluation JSON into the markdown table that goes in EVALUATION.md.

Written as a script rather than done by hand for one reason: every number in the
documentation should be traceable to a file some command produced. Numbers
retyped from a terminal drift from the runs that made them, and a benchmark
table nobody can regenerate is indistinguishable from one that was made up.

    python eval/report.py eval_results/*.json --out EVALUATION.md
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
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def detection_table(results: list[tuple[str, dict]]) -> str:
    lines = [
        "| Run | backend | AUC real vs all AI | AUC real vs **tampered** | AUC real vs synthetic |",
        "|---|---|---|---|---|",
    ]
    for name, payload in results:
        d = payload["summary"]["detection"]
        backend = payload["backend"].get("backend", "?")
        lines.append(
            f"| {name} | `{backend}` | {fmt(d['auc_real_vs_all_ai'])} | "
            f"**{fmt(d['auc_real_vs_tampered'])}** | {fmt(d['auc_real_vs_synthetic'])} |"
        )
        lines.append(
            f"| {name} (backend alone) | `{backend}` | {fmt(d['backend_auc_real_vs_all_ai'])} | "
            f"{fmt(d['backend_auc_real_vs_tampered'])} | {fmt(d['backend_auc_real_vs_synthetic'])} |"
        )
    return "\n".join(lines)


def localisation_table(results: list[tuple[str, dict]]) -> str:
    lines = [
        "| Run | n | mean IoU | median IoU | recall | precision | touch rate | IoU>=0.10 | IoU>=0.25 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, payload in results:
        loc = payload["summary"].get("localisation") or {}
        if not loc:
            continue
        lines.append(
            f"| {name} | {loc['n']} | {fmt(loc['mean_iou'])} | {fmt(loc['median_iou'])} | "
            f"{fmt(loc['mean_recall'])} | {fmt(loc['mean_precision'])} | {fmt(loc['touch_rate'])} | "
            f"{fmt(loc['hit_rate_iou_0.10'])} | {fmt(loc['hit_rate_iou_0.25'])} |"
        )
    return "\n".join(lines)


def size_table(payload: dict) -> str:
    loc = payload["summary"].get("localisation") or {}
    bands = loc.get("by_true_area_frac") or {}
    lines = [
        "| True edit size (fraction of frame) | n | mean IoU | mean recall | touch rate |",
        "|---|---|---|---|---|",
    ]
    for band, stats in sorted(bands.items()):
        lines.append(
            f"| {band} | {stats['n']} | {fmt(stats['mean_iou'])} | "
            f"{fmt(stats['mean_recall'])} | {fmt(stats['touch_rate'])} |"
        )
    return "\n".join(lines)


def false_positive_table(results: list[tuple[str, dict]]) -> str:
    lines = [
        "| Run | real images | any region proposed | mean regions | mean flagged area | verdict not authentic |",
        "|---|---|---|---|---|---|",
    ]
    for name, payload in results:
        fp = payload["summary"]["false_positives_on_real"]
        lines.append(
            f"| {name} | {fp['n']} | {fmt(fp['images_with_any_region'])} | {fmt(fp['mean_regions'], '.2f')} | "
            f"{fmt(fp['mean_flagged_area_frac'])} | {fmt(fp['verdict_not_authentic'])} |"
        )
    return "\n".join(lines)


def confusion_table(payload: dict) -> str:
    confusion = payload["summary"]["verdict_confusion"]
    verdicts = sorted({v for row in confusion.values() for v in row})
    lines = ["| True class | " + " | ".join(f"`{v}`" for v in verdicts) + " |",
             "|---" * (len(verdicts) + 1) + "|"]
    for cls in ("real", "tampered", "synthetic"):
        row = confusion.get(cls, {})
        total = sum(row.values()) or 1
        cells = [f"{row.get(v, 0)} ({row.get(v, 0) / total * 100:.0f}%)" for v in verdicts]
        lines.append(f"| **{cls}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def routing_table(payload: dict) -> str:
    routing = payload["summary"]["routing"]
    specialists = sorted({s for row in routing.values() for s in row})
    lines = ["| True class | " + " | ".join(f"`{s}`" for s in specialists) + " |",
             "|---" * (len(specialists) + 1) + "|"]
    for cls in ("real", "tampered", "synthetic"):
        row = routing.get(cls, {})
        total = sum(row.values()) or 1
        cells = [f"{row.get(s, 0)} ({row.get(s, 0) / total * 100:.0f}%)" for s in specialists]
        lines.append(f"| **{cls}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def robustness_table(payload: dict) -> str:
    lines = [
        "| Condition | AUC real vs all AI | AUC real vs tampered | mean confidence | verdict stability | localisation recall |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in payload["conditions"].items():
        lines.append(
            f"| `{name}` | {fmt(row['auc_real_vs_all_ai'])} | {fmt(row['auc_real_vs_tampered'])} | "
            f"{fmt(row['mean_confidence'], '.2f')} | {fmt(row['verdict_stability_vs_clean'], '.2f')} | "
            f"{fmt(row['mean_localisation_recall'])} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="evaluate.py JSON outputs")
    parser.add_argument("--robustness", default=None, help="robustness.py JSON output")
    parser.add_argument("--out", default=None, help="Write markdown here (default: stdout)")
    args = parser.parse_args()

    loaded = []
    for path in args.results:
        payload = json.loads(Path(path).read_text())
        loaded.append((Path(path).stem, payload))

    primary = loaded[0][1]
    parts = [
        "## Detection\n",
        detection_table(loaded),
        "\n\n## Localisation (tampered images, against ground-truth masks)\n",
        localisation_table(loaded),
        "\n\n### Localisation by true edit size\n",
        size_table(primary),
        "\n\n## False positives on authentic photographs\n",
        false_positive_table(loaded),
        "\n\n## Verdict confusion\n",
        confusion_table(primary),
        "\n\n## Routing (descriptive -- there is no routing ground truth)\n",
        routing_table(primary),
    ]
    if args.robustness:
        parts += [
            "\n\n## Robustness\n",
            robustness_table(json.loads(Path(args.robustness).read_text())),
        ]

    markdown = "\n".join(parts) + "\n"
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
