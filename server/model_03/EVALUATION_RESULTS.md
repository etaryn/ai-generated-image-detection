# Results — model_03 on SID-Set

**Run:** 2026-08-30 · 360 images from SID-Set validation shard 0 (120 real / 120
fully-synthetic / 120 tampered-with-mask) · backend
`hf:Organika/sdxl-detector` · 1024px working resolution, scales [64, 128, 224].

Regenerate everything below:

```bash
python eval/fetch_sid_set.py --shard 0 --per_class 120 --out eval_data/sid_set_val
python eval/evaluate.py --out eval_results/sid_set_calibrated.json
python eval/evaluate.py --config configs/uncalibrated.yaml --out eval_results/sid_set_uncalibrated.json
python eval/ablation.py eval_results/*.json --out eval_results/tables.md
```

Raw per-image JSON stays in `eval_results/` (gitignored — regenerable, and the
images are not ours to redistribute).

---

## Summary

**The premise is confirmed. The implementation is not usable yet.** Both halves
are load-bearing and neither should be quoted without the other.

A whole-image detector scores **AUC 0.508 on locally-edited photographs** — pure
chance. It cannot see a generated region inside a real photo, which is precisely
the gap model_03 was built to close. Region-aware analysis lifts that to
**0.589**, a real and statistically significant gain (95% CI [+0.015, +0.152]).

But 0.589 is a weak detector, localisation IoU averages 0.074, and 82.5% of
authentic photographs get flagged with at least one region. The idea is
**directionally validated, not working**.

---

## 1. Detection: with vs without localisation

Both arms are paired — same images, same backend, same run — so the difference
isolates the region machinery.

| Positive class | without (whole-image) | with (region-aware) | delta | 95% CI | significant |
|---|---|---|---|---|---|
| all AI | 0.721 | 0.739 | +0.018 | [−0.022, +0.057] | no |
| **tampered** | **0.508** | **0.589** | **+0.081** | [+0.015, +0.152] | **yes** |
| synthetic | 0.935 | 0.889 | −0.045 | [−0.072, −0.021] | yes |

Read the three rows together:

- **Tampered** is the case the thesis is about, and the thesis holds. The
  baseline is at chance; localisation beats it significantly.
- **Synthetic** is the case where the thesis predicts no benefit — there is
  nothing to localise — and the region machinery actively *costs* 0.045 AUC.
- **Net across all AI, nothing changes.** This is a **trade**, not a free win:
  accuracy is redistributed from whole-image synthesis toward local edits.
  Whether that trade is worth making depends on which error is more expensive
  for the application. For a moderation tool where partial edits are the hard
  case, it probably is. Reporting only the tampered row would be dishonest.

At a matched operating point (both arms thresholded to flag 80% of AI images),
the region-aware arm flags **49.2%** of authentic photographs against the
baseline's **61.7%**. Better — and both are far too high to deploy.

## 2. Localisation

Only the region-aware arm can do this at all; a single score has no spatial
output at any threshold. Over 120 masked images: **mean IoU 0.074**, median
0.021, recall 0.189, precision 0.169, and it overlaps the true edit at all in
**53%** of cases.

The aggregate hides the real structure, which is why `evaluate.py` stratifies:

| True edit size (fraction of frame) | n | mean IoU | mean recall | touch rate |
|---|---|---|---|---|
| < 1% | 16 | 0.000 | 0.055 | 0.19 |
| 1–5% | 30 | 0.011 | 0.100 | 0.30 |
| 5–15% | 32 | 0.068 | 0.267 | 0.72 |
| > 15% | 42 | 0.151 | 0.245 | 0.69 |

This is the finest-scale floor, measured. A region is confidently mapped only
where windows fit *inside* it, so with a 64px finest scale at 1024px working
resolution, edits below ~1% of the frame are invisible — and SID-Set has plenty
of them. Above 5% the system finds the edit about 70% of the time, but still
outlines it loosely.

## 3. False positives on authentic photographs

| | uncalibrated | calibrated |
|---|---|---|
| Real images with any region | 0.825 | 0.483 |
| Mean regions per real image | 2.73 | 1.01 |
| Mean flagged area | 0.181 | 0.068 |
| Real images given a non-authentic verdict | 0.842 | 0.533 |

Flagging 82% of real photographs makes the system unusable as shipped,
regardless of its AUC. This is the single biggest obstacle to the design being
practical.

## 4. Calibration made it worse — and that is the most useful result here

Per-scale isotonic calibration was fitted on a **separate shard** and improved
held-out ECE substantially (64px 0.234 → 0.089, 128px 0.221 → 0.080, 224px
0.191 → 0.071). It still made the system worse on every headline metric:

| Metric | uncalibrated | calibrated |
|---|---|---|
| AUC real vs tampered | **0.589** | 0.539 |
| Localisation mean IoU | **0.074** | 0.039 |
| Localisation touch rate | **0.533** | 0.225 |
| Any region found on tampered | **0.858** | 0.558 |
| Real images with any region | 0.825 | **0.483** |

Calibration did exactly what it was designed to do: discount the 64px scale,
which produced most of the false regions. Real photographs with regions fell
from 83% to 48%. But **the fine scale was also carrying the localisation
signal**, so suppressing it took the true detections along with the false ones —
regions found on genuinely tampered images fell from 86% to 56%.

The conclusion is worth more than the fix would have been: the false positives
and the localisation signal come from the *same source*, so no monotone
rescaling can separate them. The scores were never really miscalibrated. A 64px
crop of a real photograph and a 64px crop of a generated region genuinely look
alike to a detector trained on whole images. That is an information problem, not
a calibration problem.

**The shipped default is therefore `calibration_path: null`.** The calibrator,
the machinery and the fitting script are kept — they are correct, and the
premise may well hold for a patch-trained backend — but enabling them by default
would trade away the system's only working capability.

## 5. Verdict labels

A bug this run exposed: only **4%** of wholly-generated images were labelled
`ai_generated` (84% came back `ai_edited`), even though the whole-image detector
separated them from real photographs at AUC 0.935. The verdict rule demanded a
synthesis-routed region covering ≥50% of the frame, so the map had to re-prove
what the detector already knew.

Fixed after this run: the verdict now follows whichever hypothesis won the
`max()` inside `fuse()`. **The verdict-confusion numbers above predate the fix**
— scores and AUCs are unaffected, only the labels — and re-running will change
them. That is exactly why the raw JSON records the config it ran under.

---

## What to do next, in order of expected value

1. **Train a patch-scale detector.** Every result here is bounded by asking a
   whole-image model about 64px crops, and §4 is direct evidence that the
   problem cannot be patched downstream. SID-Set's train split (210k images with
   masks) is public and sufficient; `mapper/backends.py` accepts a new backend
   without any other change. This is the only item likely to move the numbers a
   lot.
2. **Re-run with the verdict fix** to get honest `ai_edited` vs `ai_generated`
   numbers.
3. **Attack the false-positive rate directly** rather than through calibration —
   for example requiring corroboration across scales before proposing a region,
   instead of `max`.
4. **Compare backends** (§ EVALUATION.md) — these numbers describe
   `Organika/sdxl-detector`, and a different feature family may behave
   differently on crops.

## Caveats that bound every number here

- **SID-Set's test split is gated**, so this is the validation split. Its real
  images come from OpenImages, which appears in many training sets, so
  **detection AUC is an upper bound**. The localisation numbers are the
  trustworthy ones — no whole-image detector could have memorised a mask.
- **n = 120 per class.** AUC standard error is roughly ±0.03; differences
  smaller than that are noise, which is why the ablation reports bootstrap
  intervals rather than bare deltas.
- One backend, one dataset, one working resolution.
