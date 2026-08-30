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

## 6. Robustness under redistribution

14 conditions × 45 images (15 per class), 768px working resolution,
uncalibrated. `python eval/robustness.py --limit 15 --max_side 768`.

| Condition | AUC all AI | AUC tampered | verdict stability | loc IoU | loc recall | regions/img |
|---|---|---|---|---|---|---|
| clean | 0.782 | 0.604 | 1.00 | 0.046 | 0.112 | 2.20 |
| jpeg q90 | 0.787 | 0.644 | 1.00 | 0.039 | 0.097 | 2.02 |
| jpeg q60 | 0.674 | 0.462 | 0.80 | 0.023 | 0.073 | 1.96 |
| jpeg q30 | 0.764 | 0.564 | 0.82 | 0.057 | 0.150 | 3.07 |
| blur σ0.5 | 0.798 | 0.636 | 0.98 | 0.046 | 0.116 | 2.27 |
| blur σ1.0 | 0.702 | 0.502 | 0.91 | 0.082 | 0.304 | 2.78 |
| blur σ2.0 | 0.693 | 0.498 | 0.84 | 0.128 | 0.844 | 1.80 |
| downscale ×0.5 | 0.733 | 0.529 | 0.91 | 0.086 | 0.278 | 2.89 |
| downscale ×0.25 | 0.758 | 0.613 | 0.89 | 0.123 | 0.635 | 2.33 |
| **noise σ0.02** | 0.756 | 0.613 | **0.40** | 0.002 | 0.002 | 1.42 |
| **noise σ0.05** | 0.764 | 0.636 | **0.09** | 0.000 | 0.000 | 0.38 |
| crop 0.9 | 0.800 | 0.667 | 1.00 | 0.053 | 0.157 | 2.44 |
| crop 0.8 | 0.802 | 0.689 | 0.93 | 0.049 | 0.155 | 2.29 |
| saturate ×1.5 | 0.773 | 0.613 | 0.73 | 0.028 | 0.038 | 2.07 |

**Detection survives, and that is less impressive than it sounds.** AUC stays in
0.674–0.802 against 0.782 clean, with no collapse under any transform. But the
clean number is already mediocre, so what this really shows is that the system
is *uniformly* weak rather than fragile.

**The prediction on record was wrong, and the way it was wrong matters.** The
expectation was that confidence would fall and verdicts would collapse toward
`uncertain` under heavy degradation — the safe failure. Instead:

- **Confidence is 0.60 in every single condition.** That is the uncalibrated cap
  (§4), which means the confidence signal carries *no information at all* right
  now. The mechanism designed to say "I can no longer tell" does not function.
  Anything reading `confidence` today is reading a constant.
- **Under noise the system does not become unsure — it changes its mind.** At
  σ=0.05, only **9%** of images keep the verdict they had when clean, while AUC
  is essentially unchanged (0.764 vs 0.782). The ranking survives; the labels do
  not. That is the dangerous failure, not the safe one: a user re-uploading a
  slightly noisier copy of the same photo gets a different answer, confidently.

Noise also erases the local evidence completely — regions fall to 0.38 per image
and localisation IoU to 0.000 — while the global score is untouched. The
whole-image hypothesis carries the score, the local hypothesis is gone, and
nothing in the output says so.

**One artifact to not misread:** blur σ2.0 shows localisation *recall* of 0.844,
the highest in the table. That is not better localisation. IoU only rises to
0.128 while regions fall to 1.80 per image: the map is flagging fewer, much
larger areas, so it overlaps the true mask by covering half the picture. Recall
alone always rewards flagging everything, which is why IoU is in the table
beside it.

**Caveat:** 15 images per class puts the AUC standard error near ±0.09, so most
differences between conditions here are not individually significant. The
patterns worth trusting are the large ones — the noise/stability collapse, and
the flat confidence — not the ordering of individual rows. Note that jpeg q60
(0.462) scoring worse than the harsher q30 (0.564) is almost certainly sampling
noise rather than a real effect.

---

## What to do next, in order of expected value

1. **Train a patch-scale detector.** Every result here is bounded by asking a
   whole-image model about 64px crops, and §4 is direct evidence that the
   problem cannot be patched downstream. SID-Set's train split (210k images with
   masks) is public and sufficient; `mapper/backends.py` accepts a new backend
   without any other change. This is the only item likely to move the numbers a
   lot.
2. **Make `confidence` mean something.** §6 shows it is a constant 0.60 in every
   condition, so the pipeline's one mechanism for saying "the evidence has been
   degraded, do not trust this" is inert. Either the uncalibrated cap has to go,
   or confidence has to be driven by something that actually varies — map
   support and scale agreement are already computed and currently only reach it
   through a term the cap flattens.
3. **Fix verdict instability under noise** (9% stability at σ=0.05 with AUC
   unchanged). The score ranking survives while the labels churn, which points
   at the verdict thresholds rather than the evidence. Hysteresis, or deciding
   the label from the same margin the score is built on, would both help.
4. **Re-run with the verdict fix** to get honest `ai_edited` vs `ai_generated`
   numbers.
5. **Attack the false-positive rate directly** rather than through calibration —
   for example requiring corroboration across scales before proposing a region,
   instead of `max`.
6. **Compare backends** (§ EVALUATION.md) — these numbers describe
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
