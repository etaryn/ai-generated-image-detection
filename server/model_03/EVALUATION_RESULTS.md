# Results — model_03 on SID-Set

**§1–§5 run:** 2026-08-30 · 360 images from SID-Set validation shard 0 (120 real
/ 120 fully-synthetic / 120 tampered-with-mask) · backend
`hf:Organika/sdxl-detector` · 1024px working resolution, scales [64, 128, 224].

**§6–§7:** shard 3, 15/class, 768px, base backend vs the trained patch scorer.
**§8–§10:** 2026-08-31 · shards 4 and 3 at 50/class, 768px, patch scorer — a
fresh larger sample, a diagnosis of where it breaks, and the fix that came out
of it. `TEST_LOG.md` is the chronological record of every run behind all of
this, with job IDs.

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

### Where this stands, 2026-08-31

That paragraph describes the **base** backend, which is what §1–§5 measure.
Three things have changed it since, and each is documented below rather than
folded back into the numbers above:

1. **A patch-scale backend was trained** (`checkpoints/patch_scorer`), and it
   moves the thesis a long way on clean data: on the same shard-4 frames,
   AUC(real vs tampered) 0.587 → **0.854** and localisation IoU 0.041 →
   **0.416** against the base backend (§8).
2. **It is fragile in a specific, now-diagnosed way.** Under heavy JPEG,
   moderate blur and downscaling the map keeps firing but stops meaning
   anything, and fusing it *inverts* the decision — AUC(tampered) 0.376 at
   jpeg_q30 against the whole-image score's 0.598 on the same images (§8, §9).
3. **Deciding per image whether to believe the map recovers it** (§10). Gating
   on the pipeline's own pre-cap confidence lifts mean AUC(tampered) across 14
   conditions from 0.692 to **0.803** and the worst condition from 0.376 to
   **0.687**, held out on a shard the gate was not tuned on, with no measurable
   cost on clean images.

The honest one-line state: **the system now works on clean and mildly degraded
images and is held up under heavy degradation by a fallback, not by
localisation.** Nothing here has been tested outside SID-Set at scale (CIFAKE
is in `CIFAKE_RESULTS.md`, and localisation degenerates there — see
`TEST_LOG.md` run 8).

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
  Anything reading `confidence` today is reading a constant. (**Revised by §9:**
  the *reported* value is a constant, but the pre-cap value it is clamped from
  is not, and gating on that pre-cap value is what §10 turns into the largest
  robustness gain in this document. The cap was hiding a working signal, not
  standing in for a missing one.)
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

## 7. Does the trained patch scorer's localisation gain survive degradation?

§1 and its held-out repeat (`eval_results/heldout_tables_*.md`) show the
patch-trained backend amplifies the localisation thesis on clean data — mean
IoU **0.433–0.509** against the base backend's **0.049–0.074**, and a bigger
AUC(tampered) gain from turning localisation on (+0.211 / +0.206 vs +0.079 /
+0.085). That is the correct comparison for the thesis (same backend, same
images, localisation on vs off) but it had only ever been run on clean data.
`eval/robustness.py` originally reported only the fused (with-localisation)
score per condition; it now also records `whole_image_score` — the same
whole-image number `fuse()` always computes internally as one of its two
inputs, so no extra backend calls — which reproduces §1's with/without pairing
per degradation condition instead of once on clean data. Run on shard 3,
untouched by any other measurement here, both backends paired on the same 45
images (15/class) at 768px, uncalibrated:

```bash
python eval/fetch_sid_set.py --split validation --shard 3 --per_class 15 --out <shard3>
python eval/robustness.py --data_dir <shard3> --backend hf:Organika/sdxl-detector \
    --limit 15 --max_side 768 --out eval_results/robustness_base.json
python eval/robustness.py --data_dir <shard3> --backend checkpoints/patch_scorer \
    --limit 15 --max_side 768 --out eval_results/robustness_patch_scorer.json
```

### 7a. The headline comparison: plain baseline vs the full upgrade

The question that matters to a user of the system: does swapping in the
trained patch scorer *and* turning localisation on (the shipped configuration)
beat the original plain whole-image baseline (no localisation, the public
`Organika/sdxl-detector`, scored once)?

| Condition | baseline: base backend, no localisation | shipped: patch scorer + localisation | delta |
|---|---|---|---|
| clean | 0.467 | **0.756** | +0.289 |
| jpeg q90 | 0.436 | **0.867** | +0.431 |
| jpeg q60 | 0.311 | 0.342 | +0.031 |
| jpeg q30 | 0.418 | 0.422 | +0.004 |
| blur σ0.5 | 0.453 | **0.778** | +0.324 |
| blur σ1.0 | 0.449 | 0.489 | +0.040 |
| blur σ2.0 | 0.440 | **0.627** | +0.187 |
| downscale ×0.5 | 0.449 | 0.480 | +0.031 |
| downscale ×0.25 | 0.436 | 0.342 | **−0.093** |
| noise σ0.02 | 0.467 | **0.853** | +0.387 |
| noise σ0.05 | 0.462 | **0.729** | +0.267 |
| crop 0.9 | 0.444 | **0.764** | +0.320 |
| crop 0.8 | 0.444 | **0.762** | +0.318 |
| saturate ×1.5 | 0.449 | **0.836** | +0.387 |

The shipped configuration beats the plain baseline at **13 of 14 conditions**,
often by a wide margin. **downscale ×0.25 is the one loss** — the only
condition anywhere in this document where the fully upgraded system is worse
than the original whole-image baseline it replaced.

### 7b. Isolating the localisation effect itself, per backend and per condition

The table above bundles two changes together (backend *and* localisation).
Unbundling them — with vs without localisation, same backend, same images,
same condition — is what actually tests the thesis:

**Base backend** (`AUC(tampered)`, with localisation → without):

| Condition | with | without | delta |
|---|---|---|---|
| clean | 0.444 | 0.467 | −0.022 |
| jpeg q90 | 0.453 | 0.436 | +0.018 |
| jpeg q60 | 0.289 | 0.311 | −0.022 |
| jpeg q30 | 0.413 | 0.418 | −0.004 |
| blur σ0.5 | 0.462 | 0.453 | +0.009 |
| blur σ1.0 | 0.449 | 0.449 | +0.000 |
| **blur σ2.0** | **0.693** | 0.440 | **+0.253** |
| downscale ×0.5 | 0.400 | 0.449 | −0.049 |
| **downscale ×0.25** | **0.604** | 0.436 | **+0.169** |
| noise σ0.02 | 0.422 | 0.467 | −0.044 |
| noise σ0.05 | 0.462 | 0.462 | +0.000 |
| crop 0.9 | 0.484 | 0.444 | +0.040 |
| crop 0.8 | 0.516 | 0.444 | +0.071 |
| saturate ×1.5 | 0.347 | 0.449 | −0.102 |

**Patch scorer** (`AUC(tampered)`, with localisation → without):

| Condition | with | without | delta |
|---|---|---|---|
| clean | 0.756 | 0.711 | +0.044 |
| jpeg q90 | 0.867 | 0.716 | +0.151 |
| **jpeg q60** | 0.342 | **0.564** | **−0.222** |
| **jpeg q30** | 0.422 | **0.609** | **−0.187** |
| blur σ0.5 | 0.778 | 0.711 | +0.067 |
| **blur σ1.0** | 0.489 | **0.684** | **−0.196** |
| blur σ2.0 | 0.627 | 0.667 | −0.040 |
| **downscale ×0.5** | 0.480 | **0.702** | **−0.222** |
| **downscale ×0.25** | 0.342 | **0.667** | **−0.324** |
| noise σ0.02 | 0.853 | 0.711 | +0.142 |
| noise σ0.05 | 0.729 | 0.729 | +0.000 |
| crop 0.9 | 0.764 | 0.733 | +0.031 |
| crop 0.8 | 0.762 | 0.744 | +0.018 |
| saturate ×1.5 | 0.836 | 0.724 | +0.111 |

(Full per-condition detail, including AUC(all), IoU, confidence and verdict
stability, is in `eval_results/robustness_base.json` /
`robustness_patch_scorer.json`.)

Two things follow from unbundling it this way, and both revise the earlier
version of this section, which inferred the localisation effect from an IoU
proxy rather than measuring it directly per condition:

**For the base backend**, localisation is mostly noise at this sample size
(deltas within ±0.05, matching §6's ±0.09 standard error) except two large,
genuine wins — **blur σ2.0 (+0.253) and downscale ×0.25 (+0.169)** — where the
region map recovers detection the whole-image score had lost.

**For the patch scorer**, the effect is far larger in both directions, and it
is *net negative* at five conditions, not the three this section originally
reported: **jpeg q60 (−0.222), jpeg q30 (−0.187), blur σ1.0 (−0.196),
downscale ×0.5 (−0.222) and downscale ×0.25 (−0.324)** — the last of these
large enough that it is also why the patch scorer loses to the plain baseline
in §7a. In every one of these five, `eval_results/robustness_patch_scorer.json`
shows recall and IoU at or above their clean-condition level (recall peaks at
**0.939** at jpeg q30) — the map keeps finding the right pixels; fusing that
finding into the score makes the decision *worse* than trusting the whole-image
number alone. This is the same "excellent within-image ranking, wrong absolute
level" failure already recorded in the patch-scorer training run above
(per-pixel map AUC 0.460→0.955, partially-AI subset AUC 0.648→**0.604**) —
independent confirmation that it recurs specifically under heavy JPEG,
moderate blur and downscaling, not an artefact of that one run.

Where the patch scorer's localisation *does* help, it helps by more than the
base backend ever does (up to +0.151 at jpeg q90, +0.142 at noise σ0.02) — the
training clearly sharpened the map. It just did not sharpen it *uniformly*:
the same training that produced a much more confident, much more accurate map
on clean and mildly-degraded images produced one that actively misleads fusion
on exactly the transforms (heavy JPEG, moderate-to-heavy downscale, moderate
blur) that most directly attack patch-level statistics.

At **noise σ0.05**, both backends show delta = 0.000 exactly — regions per
image collapse to near zero for both (0.04 and 0.09, from §6/§7's raw JSON),
so there is no local evidence for fusion to add or subtract. Whatever
detection the patch scorer has left there is the classifier itself, not
localisation.

**Caveat:** n=15/class, the same ±0.09 AUC standard error as §6. Several of
the individually large within-backend deltas above (base backend's blur σ2.0
and downscale ×0.25 especially) are plausible but not independently
significance-tested at this sample size — treat the *pattern* (patch-scorer
localisation is unreliable under JPEG/blur/downscale specifically) as the
finding, not the exact magnitude of any one row. Confirming it needs the same
fix as everywhere else in this document: more images per condition.

---

## 8. The same comparison on a fresh shard, at 3× the sample

§6–§7 are 15 images per class, with an AUC standard error near ±0.09 — enough to
establish a pattern, not a number. This repeats the identical protocol on SID-Set
validation **shard 4** (untouched by any training, calibration or decision
fitting) at **50 per class**, both backends, 14 conditions, 768px, uncalibrated.
Full tables: `SID_SET_RESULTS.md`. Job 779368, `TEST_LOG.md` run 9.

| | base backend, no localisation | patch scorer + localisation | delta |
|---|---|---|---|
| AUC(all AI), clean | 0.759 | **0.919** | +0.160 |
| AUC(tampered), clean | 0.587 | **0.854** | +0.267 |
| AUC(tampered), jpeg q30 | 0.555 | **0.367** | **−0.188** |
| AUC(tampered), downscale ×0.25 | 0.544 | **0.411** | **−0.134** |
| localisation IoU, clean | 0.041 | **0.416** | +0.375 |

**The pattern from §7 reproduces at the larger sample, on different images, and
the two halves of it get sharper.** The shipped configuration is far better on
clean and mildly degraded images (+0.267 AUC(tampered) clean, IoU 10× the base
backend's) and *worse than a plain whole-image pass* under heavy JPEG and
downscaling. Isolating localisation alone on this shard, with the patch scorer:
**+0.279 on clean** and **−0.238 at jpeg q30, −0.158 at downscale ×0.25**
(AUC(tampered), with-localisation minus without).

Two things this larger run settles that §7 could not:

- **For the base backend, localisation is a small net negative**, not noise:
  AUC(all AI) with-minus-without is negative in **13 of 14 conditions**, ranging
  −0.091 to +0.017. §7's ±0.09 interval could not resolve that sign; 50/class
  can. The region machinery only pays for itself once the backend can actually
  score a patch.
- **The failure is not localisation being weak, it is localisation being
  confidently wrong.** At jpeg q30 the patch scorer's map still has IoU 0.372
  and recall 0.887 — near its clean-condition quality — while the fused decision
  falls to 0.367. The map finds the right pixels and the score built on them is
  worse than chance.

## 9. Why it breaks: the map's score distribution moves, and it moves on real images

§8 leaves an obvious question — is the map mis-*cut* or mis-*informed*? Region
proposal uses absolute thresholds (`threshold_lo/hi` = 0.45/0.75), so a
degradation that shifts the whole map up or down changes how much of the frame
fires without the content changing at all. Job 779811 instrumented the harness to
answer it: per-class map statistics, the map's median, and the pre-cap
confidence are now recorded per image. Full tables: `THRESHOLD_RESULTS.md`,
`TEST_LOG.md` run 10.

**The distribution does move, and it moves asymmetrically.** Mean map median,
patch scorer, shard 4:

| Condition | on **real** images | on tampered images | real images that grew a region |
|---|---|---|---|
| clean | 0.022 | 0.081 | 14% |
| jpeg q60 | 0.120 | 0.081 | 56% |
| **jpeg q30** | **0.382** | 0.380 | **96%** |
| blur σ1.0 | 0.133 | 0.086 | 70% |
| **blur σ2.0** | **0.683** | 0.287 | **98%** |
| downscale ×0.25 | 0.228 | 0.051 | 60% |
| **noise σ0.05** | **0.008** | 0.011 | **0%** |

Read the first two columns together. Under compression and blur the map does not
lift uniformly — **the real images rise to meet the tampered ones**, and by
jpeg q60 they have already overtaken them. This is the 64px-crop failure of §4 again, one level
up: a compressed or blurred real photograph looks, patch by patch, exactly like
what a patch-scale detector was trained to call generated. Under noise the
opposite happens — everything collapses to zero and the map switches off (0.09
regions per image, IoU 0.005), which is why noise is the one severe condition
where fusing costs nothing: there is nothing to fuse.

### Re-placing the cuts does not fix it

Two shift-invariant threshold modes were added to `mapper/heatmap.py` and run on
the identical shard-4 frames (`--threshold_mode quantile | median_shift`):

| Mode | mean AUC(tampered) | worst condition | mean loc IoU | real images firing, clean |
|---|---|---|---|---|
| absolute (shipped) | 0.693 | 0.368 | 0.306 | 14% |
| quantile | 0.693 | 0.372 | 0.218 | 38% |
| median_shift | **0.708** | 0.378 | **0.346** | 42% |

**Neither rescues detection.** `median_shift` is the better *localiser* — higher
IoU in 11 of 14 conditions, and it is what recovers blur σ2.0 (0.500 → 0.668) —
but it buys that by firing on three times as many authentic photographs on clean
data, and the worst condition barely moves (0.368 → 0.378). `quantile` is worse
on both counts: cutting at a map's own upper tail manufactures a tail on a map
that has none.

**Both halves of that replicate held-out.** Job 780095 reran the same two modes
on shard 3 at 50/class: `median_shift` again wins IoU in **11 of 14** conditions
(mean 0.264 → 0.318) and again pays for it on authentic photographs (8% → 28%
firing on clean data), with mean AUC(tampered) 0.692 → 0.703 and the worst
condition slightly *worse*, 0.376 → 0.333. The shipped default stays `absolute`;
`median_shift` is the right choice only for a deployment that wants the map
itself and can afford the false positives.

The conclusion is the same shape as §4's, and worth as much: **at these
severities the per-patch signal is gone, not mis-cut.** The whole-image pass on
the same jpeg q30 images still separates real from tampered at 0.605 while every
fused variant sits at 0.37–0.45. No threshold placement recovers information the
patches no longer carry.

### The confidence signal was never flat — the cap was hiding it

§6 recorded confidence as a constant 0.60 in every condition and called the
signal inert. That was the *reported* number. `fusion.py` now also records
`confidence_uncapped`, the value before `UNCALIBRATED_CONFIDENCE_CAP` clamps it,
and it is not flat: mean 0.875 clean, 0.813 at jpeg q60, **0.784 at jpeg q30**,
with the cap binding on **98%** of images. The mechanism designed to say "I can
no longer tell" works; it was being flattened on the way out. That is what makes
§10 possible.

## 10. Routing, not robustness: decide per image whether to believe the map

§8 and §9 together say localisation is both the entire gain and the entire
fragility, and that it cannot be made robust by rescaling. So the question
changes from *how do we make the map survive degradation* to *can the system
tell, per image, when not to listen to it* — and fall back to the whole-image
score it already computes.

It can. Ten candidate signals were searched by leave-one-condition-out on shard
4; the winner is the pre-cap confidence of §9, with a cut at **0.8577**. Below
it, the score comes from the whole-image pass and the regions are not reported
(`dual_backend.py`, `eval/validate_gate.py`).

### Held out on a shard the gate never saw

The gate was tuned on shard 4, so shard 4 cannot score it. Job 780095 reran the
patch scorer on **shard 3** at 50/class with the threshold **frozen as a
literal** — nothing refitted — and replayed the rule offline. `AUC(real vs
tampered)`, 14 conditions:

| | always fuse | always whole-image | **gate** |
|---|---|---|---|
| mean over 14 conditions | 0.692 | 0.665 | **0.803** |
| worst condition | 0.376 | 0.528 | **0.687** |
| conditions below 0.5 | 3 | 0 | **0** |
| clean | 0.847 | 0.665 | 0.856 |
| jpeg q30 | 0.376 | 0.598 | **0.775** |

Shard 4's own numbers were mean 0.693 → 0.806 and worst 0.368 → 0.666. **The
held-out shard reproduces both to within 0.02**, and the tuning optimism is
measurable and tiny: shard 3's own best threshold (0.8744) would have scored
0.807 against the frozen threshold's 0.803.

On AUC(real vs all AI) the same rule gives mean 0.809 → **0.890**, worst 0.473 →
**0.836**, and it moves the false-positive rate the whole project has been stuck
on: at 80% recall on AI, **0.080 → 0.040 on clean images** (the base backend's
was 0.44).

Paired bootstrap on shard 3, gate minus always-fuse:

| Condition | AUC(tampered) delta | 95% CI |
|---|---|---|
| clean | +0.009 | [−0.063, +0.080] |
| jpeg q30 | **+0.400** | [+0.224, +0.567] |

**That is the shape a fix should have.** It is a wash where the map was already
right and a large, significant recovery exactly where the map was lying. The
gate is not adding detection power; it is declining to destroy it.

### The class, not the replay — and what the gate costs

Everything above is an offline replay of the rule over per-image JSON. Job 780205
ran `DualBackendAnalyzer` itself over shard 3, 50/class, threshold frozen, and
reproduces the replay to within 0.001 on every summary statistic: mean
AUC(tampered) **0.803**, worst **0.687**, mean AUC(all AI) **0.890**, worst
0.837. The gate declines to trust localisation on
**36%** of images on average — 23% on clean, 75% at jpeg q30.

That is the cost, and it is the honest way to state the result: **reported**
localisation IoU falls from 0.264 to **0.181**, because a distrusted image
contributes an empty mask. The gate buys its detection robustness by giving up
localisation on the images it does not believe — which is the correct trade only
if the score matters more than the map. `dual_backend.py` drops the findings
rather than reporting them beside a score that ignored them, and the harness
scores localisation on what was reported, so this cost cannot be hidden by
measuring the map the pipeline privately computed.

### Three things the gate gets wrong if built as a bare `if`

1. **The two scores are not on the same scale.** Substituting one for the other
   shifts the ranking for reasons unrelated to the image. Rank-normalising both
   arms within a condition first — the conservative reading — cuts the gain from
   +0.111 to **+0.034** mean on shard 3 (worst-condition +0.311 → +0.185). Both
   numbers are reported by `eval/validate_gate.py`; the raw one is what a
   deployment doing exactly this substitution gets, the rank one is what the gate
   is worth net of scale effects. `ScoreAligner` exists to remove that gap when
   the fallback is a *separate* model.
2. **Falling back means the regions are not trusted either.** When the gate
   fires, `dual_backend.py` drops the findings and says why in a note — reporting
   a fallback score beside the regions it just overruled invites belief in both.
   That is what the IoU cost above is measuring.
3. **The threshold is not a universal constant.** It was tuned against the patch
   scorer under absolute cuts. Carried unchanged onto `median_shift` cuts it made
   results worse than either arm alone, so it lives in config, per deployment.

### The fallback should be the primary's own global view, not a second model

Mean AUC(tampered) over 14 shard-4 conditions: primary alone 0.693, **self-
fallback 0.806**, a separate `Organika/sdxl-detector` as fallback 0.745 (0.728
with the fitted quantile alignment). Job 780205 ran both arms for real on shard
3 and reproduces it: **self 0.803 against separate 0.714**, worst condition
0.687 against 0.581, self winning **13 of 14** conditions. Both arms distrust
the same 36% of images and report the same localisation — the gate is identical;
only the substituted number differs.

The useful split is therefore between two *pathways* — local evidence versus
global — not between two detectors, and the fine-tuned backend's own whole-image
pass is the better global view. It is also free: `analyze.py` already computes
that score and hands it to `fuse()`, so the default `fallback.backend: "self"`
costs no extra forward pass. `ScoreAligner` stays for the case where someone does
want a separate model; it is not what closes this gap (alignment made shard 4's
separate arm slightly *worse*, 0.745 → 0.728).

---

## What to do next, in order of expected value

The previous list's item 1 (train a patch-scale detector) is done —
`checkpoints/patch_scorer`, `TEST_LOG.md` runs 3/5b. Item 2 (make `confidence`
mean something) is half done: the signal turns out to exist and to be useful
(§9, §10), but what the pipeline *reports* is still a constant 0.60, so anything
downstream reading `confidence` is still reading nothing. What follows replaces
the rest.

1. **Decide whether the gate ships, and measure it end to end.** §10 validates
   the *rule* on held-out data, and `TEST_LOG.md` run 12 exercises the class on
   real images, but `configs/default.yaml` still runs the ungated pipeline and
   the server does not import `dual_backend.py`. Wiring it in means choosing the
   threshold per deployment (it is not a constant — §10) and re-running §1–§3's
   detection/localisation/false-positive tables through the gated path, since
   every number in them predates it.
2. **The decision rule is still not the one the evidence recommends.** The
   `max` map reduction scored AUC 0.980 / 0.979 on two shards
   (`TEST_LOG.md` runs 5b, 6) against the shipped `region_evidence` design's
   ~0.82 through `fuse()`. That gap has been open since 2026-08-31 03:02 and is
   independent of everything in §8–§10 — the gate decides *whether* to use the
   map, not how to reduce it.
3. **Fix verdict instability directly.** It survived every change here: at noise
   σ0.05 only 30–44% of images keep their clean verdict while AUC is unchanged,
   and the gate does not help because it substitutes a *score*, not a label.
   Hysteresis, or deriving the label from the same margin the score is built on.
4. **Push past a single 50/class shard.** Every §8–§10 number has ~±0.05 on it.
   The gate's held-out reproduction (0.803 vs 0.806) is well inside that, which
   is reassuring, but the per-condition rows are not individually significant.
5. **Attack the patch-level fragility at its source** rather than routing around
   it: §9 shows degraded *real* patches are what the patch scorer misreads, so
   training with degradation-augmented hard negatives is the direct fix, and the
   one that would let localisation be trusted under compression instead of
   switched off.
6. **Re-run with the verdict fix** to get honest `ai_edited` vs `ai_generated`
   numbers (§5), and **compare backends** (§ EVALUATION.md) — every number here
   describes one feature family.

## Caveats that bound every number here

- **SID-Set's test split is gated**, so this is the validation split. Its real
  images come from OpenImages, which appears in many training sets, so
  **detection AUC is an upper bound**. The localisation numbers are the
  trustworthy ones — no whole-image detector could have memorised a mask.
- **n = 120 per class** in §1–§5, **15** in §6–§7, **50** in §8–§10. AUC standard
  error is roughly ±0.03, ±0.09 and ±0.05 respectively; differences smaller than
  that are noise, which is why the ablation and §10 report bootstrap intervals
  rather than bare deltas.
- One backend family, one dataset, one working resolution. §8–§10 all run at
  768px on SID-Set validation; the shipped default is 1024px, and CIFAKE
  (`CIFAKE_RESULTS.md`) is the only other dataset tried — where the window
  mapper degenerates entirely on 32px images.
- **The gate's threshold (0.8577) is fitted, not derived.** It was tuned on
  shard 4 and validated frozen on shard 3; it is specific to this checkpoint,
  these absolute cuts and this working resolution.
