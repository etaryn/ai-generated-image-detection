# Test log — model_03

Every test model_03 has been through, in the order it happened: what was run,
against what data, with what parameters, what came back, and the exact
command/script/job that produced it. `EVALUATION.md` and
`EVALUATION_RESULTS.md` are the narrative write-ups of a subset of this; this
document is the complete chronological record, including the runs that don't
appear in either (the training runs, the decision-rule sweep, the held-out
check, the cross-dataset comparisons, and the threshold/gate diagnostics).

**Convention carried over from `EVALUATION.md`: no number below should exist
that can't be regenerated.** Every entry gives the reproduction command. Job
IDs are this cluster's Slurm IDs (`sacct -j <id>`), logs are
`slurm_logs/<name>_<id>.{out,err}`, raw per-image JSON is in
`eval_results/` (gitignored — regenerable, not redistributed).

| # | Date | Job | Script | Dataset | What it tested |
|---|---|---|---|---|---|
| 1 | predates job history below | — (ad hoc) | `scripts/check_backend.py` | CIFAKE, 60/class | Does each candidate backend even point the right way? |
| 2 | predates job history below | — (ad hoc) | `eval/evaluate.py` + `eval/ablation.py` | SID-Set val shard 0, 120/class | Per-scale calibration: does it help the shipped pipeline? |
| 3 | 2026-08-30 23:43–00:03 | 774937 `aigc-m03-patch` | `scripts/train_patch_scorer.py` | SID-Set train shards 0–5, 100/class | First patch-scorer fine-tune (1,800 train images) |
| 4 | 2026-08-31 00:29–01:02 | 775125 `aigc-m03-eval` | `eval/evaluate.py` + `eval/ablation.py` | SID-Set val shard 0, 120/class | Paired ablation of the 774937 checkpoint vs base backend |
| 5a | 2026-08-31 01:44–01:55 | 776852 `aigc-m03-scaled` (CANCELLED) | `scripts/train_patch_scorer.py` | SID-Set train, 40 shards × 250/class | Scaled fine-tune — killed, oversized for the time budget |
| 5b | 2026-08-31 01:55–03:02 | 776896 `aigc-m03-scaled` | `scripts/train_patch_scorer.py` + `scripts/fit_decision.py` | SID-Set train 10 shards × 150/class; decision sweep on val shard 0, 120/class | Right-sized scaled fine-tune (current checkpoint) + decision-rule sweep |
| 6 | 2026-08-31 03:08–03:50 | 777091 `aigc-m03-heldout` | `eval/evaluate.py` + `eval/ablation.py` + `scripts/fit_decision.py` | SID-Set val shard 2, 120/class | Does everything above generalise to data untouched by training or decision-fitting? |
| 7 | 2026-08-31 11:58–15:17 (×3) | 778211 / 778685 / 778939 `aigc-m03-robust` | `eval/robustness.py` | SID-Set val shard 3, 15/class | Transform × severity matrix, base vs patch scorer, paired |
| 8 | 2026-08-31 16:21–16:27 | 779287 `aigc-m03-cifake` | `eval/fetch_cifake.py` + `eval/robustness.py` + `eval/report_cifake.py` | CIFAKE, 100/class | Same robustness comparison, on a real/synthetic-only dataset with no partially-AI case |
| 9 | 2026-08-31 16:43–18:14 | 779368 `aigc-m03-sidset` | `eval/fetch_sid_set.py` + `eval/robustness.py` + `eval/report_robustness.py` | SID-Set val shard 4, 50/class | Same robustness comparison, fresh SID-Set shard, bigger sample |
| 10 | 2026-08-31 18:40–20:17 | 779811 `aigc-m03-diag` | `eval/robustness.py` (instrumented) ×4 + `eval/report_thresholds.py` | SID-Set val shard 4, 50/class | Why does fusion invert under degradation? Absolute vs quantile vs median_shift cuts, and can trust be routed per image? |
| 11 | 2026-08-31 20:38–21:26 | 780095 `aigc-m03-val3` | `eval/robustness.py` ×2 + `eval/validate_gate.py` | SID-Set val shard 3, 50/class | Held-out test of run 10's two findings, constants frozen |
| 12 | 2026-08-31 21:03–21:52 | 780205 `aigc-m03-dual` | `pytest tests/` + `eval/robustness.py --dual` ×2 | SID-Set val shard 3, 50/class | First real-backend run of `DualBackendAnalyzer`, and the first test run over the modified pipeline |

Rows 1–7 correspond to material already narrated in `EVALUATION.md` /
`EVALUATION_RESULTS.md`; the tables below reproduce their numbers alongside
the run parameters those documents don't spell out (job ID, exact config,
image counts by class). Rows 9–12 are written up in `EVALUATION_RESULTS.md`
§8–§10; row 8 (CIFAKE) has `CIFAKE_RESULTS.md` and this log only.

---

## 1. Backend sanity check — which detector even points the right way

**Question:** of the candidate backends, which correctly separates AI from
real, at all, before anything else about them is trusted?

**Parameters:** `scripts/check_backend.py --data_dir <cifake real/fake> --backend hf`, 60 images/class, no mapper/windowing involved — this scores whole images directly through each backend's own classifier head.

**Dataset:** CIFAKE (`data/raw/cifake/{real,fake}`), native 32×32.

**Result:**

| Backend | mean score, fake | mean score, real | AUC | Verdict |
|---|---|---|---|---|
| `Organika/sdxl-detector` | 0.342 | 0.131 | 0.696 | CORRECT |
| `dima806/ai_vs_real_image_detection` | 0.991 | 0.025 | 1.000 | CORRECT |
| `Ateeqq/ai-vs-human-image-detector` | 0.581 | 0.764 | 0.454 | NON-DISCRIMINATIVE |

**Caveats (from `EVALUATION.md`):** a demonstration of the check, not a
backend ranking — these are 224px models scoring 32px thumbnails. `dima806`'s
perfect score most likely means CIFAKE leaked into its own training set.
`Ateeqq` saturates on tiny thumbnails and may be fine on real photos; the
labels-by-name resolution this check exists to verify matters because 5 of 6
surveyed backends put "AI" at logit index 0 and one (`dima806`) puts it at
index 1 — a hard-coded index would silently invert that one.

**Source:** `server/model_03/EVALUATION.md`, "The backend's labels must be
resolved by name, never by index." No Slurm job — this predates the job
history in this log (likely run interactively; not GPU-bound at this scale).

---

## 2. Per-scale calibration — does it help the shipped pipeline?

**Question:** the raw map over-flags real photos at the fine scale (see
table below) — does fitting a per-scale isotonic calibrator on that
distortion improve the pipeline's actual output?

**Parameters:** backend `hf:Organika/sdxl-detector`, mapper scales `[64, 128,
224]`, 1024px working resolution. Calibrators fit with
`scripts/calibrate_mapper.py --manifest <shard 1 sample>/manifest.json --backend hf --method isotonic --out configs/calibration_sdxl_detector.json`, evaluated with
`eval/evaluate.py --data_dir eval_data/sid_set_val --out eval_results/sid_set_calibrated.json` and again with `--config configs/uncalibrated.yaml --out eval_results/sid_set_uncalibrated.json`, compared by `eval/ablation.py`.

**Dataset:** SID-Set validation, shard 0, 120/class = 360 images (real /
synthetic / tampered), calibrators fit on a disjoint shard 1 sample (80/class)
— never fit and scored on the same shard.

**Result — fine scale is a false-positive engine, and calibration fixes ECE
but not the pipeline:**

| Patch scale | frac. of patches >0.75 on **authentic** photos | mean score on synthetic images |
|---|---|---|
| 64px | 36.6% | 0.708 |
| 128px | 16.3% | 0.709 |
| 224px | 10.4% | 0.702 |

| Scale | held-out ECE, before | after isotonic |
|---|---|---|
| 64px | 0.234 | **0.089** |
| 128px | 0.221 | **0.080** |
| 224px | 0.191 | **0.071** |

| Metric | uncalibrated | calibrated |
|---|---|---|
| AUC real vs tampered | **0.589** | 0.539 |
| Localisation mean IoU | **0.074** | 0.039 |
| Localisation touch rate | **0.533** | 0.225 |
| Real images with any region | 0.825 | **0.483** |

**Conclusion (why this matters for every later run):** calibration improved
held-out ECE substantially at every scale but made every headline pipeline
metric *worse* — it discounted the fine scale, which was carrying the false
positives **and** the only real localisation signal, inseparably. **This is
why every run after this one uses `calibration_path: null`** (now the shipped
default in `configs/default.yaml`) — Platt scaling was tried first and made
ECE worse still (0.221→0.273 at 128px), which is why isotonic was adopted
before this finding shelved calibration entirely.

**Source:** `server/model_03/EVALUATION.md` ("The fine scale was a
false-positive engine…", "Isotonic, not Platt") and
`server/model_03/EVALUATION_RESULTS.md` §1–§4 ("Run: 2026-08-30 · 360 images
from SID-Set validation shard 0"). No Slurm job in the current history (`sacct`
has nothing before 774937 on 2026-08-30 23:43) — predates this log's job
tracking; reproduce with the commands above.

---

## 3. Patch-scorer training run #1 — first fine-tune, 1,800 images

**Question:** does fine-tuning the backend on labeled *patches* (not whole
images) fix the "backend was trained on whole images, asked about 64px crops"
mismatch that §1–2 traces every failure back to?

**Parameters:** `scripts/train_patch_scorer.py --data <train sample> --out
checkpoints/patch_scorer --epochs 3 --per_image 4 --batch_size 32` (no
`--hard_negatives` yet). Base model `Organika/sdxl-detector`, scales `[64,
128, 224]`, positive/negative patch fractions 0.7/0.05.

**Dataset:** SID-Set **train** split, 6 shards × 100/class ≈ 1,800 images
(`eval/fetch_sid_set.py --split train --shard 0 --shards 6 --per_class 100`).

**Result — the training script's own per-epoch validation split:**

| Metric | base | patch-trained (this run) |
|---|---|---|
| **AUC(real vs AI)** | 0.771 | **0.802** |
| ├ fully-AI subset | 0.893 | **1.000** |
| └ **partially-AI subset** | 0.648 | **0.604** ⚠ |
| mean score, real images | 0.539 | 0.470 |
| Per-pixel map AUC (diagnostic) | 0.460 | 0.955 |

**Conclusion:** training fixed the easy half (fully-synthetic) and *lost*
ground on the hard half (partially-AI) — excellent within-image pixel ranking
(map AUC 0.460→0.955), wrong absolute level. This is the finding that added
`--hard_negatives` (re-shown the real-image patches the model scores
highest) to the training script for the next run.

**Source:** `server/model_03/EVALUATION.md`, "Training a patch scorer (the
server job)". Job **774937** `aigc-m03-patch`, 2026-08-30 23:43:57 →
00:03:22, 19m25s, `slurm_logs/model03_774937.{out,err}`. This checkpoint was
later overwritten by run 5b below — its own `training.json` no longer exists
in the repo (a copy was kept at `$SCRATCH/patch_scorer_prev` on the cluster,
not in git).

---

## 4. Paired ablation — the same checkpoint through the full pipeline

**Question:** run 3 measured the training script's own validation split (raw
patch/map metrics). Does the gain survive being routed through the actual
shipped pipeline — regions, routing, fusion, verdict — rather than measured
as a bare patch classifier?

**Parameters:** `eval/evaluate.py --data_dir <val shard 0> --backend
checkpoints/patch_scorer --out eval_results/patch_scorer.json`, and the same
against `--backend hf:Organika/sdxl-detector --out
eval_results/base_rerun.json`, both on `configs/default.yaml`
(`calibration_path: null`), then `eval/ablation.py` on each.

**Dataset:** SID-Set validation shard 0, 120/class = 360 images — the same
sample runs 2/5b/6 use, so results are comparable across all of them.

**Result — base backend (`eval_results/tables_base.md`):**

| Positive class | without (whole-image) | with (region-aware) | delta | 95% CI | significant |
|---|---|---|---|---|---|
| all AI | 0.721 | 0.738 | +0.017 | [−0.023, +0.058] | no |
| **tampered** | 0.508 | 0.587 | +0.079 | [+0.012, +0.149] | yes |
| synthetic | 0.935 | 0.889 | −0.045 | [−0.072, −0.021] | yes |

Localisation (120 masked images): mean IoU 0.074, median 0.021, recall 0.191,
touch rate 0.542. FPR on real at 80% recall on AI: 0.617 without → 0.508 with.

**Result — patch-scorer checkpoint (`eval_results/tables_patch_scorer.md`):**

| Positive class | without (whole-image) | with (region-aware) | delta | 95% CI | significant |
|---|---|---|---|---|---|
| all AI | 0.718 | 0.820 | +0.102 | [+0.051, +0.153] | yes |
| **tampered** | 0.439 | 0.650 | +0.211 | [+0.118, +0.308] | yes |
| synthetic | 0.997 | 0.989 | −0.008 | [−0.017, −0.002] | yes |

Localisation: mean IoU 0.433, median 0.433, recall 0.851, touch rate 0.983.
FPR on real at 80% recall on AI: 0.650 without → 0.433 with.

**Conclusion:** the trained checkpoint's gain is real through the whole
pipeline, not an artifact of the training script's own metric — tampered AUC
+0.211 vs the base backend's +0.079, mean IoU 0.433 vs 0.074. Both are
directionally validated, not deployable (43–51% of real photos still flagged
at 80% AI recall).

**Source:** `server/model_03/EVALUATION.md` refers to this indirectly; the
raw tables are `eval_results/tables_base.md` and
`eval_results/tables_patch_scorer.md`. Job **775125** `aigc-m03-eval`,
2026-08-31 00:29:19 → 01:02:37, 33m18s,
`slurm_logs/model03_eval_775125.{out,err}`.

---

## 5a. Scaled training run — attempt (cancelled)

**Parameters:** `scripts/train_patch_scorer.py --epochs 3 --batch_size 32
--hard_negatives 512` on a fetch of **40 shards × 250/class** SID-Set train
(≈30k images) — an order of magnitude past what runs 3/5b used.

**Outcome:** cancelled by user 10m42s in, mid-fetch (row group logs show only
shard 0/40 done). Too large for the 3-hour `gpu`-partition time cap once
fetch + 3 epochs over ~30k images were accounted for; superseded immediately
by 5b at a size that fits the budget.

**Source:** Job **776852** `aigc-m03-scaled`, 2026-08-31 01:44:21 → 01:55:03,
`slurm_logs/model03_scaled_776852.{out,err}` (ends "CANCELLED AT … DUE to
SIGNAL Terminated").

## 5b. Scaled training run — right-sized, current checkpoint

**Question:** does more training data (and hard negatives) move the
partially-AI regression run 3 found?

**Parameters:** `scripts/train_patch_scorer.py --data <train sample> --out
checkpoints/patch_scorer --epochs 3 --batch_size 32 --hard_negatives 512`.
Previous checkpoint backed up to `$SCRATCH/patch_scorer_prev` first (not in
git — the 1,800-image run's own numbers survive only as the table in run 3
above).

**Dataset:** SID-Set train, 10 shards × 150/class (`--split train --shard 0
--shards 10 --per_class 150`); decision sweep scored against SID-Set
validation shard 0, 120/class, `--limit 50`.

**Result — `checkpoints/patch_scorer/training.json` (the checkpoint every
later run in this log, except run 3, uses):**

| Field | Value |
|---|---|
| base_model | `Organika/sdxl-detector` |
| scales | [64, 128, 224] |
| positive_frac / negative_frac | 0.7 / 0.05 |
| epochs | 3 |
| lr | 2e-5 |
| train_images | 4,080 |
| val_patches | 3,600 |
| patch_auc_before → after | 0.661 → **0.990** |
| flip | true |

**Result — `eval/fit_decision.py` sweep** (what's the best way to reduce the
per-pixel map to one real-vs-AI number? — a *different* question from run 4's
paired ablation: this bypasses `fuse()`'s calibration/verdict logic and
sweeps raw map reductions directly against the map cached once per image):

| Reduction | AUC(real vs AI) | AUC(partially-AI) | FPR @ 80% recall |
|---|---|---|---|
| whole-image only (no map) | 0.761 | 0.530 | 0.44 |
| **max** (best candidate) | **0.980** | **0.960** | **0.00** |
| p99 | 0.965 | 0.929 | 0.00 |
| mean | 0.917 | 0.834 | 0.18 |
| region_evidence (the shipped design, hi=0.7, area=0.002) | 0.975 | 0.951 | 0.00 |

Full sweep (29 candidates × threshold/area combinations) in
`eval_results/decision_sweep.json`.

**Read this against run 4 carefully — the numbers are not comparable at face
value.** This sweep's AUCs (~0.96–0.98) are far higher than the shipped
pipeline's paired ablation (run 4: 0.650 tampered, 0.820 all-AI) because it
scores raw map-reduction candidates directly, without `fuse()`'s
confidence/verdict/calibration layer in between — it answers "how much
signal is *in the map*", not "what does the shipped pipeline output". As of
this log, `configs/default.yaml` has **not** been changed to adopt the `max`
reduction this sweep recommends — `fuse()` still uses the original
region-evidence design. That gap (map ceiling ~0.98 vs. shipped output ~0.82)
is unresolved and worth someone's attention before the next round of
decision-rule work.

**Source:** `server/model_03/README.md`-adjacent script docstrings; no
narrative write-up in `EVALUATION.md`/`EVALUATION_RESULTS.md` yet. Job
**776896** `aigc-m03-scaled`, 2026-08-31 01:55:33 → 03:02:22, 1h06m49s,
`slurm_logs/model03_scaled_776896.{out,err}`. Raw sweep:
`eval_results/decision_sweep.json`.

---

## 6. Held-out generalisation check

**Question:** shard 0 was used to *fit* the decision sweep (5b) — does its
`max`-reduction result, and the training gain generally, hold on data the
checkpoint and the sweep have never seen at all?

**Parameters:** same as runs 4 and 5b's sweep, pointed at a shard neither
touched. `eval/evaluate.py` + `eval/ablation.py` (tuned vs base, paired) and
`scripts/fit_decision.py --limit 120` (no `--cache`, so it's a fresh compute,
not a re-read of shard 0's cache).

**Dataset:** SID-Set validation **shard 2** — chosen specifically because
shard 0 was used by the decision sweep and shard 1 is the conventional
calibration shard; 120/class = 360 images.

**Result — paired ablation, base backend (`heldout_tables_base.md`):**

| Positive class | without | with | delta | 95% CI | significant |
|---|---|---|---|---|---|
| all AI | 0.729 | 0.765 | +0.036 | [+0.004, +0.067] | yes |
| **tampered** | 0.559 | 0.644 | +0.085 | [+0.031, +0.140] | yes |
| synthetic | 0.899 | 0.886 | −0.013 | [−0.031, +0.005] | no |

Localisation: mean IoU 0.049, touch rate 0.467. Verdict: "detection improves
but localisation misses more often than it lands... do not present the maps
to users on this evidence" (base backend specifically).

**Result — paired ablation, patch scorer (`heldout_tables_patch_scorer.md`):**

| Positive class | without | with | delta | 95% CI | significant |
|---|---|---|---|---|---|
| all AI | 0.789 | 0.884 | +0.095 | [+0.051, +0.143] | yes |
| **tampered** | 0.588 | 0.794 | +0.206 | [+0.123, +0.294] | yes |
| synthetic | 0.990 | 0.974 | −0.016 | [−0.029, −0.007] | yes |

Localisation: mean IoU 0.509, median 0.567, touch rate 0.883 — *higher* than
run 4's shard-0 number (0.433), evidence the training gain isn't an artifact
of the fitting shard. Verdict recorded verbatim: "the idea works on this
evidence — it detects better AND localises well."

**Result — decision sweep, replayed on shard 2 (`heldout_decision_sweep.json`):**

| Reduction | AUC(real vs AI) | AUC(partially-AI) |
|---|---|---|
| whole-image only | 0.789 | 0.588 |
| **max** | **0.979** | **0.957** |

Matches shard 0's sweep (0.980 / 0.960) closely — the `max`-reduction
finding generalises, it wasn't fit-shard noise, even though (per run 5b) it
still isn't what the shipped `fuse()` implements.

**Source:** `eval_results/heldout_*.{json,md}`. Job **777091**
`aigc-m03-heldout`, 2026-08-31 03:08:39 → 03:50:57, 42m18s,
`slurm_logs/model03_heldout_777091.{out,err}`.

---

## 7. Robustness matrix — SID-Set, transform × severity

**Question:** does the with/without-localisation gain (runs 4/6) survive
JPEG recompression, blur, downscaling, noise, cropping, saturation — the
redistribution transforms a moderation pipeline actually meets?

**Parameters:** `eval/robustness.py --backend <base|checkpoints/patch_scorer>
--limit 15 --max_side 768`, 14 conditions (clean; jpeg q90/60/30; blur
σ0.5/1.0/2.0; downscale ×0.5/0.25; noise σ0.02/0.05; crop 0.9/0.8; saturate
×1.5). Both `whole_image_score` (no-localisation arm) and the fused `score`
(with-localisation arm) are recorded per image per condition — one run per
backend covers the full with/without pairing.

**Dataset:** SID-Set validation shard 3, 15/class = 45 images per condition
(15 real / 15 synthetic / 15 tampered), same 45 images at every severity.

**Result — headline, plain baseline vs the full upgrade** (from
`EVALUATION_RESULTS.md` §7a; `auc_real_vs_tampered`):

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

Beats the plain baseline at 13/14 conditions; downscale ×0.25 is the one loss
anywhere in this log where the full upgrade underperforms the original
baseline it replaces.

**Result — isolating localisation alone** (base backend: mostly noise at
n=15, ±0.05, except blur σ2.0 +0.253 and downscale ×0.25 +0.169; patch
scorer: **net negative at 5/14 conditions** — jpeg q60 −0.222, jpeg q30
−0.187, blur σ1.0 −0.196, downscale ×0.5 −0.222, downscale ×0.25 −0.324 —
where the map keeps finding the right pixels (recall ≥ clean-level, up to
0.939 at jpeg q30) but fusing that finding in makes the decision *worse* than
trusting the whole-image number alone). Full tables in
`EVALUATION_RESULTS.md` §7b.

**Result — confidence and verdict stability** (§6): mean confidence is a flat
0.60 in every single condition (the uncalibrated cap — carries no
information right now). Under noise σ=0.05, only 9% of images keep the same
verdict as clean while AUC is essentially unchanged (0.764 vs 0.782) — the
score ranking survives, the label does not.

**Note on the three job IDs:** 778211 / 778685 / 778939 ran the identical
command against the identical (deterministically-fetched) shard 3 sample
three times in succession on 2026-08-31, each ~30 minutes, each overwriting
`eval_results/robustness_{base,patch_scorer}.json`. This reflects
`eval/robustness.py` itself being iterated on across the three runs (e.g. the
`whole_image_score` field this whole with/without pairing depends on was
added between runs — see `EVALUATION_RESULTS.md` §7's note that it "now also
records `whole_image_score`... which reproduces §1's with/without pairing per
degradation condition"). Only the **last** run's output (778939) is what
`eval_results/robustness_{base,patch_scorer}.json` and the tables above
currently reflect.

**Source:** `server/model_03/EVALUATION_RESULTS.md` §6–§7. Jobs **778211**
(2026-08-31 11:58:54–12:28:26), **778685** (13:45:29–14:15:59), **778939**
(14:45:55–15:17:40) `aigc-m03-robust`,
`slurm_logs/model03_robust_{778211,778685,778939}.{out,err}`. Current data:
`eval_results/robustness_base.json`, `eval_results/robustness_patch_scorer.json`.

---

## 8. CIFAKE robustness test — real vs synthetic only

**Question:** everything above uses SID-Set, whose whole design is the
partially-AI (`tampered`) case. What does the same base-vs-shipped comparison
look like on a dataset with **no** in-between case — just real photos and
wholly AI-generated ones?

**Parameters:** new tooling written for this run —
`eval/fetch_cifake.py --per_class 100` (samples CIFAKE, pools Kaggle's
train/test splits, writes a `manifest.json` with `class: real|synthetic`
only, no mask), then `eval/robustness.py --backend <base|patch_scorer>
--limit 100 --max_side 768` (same 14 conditions as run 7), then
`eval/report_cifake.py` to build the table.

**Dataset:** CIFAKE, 100/class = 200 images, native 32×32 (no resize —
smaller than `max_side`).

**Result — headline:**

| Condition | base backend | patch scorer | delta |
|---|---|---|---|
| clean | 0.617 | 0.669 | +0.052 |
| jpeg q90 | 0.615 | 0.672 | +0.056 |
| jpeg q60 | 0.620 | 0.645 | +0.025 |
| jpeg q30 | 0.588 | 0.616 | +0.028 |
| blur σ0.5 | 0.597 | **0.709** | **+0.112** |
| blur σ1.0 | 0.583 | **0.701** | **+0.118** |
| blur σ2.0 | 0.546 | 0.614 | +0.068 |
| downscale ×0.5 | 0.582 | 0.611 | +0.029 |
| downscale ×0.25 | 0.520 | 0.585 | +0.065 |
| noise σ0.02 | 0.579 | 0.582 | +0.003 |
| noise σ0.05 | 0.532 | 0.604 | +0.072 |
| crop 0.9 | 0.610 | **0.701** | **+0.091** |
| crop 0.8 | 0.548 | 0.641 | +0.093 |
| saturate ×1.5 | 0.618 | 0.643 | +0.025 |

Patch scorer wins all 14/14 conditions (+0.003 to +0.118); AUC stays weak
throughout (0.52–0.71), consistent with a 224px-trained backend scoring
native 32×32 content.

**Critical finding — localisation contributed exactly nothing, measured, not
assumed:** with-localisation and without-localisation scores were
bit-identical (delta 0.000 to 3 decimals) in **all 28 rows** (14 conditions ×
2 backends). `mapper/windows.py` clamps the `[64,128,224]` scales down to a
32px image's own short side; the duplicate clamped scales collapse to one,
so the multi-scale window mapper degenerates to a single whole-image window.
The entire headline table above is a backend/classifier comparison, not a
localisation result — confirmed empirically this run, not just predicted.

Verdict stability under blur/downscale is markedly worse for the base
backend (drops to 0.13–0.41) than the patch scorer (stays ≥0.94 everywhere)
— the same "confident but changes its mind" pattern run 7 found on SID-Set.

**Source:** New this run — no prior write-up. Job **779287**
`aigc-m03-cifake`, 2026-08-31 16:21:15 → 16:27:57, 6m42s,
`slurm_logs/model03_cifake_779287.{out,err}`. Full tables:
`server/model_03/CIFAKE_RESULTS.md`. Raw data:
`eval_results/robustness_cifake_{base,patch_scorer}.json`. New scripts:
`eval/fetch_cifake.py`, `eval/report_cifake.py`.

---

## 9. SID-Set robustness test — fresh shard, bigger sample

**Question:** run 7's numbers are from shard 3 at 15/class (±0.09 AUC
standard error, called out repeatedly in `EVALUATION_RESULTS.md` as too
noisy to trust individual-row deltas). Does a fresh, larger sample tell the
same story?

**Parameters:** identical to run 7's methodology, scaled up:
`eval/fetch_sid_set.py --split validation --shard 4 --per_class 50`, then
`eval/robustness.py --backend <base|patch_scorer> --limit 50 --max_side 768`,
then the new general-purpose `eval/report_robustness.py` (handles the
`tampered` class run 8's CIFAKE-specific report script doesn't need to).

**Dataset:** SID-Set validation **shard 4** (untouched by every prior run in
this log — 3/5b/6 used shards 0–2 for training/fitting, 7 used shard 3),
50/class = 150 images per condition.

**Result — headline, plain baseline vs the full upgrade** (`auc_real_vs_tampered`,
and `auc_real_vs_all_ai`; full tables in `SID_SET_RESULTS.md`):

| Condition | base, no localisation | patch scorer + localisation | delta |
|---|---|---|---|
| clean | 0.587 | **0.854** | +0.267 |
| jpeg q90 | 0.592 | **0.872** | +0.280 |
| jpeg q60 | 0.387 | 0.594 | +0.207 |
| **jpeg q30** | 0.555 | **0.367** | **−0.188** |
| blur σ0.5 | 0.590 | **0.835** | +0.245 |
| blur σ1.0 | 0.576 | 0.554 | −0.023 |
| blur σ2.0 | 0.562 | 0.501 | −0.061 |
| downscale ×0.5 | 0.568 | 0.575 | +0.007 |
| **downscale ×0.25** | 0.544 | **0.411** | **−0.134** |
| noise σ0.02 | 0.589 | **0.775** | +0.186 |
| noise σ0.05 | 0.630 | 0.726 | +0.096 |
| crop 0.9 | 0.587 | **0.872** | +0.285 |
| crop 0.8 | 0.635 | **0.871** | +0.235 |
| saturate ×1.5 | 0.603 | **0.901** | +0.298 |

On AUC(all AI) the same comparison runs 0.759 → 0.919 clean and 0.735 → 0.430 at
jpeg q30. Localisation, patch scorer: mean IoU 0.416 clean (base: 0.041), and
**still 0.372 at jpeg q30** while the fused decision there is below chance.

**Conclusion — run 7's pattern reproduces at 3.3× the sample, and two things it
could not resolve are now resolved:**

1. **For the base backend, localisation is a small net negative**, not noise:
   AUC(all AI) with-minus-without is negative in **13 of 14** conditions
   (−0.091 to +0.017). At 15/class the ±0.09 interval could not establish that
   sign.
2. **The patch scorer's failure mode is confident wrongness, not weakness.** At
   jpeg q30 the map keeps near-clean quality (IoU 0.372, recall 0.887) while the
   score built on it inverts; at downscale ×0.25 the map itself degrades (IoU
   0.416 → 0.123) and fusion trusts it anyway. Both are the map being believed
   when it should not be — which is what run 10 goes after.

Verdict stability under noise is unchanged from run 7 (0.34 at σ0.05 for the
patch scorer, 0.30 base), and mean reported confidence is a flat 0.60
everywhere — the finding run 10 shows to be an artifact of the cap.

**Source:** `server/model_03/SID_SET_RESULTS.md` (generated by
`eval/report_robustness.py`), narrated in `EVALUATION_RESULTS.md` §8. Job
**779368** `aigc-m03-sidset`, 2026-08-31 16:43:57 → 18:14:30, 1h30m33s,
`slurm_logs/model03_sidset_779368.{out,err}`. Raw data:
`eval_results/robustness_sidset_shard4_{base,patch_scorer}.json`.

---

## 10. Threshold-mode diagnostic — why fusion inverts, and can it be re-cut?

**Question:** run 9 leaves the map firing accurately and the decision built on
it inverting. Two hypotheses: the map's scores are *mis-cut* (region proposal
uses absolute thresholds, and degradation shifts the score distribution
bodily), or the per-patch signal is simply *gone* at those severities. If it is
the first, re-placing the cuts fixes it.

**Parameters:** four paired arms on the identical frames, `eval/robustness.py
--data_dir <shard 4> --limit 50 --max_side 768`:

| Arm | Backend | `--threshold_mode` |
|---|---|---|
| 1 | `checkpoints/patch_scorer` | `absolute` (shipped) |
| 2 | `checkpoints/patch_scorer` | `quantile` |
| 3 | `checkpoints/patch_scorer` | `median_shift --adaptive_ref_median 0.3660738479313538` |
| 4 | `hf:Organika/sdxl-detector` | `absolute` |

The `median_shift` reference median is **measured from arm 1's own clean
condition inside the job**, not hardcoded, so the constant belongs to the same
backend and working resolution it is applied at. This run also added the
instrumentation everything after it depends on: per-class map statistics, the
map median, `confidence_uncapped`, and the full `per_image` rows in the output
JSON (`eval/robustness.py`, `fusion.py`, `mapper/heatmap.py`).

**Dataset:** SID-Set validation **shard 4**, 50/class — deliberately the same
sample as run 9, so every arm is directly comparable to
`robustness_sidset_shard4_*.json`.

**Result — the distribution shift is real, and it is on the real images:**

| Condition | mean map median, **real** | mean map median, tampered | real images that grew a region |
|---|---|---|---|
| clean | 0.022 | 0.081 | 14% |
| jpeg q60 | 0.120 | 0.081 | 56% |
| jpeg q30 | **0.382** | 0.380 | **96%** |
| blur σ1.0 | 0.133 | 0.086 | 70% |
| blur σ2.0 | **0.683** | 0.287 | **98%** |
| downscale ×0.25 | 0.228 | 0.051 | 60% |
| noise σ0.05 | 0.008 | 0.011 | **0%** |

**Result — re-cutting does not fix detection:**

| Mode | mean AUC(tampered) | worst condition | mean loc IoU | real images firing, clean |
|---|---|---|---|---|
| absolute | 0.693 | 0.368 | 0.306 | 14% |
| quantile | 0.693 | 0.372 | 0.218 | 38% |
| median_shift | **0.708** | 0.378 | **0.346** | 42% |

`median_shift` beats `absolute` on IoU in **11 of 14** conditions and recovers
blur σ2.0 (0.500 → 0.668), but triples the clean false-positive rate and barely
moves the worst condition.

**Result — the confidence signal is not flat, the cap is:** `confidence_uncapped`
averages 0.875 clean, 0.813 at jpeg q60, 0.784 at jpeg q30, with the
uncalibrated cap binding on **98%** of images. Gating on it (cut 0.8577, chosen
by leave-one-condition-out over ten candidate signals) takes mean AUC(tampered)
from 0.693 to **0.806** and the worst condition from 0.368 to **0.666**;
rank-normalising both arms first — which removes any gain that is merely the
two score scales sitting apart — leaves **+0.038** mean. A separate
`sdxl-detector` as the fallback measured worse (0.745; 0.728 with a fitted
quantile alignment) than reusing the patch scorer's own whole-image pass.

**Conclusion:** the per-patch signal is gone rather than mis-cut — the
whole-image pass still separates at 0.605 at jpeg q30 while every fused variant
sits at 0.37–0.45. Threshold placement is therefore a *localisation* knob, not a
robustness fix, and the productive move is to decide per image whether to
believe the map at all. That is what `dual_backend.py` implements and what runs
11 and 12 test.

**Source:** `server/model_03/THRESHOLD_RESULTS.md` (generated by
`eval/report_thresholds.py`), narrated in `EVALUATION_RESULTS.md` §9–§10. Job
**779811** `aigc-m03-diag`, 2026-08-31 18:40:00 → 20:17:31, 1h37m31s,
`slurm_logs/model03_diag_779811.{out,err}` (job 779757 is the same script
cancelled at submit time). Raw data: `eval_results/diag_shard4_{ps_absolute,
ps_quantile,ps_median_shift,base_absolute}.json`. New/changed code:
`mapper/heatmap.py` (`threshold_mode`), `fusion.py` (`confidence_uncapped`),
`eval/robustness.py` (per-class stats, `per_image` rows), `dual_backend.py`,
`eval/report_thresholds.py`, `scripts/fit_score_alignment.py`.

---

## 11. Held-out validation — the shard-4 findings with the constants frozen

**Question:** run 10 chose a threshold mode and a gate threshold *on shard 4*.
Both were selected by searching that shard, so shard 4 cannot score them. Do
they survive on data neither was fitted on?

**Parameters:** `eval/robustness.py --data_dir <shard 3> --limit 50 --max_side
768 --backend checkpoints/patch_scorer`, twice: `--threshold_mode absolute`, and
`--threshold_mode median_shift --adaptive_ref_median 0.3660738479313538` — the
reference median **carried over from shard 4 as a literal**. The gate threshold
0.8577 likewise stays a literal in `eval/validate_gate.py`; `--retune` reports
what this shard would have chosen, as a diagnostic only.

**Dataset:** SID-Set validation **shard 3**, 50/class. A different parquet file
from shard 4, so the images are disjoint. Runs 7's shard-3 numbers used 15/class
and the uninstrumented harness; this writes to its own files
(`val3_shard3_*.json`), so nothing already cited is overwritten.

**Result — the gate reproduces** (`AUC(real vs tampered)`, 14 conditions):

| | always fuse | always whole-image | gate | shard 4's gate |
|---|---|---|---|---|
| mean | 0.692 | 0.665 | **0.803** | 0.806 |
| worst | 0.376 | 0.528 | **0.687** | 0.666 |
| conditions below 0.5 | 3 | 0 | **0** | 0 |
| clean | 0.847 | 0.665 | 0.856 | 0.854 |
| jpeg q30 | 0.376 | 0.598 | **0.775** | 0.841 |

On AUC(real vs all AI): mean 0.809 → **0.890**, worst 0.473 → **0.836**, and FPR
at 80% recall on clean data 0.080 → **0.040**. Rank-normalised, the gain is
+0.034 mean / +0.185 worst. `--retune` puts the tuning optimism at 0.004: shard
3's own best threshold (0.8744) would have scored 0.807 against the frozen
threshold's 0.803.

**Result — the threshold-mode finding reproduces, including its cost:**
`median_shift` again wins IoU in **11 of 14** conditions (mean 0.264 → 0.318),
again fires on far more authentic photographs (8% → 28% clean), and again fails
to fix detection (mean AUC(tampered) 0.692 → 0.703, worst 0.376 → **0.333**).

**Conclusion:** both shard-4 findings are real, and the gate's held-out summary
statistics land within 0.021 of their tuning-shard values (mean 0.803 vs 0.806,
worst 0.687 vs 0.666); individual conditions move more, as the sample size
implies. `absolute`
stays the default; `median_shift` is a localisation option for a deployment that
can afford the false positives.

**Source:** narrated in `EVALUATION_RESULTS.md` §9–§10. Job **780095**
`aigc-m03-val3`, 2026-08-31 20:38:57 → 21:26:05, 47m08s,
`slurm_logs/model03_val3_780095.{out,err}`. Raw data:
`eval_results/val3_shard3_ps_{absolute,median_shift}.json`. Replay:
`python eval/validate_gate.py eval_results/val3_shard3_ps_absolute.json --retune`.

---

## 12. Dual-backend run — the class itself, and the first test pass over the new code

**Question:** two gaps. Everything about the gate so far was replayed offline
from per-image JSON, so `DualBackendAnalyzer` had never actually run; and
`fusion.py`, `mapper/heatmap.py` and `eval/robustness.py` were all modified in
run 10 without the test suite ever seeing the changes — the login node has
neither torch nor pytest.

**Parameters:** `python -m pytest tests/ -q` (the sbatch deliberately does *not*
use `set -e`, so a failing suite still reports and the run continues), then
`eval/robustness.py --data_dir <shard 3> --limit 50 --max_side 768 --backend
checkpoints/patch_scorer --dual` twice: `--fallback_backend self` and
`--fallback_backend hf:Organika/sdxl-detector`. The trust threshold stays the
frozen shard-4 value; `eager` scoring is on so both arms exist for every frame
in one pass.

**Dataset:** SID-Set validation shard 3, 50/class — the same sample as run 11,
so the class's output is directly comparable with run 11's offline replay.

**Result — tests: 101 passed** (`test_dual_backend.py` is new: 22 of them),
first green run over the `threshold_mode` code path.

**Result — the class reproduces the replay exactly:**

| | ungated (run 11) | `--dual --fallback_backend self` |
|---|---|---|
| mean AUC(tampered) | 0.692 | **0.803** |
| worst condition | 0.376 | **0.687** |
| mean AUC(all AI) | 0.809 | **0.890** |
| worst AUC(all AI) | 0.473 | **0.837** |
| mean reported localisation IoU | 0.264 | **0.181** |

The gate distrusts localisation on **36%** of images on average (23% clean, 75%
at jpeg q30). The IoU row is the cost, and it is reported rather than hidden: a
distrusted image contributes an empty mask, because `dual_backend.py` drops the
findings when it declines to believe them and the harness scores localisation on
what the system actually reported.

**Result — a separate model is the worse fallback, held out too:**

| Fallback | mean AUC(tampered) | worst | mean AUC(all AI) | worst |
|---|---|---|---|---|
| `self` (the patch scorer's own whole-image pass) | **0.803** | **0.687** | **0.890** | **0.837** |
| `hf:Organika/sdxl-detector` | 0.714 | 0.581 | 0.824 | 0.717 |

`self` wins **13 of 14** conditions. Both arms distrust the same 36% of images
and report the same localisation (0.181 IoU) — the gate is identical, only the
number substituted differs. This arm ran **unaligned** (`--dual` does not take an
`alignment_path`, so `ScoreAligner` is the identity and the run's own notes say
so); run 10's offline analysis measured the aligned variant too and it was no
better (0.745 unaligned / 0.728 aligned on shard 4), so alignment is not what is
missing — a second detector's global view is simply worse than the fine-tuned
backend's own.

**Conclusion:** the gate is a real, working component and not an artifact of
offline replay, and the cheap version of it is the right one — no second model,
no extra forward pass. It buys detection robustness by giving up localisation on
the images it does not trust, which is the right trade only where the score
matters more than the map; that is a deployment decision, not a measurement.

**Source:** narrated in `EVALUATION_RESULTS.md` §10. Job **780205**
`aigc-m03-dual`, 2026-08-31 21:03:49 → 21:52:46, 48m57s,
`slurm_logs/model03_dual_780205.{out,err}`. Raw data:
`eval_results/dual_shard3_{self,separate}.json`. Reproduce with
`test_model03_dual.sbatch`.

---

## Appendix: where everything lives

| Artifact | Path |
|---|---|
| Current patch-scorer checkpoint | `checkpoints/patch_scorer/` (from run 5b; run 3's original 1,800-image checkpoint was overwritten, backed up off-repo to `$SCRATCH/patch_scorer_prev`) |
| Calibrator (fitted, not used by default) | `configs/calibration_sdxl_detector.json` |
| Uncalibrated config (shipped default behaviour) | `configs/uncalibrated.yaml`; `configs/default.yaml` also carries `calibration_path: null` since run 2 |
| Fetch scripts | `eval/fetch_sid_set.py` (SID-Set, 3-class + masks), `eval/fetch_cifake.py` (CIFAKE, 2-class) |
| Evaluation scripts | `eval/evaluate.py` (detection+localisation+false-positives, one run), `eval/ablation.py` (with/without-localisation comparison from one `evaluate.py` output), `eval/robustness.py` (transform × severity matrix, both arms per run; since run 10 also `--threshold_mode`, `--dual`, per-class stats and `per_image` rows) |
| Decision-rule tooling | `scripts/fit_decision.py` (sweeps map-reduction strategies; not yet reconciled with `fuse()`'s shipped design — see run 5b) |
| Trust-routing tooling (new, runs 10–12) | `dual_backend.py` (`DualBackendAnalyzer` + `ScoreAligner`; the frozen threshold lives here), `eval/validate_gate.py` (held-out replay of the frozen gate, raw and rank-normalised), `scripts/fit_score_alignment.py` + `configs/score_alignment_patch_scorer_vs_sdxl.json` (only needed when the fallback is a separate model) |
| Report generators | `eval/report.py` (detection/localisation tables from `evaluate.py` JSON), `eval/report_cifake.py` (CIFAKE-specific robustness report, carries the windowing-collapse caveat), `eval/report_robustness.py` (general-purpose — handles a `tampered` class when present), `eval/report_thresholds.py` (threshold modes + routing, new in run 10) |
| Result documents | `EVALUATION.md` (how to evaluate + standing findings), `EVALUATION_RESULTS.md` (the narrative results), `CIFAKE_RESULTS.md` (run 8), `SID_SET_RESULTS.md` (run 9), `THRESHOLD_RESULTS.md` (run 10) |
| Slurm scripts | `test_model03_*.sbatch`, `train_model03_*.sbatch`, `eval_model03_patch_scorer.sbatch`, `diag_model03_thresholds.sbatch`, `validate_model03_shard3.sbatch` at the repo root |
| Slurm logs | `slurm_logs/model03_*_<jobid>.{out,err}` at the repo root |
| Unit tests | `tests/` — 101 as of run 12, all passing on GPU with torch installed |

Regenerate this document's numbers with the commands quoted in each section;
nothing here was typed from memory.
