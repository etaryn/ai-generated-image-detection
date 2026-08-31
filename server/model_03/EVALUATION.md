# Evaluating model_03

model_03 has two very different kinds of check, and conflating them is how a
prototype ends up sounding validated when it isn't:

- **`tests/`** — 101 unit tests. They answer *does the machinery do what it says?*
  They run against a stub scorer, need no network and no GPU, and finish in
  seconds. They cannot tell you whether the maps land on real edits.
- **`eval/`** — this document. It answers *is it actually right?*, against
  ground-truth tamper masks. It needs the dataset, a GPU, and time.

Everything here is reproducible from the commands below. No number in this repo
should exist that you cannot regenerate.

---

## The objective, and what counts as success

**There are two classes, real and AI. Each has an unprocessed and a processed
subset, where "processed" means the image was degraded — resized, blurred,
cropped, recompressed. The system makes one call, real vs AI, and it has to hold
up across that degradation.**

|  | unprocessed | processed (degraded) |
|---|---|---|
| **real** | real, clean | real, degraded |
| **AI** | AI, clean | AI, degraded |

This matters for how SID-Set is read. Its label 2 is **not** a third class: it is
an AI image whose AI content occupies only part of the frame. That makes it the
*hardest AI case* — and the entire reason a localisation layer exists, since a
whole-image score averages a small AI region into invisibility.

**Localisation is a means, not an end.** Per-pixel map AUC and mask IoU are
diagnostics that explain behaviour. They are not success criteria. The only
metric that decides anything is AUC(real vs AI).

| # | Condition | Target | Base model | Patch scorer | + confidence gate |
|---|---|---|---|---|---|
| 1 | AUC(real vs AI), clean | ≥ 0.90 | 0.739 | **0.921** ✓ | **0.926** ✓ |
| 2 | AUC(real vs AI), worst degradation | ≥ 0.85 | 0.674 | 0.473 ✗ | 0.836 |
| 3 | AUC on the **partially-AI** subset, clean | ≥ 0.80 | 0.589 | **0.847** ✓ | **0.856** ✓ |
| 4 | FPR on real images at 80% recall, clean | ≤ 0.10 | 0.492 | **0.080** ✓ | **0.040** ✓ |
| 5 | Region-aware beats whole-image on all AI | significant | +0.018, n.s. | **+0.090** [+0.028, +0.153] | — |

Condition 5 is the project's thesis. Base-model columns are the 2026-08-30
shard-0 run; the last two are held-out shard 3 at 50/class, 768px
(`EVALUATION_RESULTS.md` §8–§10). Conditions 1, 3, 4 and 5 are met by the
trained patch scorer; **condition 2 is the one still open**, and it is now the
only one.

The reason is worth stating precisely, because it inverted since the base model.
For the base model, degradation was not the binding constraint — AUC stayed in
0.674–0.802 across 14 conditions and partially-AI images were the whole problem.
The patch scorer fixed partially-AI images and *created* a degradation problem:
under heavy JPEG and downscaling its map fires confidently on degraded
**authentic** photographs, and fusing that map drives the decision below chance
(0.376 at jpeg q30 against the same backend's whole-image 0.598). Gating on the
pipeline's own pre-cap confidence recovers most of it — worst condition 0.376 →
0.687 on the partially-AI subset, 0.473 → 0.836 on all AI — which is what the
last column reports, and why it is a gate rather than a better map.

## The primary experiment: does the idea work?

Everything else here characterises the pipeline. This one tests the thesis, in
three steps:

1. **With localisation** — the full pipeline: multi-scale map → regions →
   specialist routing → fusion.
2. **Without localisation** — the same backend, on the same image, scored once
   as a whole. This is exactly what model_01 and model_02 do, and what every
   whole-image detector does.
3. **Compare** — and conclude.

The two arms are **paired**: `eval/evaluate.py` records both numbers for every
image in the same run, through the same backend, with the same preprocessing. So
the difference isolates the region machinery and nothing else — not a different
model, not a different sample. `eval/ablation.py` then does the comparison:

```bash
python eval/ablation.py eval_results/sid_set_calibrated.json --out EVALUATION_RESULTS.md
```

It reports three things that must not be conflated:

| | What it answers | Baseline comparable? |
|---|---|---|
| **Detection AUC**, with a paired bootstrap CI on the difference | Does localisation make the image-level verdict *better*? | Yes — head to head |
| **False positives at matched recall** | Was any gain bought by flagging everything? | Yes — both thresholded to flag the same share of AI images |
| **Localisation IoU, and `ai_edited` vs `ai_generated`** | Where is the edit, and what kind is it? | **No** — a single score cannot do either at any threshold |

The third row is the honest awkwardness of this evaluation: those are the
capabilities the system exists for, and they have no baseline to beat, so they
are reported separately rather than folded into a win/loss.

The detection comparison is reported separately for **tampered** and
**synthetic** images, because the thesis makes different predictions for each: a
small edit is diluted in a whole-image score (localisation should help), while a
wholly generated image has nothing to localise (it should not).

`eval/ablation.py` prints an explicit overall verdict, and it can return a
negative one — "the idea does NOT yet work" — which `tests/test_ablation.py`
verifies on constructed data. An ablation that can only conclude "it works" is
not an ablation.

## The dataset

[SID-Set](https://huggingface.co/datasets/saberzl/SID_Set) (Huang et al., *SIDA:
Social Media Image Deepfake Detection, Localization and Explanation*) — the
dataset this repo's top-level README already names. It is the right instrument
here because it has **three** classes, not two:

| label | class | what it contributes |
|---|---|---|
| 0 | `real` | authentic photographs (OpenImages V7) — false-positive rate |
| 1 | `synthetic` | wholly generated images — the "generated" hypothesis |
| 2 | `tampered` | real photos with a generated region **+ a binary mask** — localisation ground truth |

A two-class dataset cannot measure the distinction model_03 is built around
("this image was generated" vs "this photograph was locally edited") at all.

`eval/fetch_sid_set.py` streams parquet row groups over HTTP range requests and
stops when it has enough, so a few-hundred-image sample costs a few hundred MB
rather than the 17GB validation split.

**Two caveats to carry into every number below.**

1. The official **test** split is gated (the authors withhold it to prevent
   leakage), so this uses **validation**. For model_03 that is sound — it trains
   on nothing — but the *backend* is a third-party detector whose training data
   is unknown, and SID-Set's real images come from OpenImages, which appears in
   very many training sets. **Read detection AUC as an upper bound.** The
   localisation numbers are the trustworthy ones: no whole-image detector could
   have memorised where a mask is.
2. Fit calibration on a **different shard** than you evaluate on. The commands
   below use shard 1 to fit and shard 0 to score. Doing otherwise reports
   training error as a result.

---

## Running the full evaluation

From `server/model_03`, in order. On a single RTX 4060 the whole sequence is a
few hours; on a stronger machine it is bounded by the same knobs, listed under
"Scaling up".

```bash
# 0. deps
pip install -r requirements.txt          # numpy, Pillow, torch, transformers

# 1. sanity: does the backend even point the right way on your data?
#    Reports CORRECT / INVERTED / NON-DISCRIMINATIVE with the AUC behind it.
python scripts/check_backend.py --data_dir <folder with real/ and fake/> --backend hf

# 2. fetch two disjoint samples: one to score, one to calibrate on
python eval/fetch_sid_set.py --shard 0 --per_class 120 --out eval_data/sid_set_val
python eval/fetch_sid_set.py --shard 1 --per_class 80  --out eval_data/sid_set_cal

# 3. fit per-scale calibrators (isotonic; see "What is already known")
python scripts/calibrate_mapper.py \
    --manifest eval_data/sid_set_cal/manifest.json \
    --backend hf --out configs/calibration_sdxl_detector.json

# 4. detection + localisation + false positives + verdict confusion
python eval/evaluate.py --data_dir eval_data/sid_set_val \
    --out eval_results/sid_set_calibrated.json

#    the uncalibrated baseline, for the before/after
python eval/evaluate.py --data_dir eval_data/sid_set_val \
    --config configs/uncalibrated.yaml \
    --out eval_results/sid_set_uncalibrated.json

# 4b. THE PRIMARY EXPERIMENT: with vs without localisation, paired, on the
#     same images. Prints an explicit verdict on whether the idea works.
python eval/ablation.py eval_results/sid_set_calibrated.json \
    eval_results/sid_set_uncalibrated.json \
    --out EVALUATION_RESULTS.md --json_out eval_results/ablation.json

# 5. transform x severity robustness matrix
python eval/robustness.py --data_dir eval_data/sid_set_val --limit 20 \
    --out eval_results/robustness.json

# 6. compare backends on the same images
python eval/evaluate.py --backend haywoodsloan/ai-image-detector-deploy \
    --limit 60 --out eval_results/sid_set_swinv2.json

# 7. regenerate the tables
python eval/report.py eval_results/*.json \
    --robustness eval_results/robustness.json --out EVALUATION_RESULTS.md

# 8. WHERE DOES IT BREAK, AND SHOULD THE MAP BE BELIEVED? Steps 5-7 tell you
#    that a condition is bad; these tell you why and what to do about it.
#    Every arm below reads the per-image rows eval/robustness.py now writes.

#    8a. does re-placing the map's cuts fix a degradation, or is the patch
#        signal simply gone? Run the same frames under each threshold mode.
for mode in absolute quantile median_shift; do
  python eval/robustness.py --data_dir eval_data/sid_set_val --limit 50 \
      --backend checkpoints/patch_scorer --threshold_mode $mode \
      --out eval_results/diag_ps_$mode.json
done
python eval/report_thresholds.py eval_results/diag_ps_*.json \
    --out THRESHOLD_RESULTS.md

#    8b. the gate: is the map trustworthy on THIS image? Replays the frozen
#        rule offline, reports raw and rank-normalised gains, and --retune
#        quantifies how optimistic a threshold fitted here would be.
python eval/validate_gate.py eval_results/diag_ps_absolute.json --retune

#    8c. the same gate through the real class, on a held-out shard. --dual
#        wraps the pipeline in DualBackendAnalyzer; the threshold stays frozen.
python eval/robustness.py --data_dir eval_data/sid_set_val_heldout --limit 50 \
    --backend checkpoints/patch_scorer --dual --fallback_backend self \
    --out eval_results/dual_self.json

#    8d. only if the fallback is a SEPARATE model: put its scores on the
#        primary's scale first, fitted on clean images, or the substitution
#        moves the ranking for reasons unrelated to the image.
python scripts/fit_score_alignment.py \
    --primary eval_results/diag_ps_absolute.json \
    --fallback eval_results/diag_base_absolute.json \
    --out configs/score_alignment_patch_scorer_vs_sdxl.json
```

`eval_data/` and `eval_results/` are gitignored — they are regenerable, and the
images are not ours to redistribute.

### Scaling up

The knobs, roughly in order of value per unit of compute:

| Knob | Default | On a bigger machine | Cost |
|---|---|---|---|
| `--per_class` on the fetch | 120 | 500–1000 | linear; AUC standard error ~0.03 at 120/class |
| `--shard` | 0 and 1 | several shards, concatenated | linear |
| `mapper.max_side` | 1024 | 1536–2048 | quadratic in patch count |
| `backend.batch_size` | 32 | 128–256 | frees GPU utilisation (see below) |
| `--limit` on robustness | 15–20 | 100+ | linear × 14 conditions |
| backend | `Organika/sdxl-detector` | compare 3–4 from `PUBLIC_MODELS` | linear per backend |

**A bigger GPU now helps — this changed.** The pipeline used to run at 21% GPU
utilisation, bottlenecked on CPU-side patch cropping and `AutoImageProcessor`
resizing (~1250 PIL crops per image). That work now happens on device as batched
tensor ops (`HFImageClassifierBackend.score_crops`), which made patch scoring
**3.05× faster** and moved utilisation to **96%**. The pipeline is now genuinely
compute-bound on the model forward, so more FLOPs translate roughly linearly —
which was not true before.

Two things measured and *not* worth doing, recorded so they are not re-attempted:

- **Bigger batches.** 64 is the optimum on an 8GB card; 128 and 256 were both
  slower (3.75s → 4.36s → 4.69s for 1267 windows). Worth re-checking on a card
  with more memory, but do not assume.
- **TF32 / `channels_last`.** No measurable gain on an fp16 attention model.

Still untried and plausible on a Linux server: `torch.compile` on the backbone,
which typically pays for its compile time over a few hundred images.

---

## What is already known

Findings that are measured and reproducible, independent of the full run.

### The backend's labels must be resolved by name, never by index

Of six surveyed public detectors, five put the AI class at logit index 0 and
`dima806/ai_vs_real_image_detection` puts it at index 1, with vocabularies
ranging over `artificial/human`, `REAL/FAKE`, `ai/hum`, `Fake/Real`. A
hard-coded index inverts that model silently — every generated image reading as
authentic, the heatmap highlighting precisely the untampered regions. See
`mapper/labels.py`, and `scripts/check_backend.py` for the empirical check that
catches a config whose labels are themselves transposed.

On this repo's CIFAKE samples, 60 per class (a demonstration of the check, not a
ranking — CIFAKE is 32×32 and these are 224px models):

| Backend | fake | real | AUC | Verdict |
|---|---|---|---|---|
| `Organika/sdxl-detector` | 0.342 | 0.131 | 0.696 | CORRECT |
| `dima806/ai_vs_real_image_detection` | 0.991 | 0.025 | 1.000 | CORRECT |
| `Ateeqq/ai-vs-human-image-detector` | 0.581 | 0.764 | 0.454 | NON-DISCRIMINATIVE |

`dima806`'s perfect score most likely means CIFAKE was in its training set.
`Ateeqq` is not broken — it saturates on 32px thumbnails, and may be the better
model on real photographs. Re-run this on full-resolution data before choosing.

### The fine scale was a false-positive engine, and calibration is per scale

Fraction of patches scoring above the nominal 0.75 "likely AI" threshold, with
the default backend on SID-Set:

| Patch scale | from **authentic** photographs | from synthetic images |
|---|---|---|
| 64px | **36.6%** | 0.708 mean |
| 128px | 16.3% | 0.709 mean |
| 224px | 10.4% | 0.702 mean |

The fine scale is not more sensitive to generated content — synthetic images
score the same at every scale. It is far more prone to calling *authentic*
content generated, because a 64px crop blown up to a 224px input looks smooth
and textureless, which is what these models were trained to read as "generated".
Combined with `max` scale-combination, that made real photographs light up.

A single calibrator cannot fix a scale-dependent distortion, so calibration is
per scale (`ScaleCalibrators`). Fitted on SID-Set shard 1, held-out ECE:

| Scale | before | after |
|---|---|---|
| 64px | 0.234 | **0.089** |
| 128px | 0.221 | **0.080** |
| 224px | 0.191 | **0.071** |

### Isotonic, not Platt

Platt scaling made held-out ECE **worse** at every scale (0.221 → 0.273 at
128px). These detectors saturate into a bimodal pile at 0 and 1, which a
two-parameter logistic cannot represent; isotonic regression can. `--method
isotonic` is now the default, and `calibrate_mapper.py` refuses to write any
calibrator that does not improve held-out ECE — running uncalibrated, with
confidence capped at 0.60, is the honest state rather than shipping a map that
makes things worse.

### Localisation floor

A region is confidently mapped only where windows fit *inside* it, so the finest
scale bounds the smallest edit that can be outlined (as opposed to merely
centred on). On the synthetic fixture in `tests/test_pipeline.py`, a 192px edit
recovers at IoU 0.66 with scales `[64, 128, 224]` and IoU 0.17 with
`[128, 224]`. Many SID-Set tampered regions are ~1% of the frame, at or below
this floor — which is why `eval/evaluate.py` stratifies localisation by true
edit size instead of reporting one average.

### Degradation moves the map's scores, and it moves them on *real* images

The region thresholds are absolute (0.45 / 0.75), so anything that shifts the
map's score distribution changes how much of the frame fires without the content
changing. Measured on shard 4 with the patch scorer, mean map median on
**authentic** photographs: 0.022 clean → 0.120 at jpeg q60 → **0.382** at jpeg
q30 → **0.683** at blur σ2.0, against tampered images that barely move
(0.081 → 0.380 → 0.287). Real images with any region go 14% → 96%. Under noise
the reverse: everything collapses to ~0.01 and the map switches off entirely.

Two consequences, both measured rather than argued:

- **This is the 64px-crop problem one level up.** A compressed or blurred
  authentic photograph looks, patch by patch, like what a patch-scale detector
  was trained to call generated. The fix is degradation-augmented hard negatives
  in training, not anything downstream.
- **Adaptive cuts do not rescue it.** `--threshold_mode quantile` and
  `median_shift` (`mapper/heatmap.py`) both remove the shift and neither
  recovers detection: mean AUC(tampered) 0.693 → 0.693 / 0.708, worst condition
  0.368 → 0.372 / 0.378. `median_shift` *is* the better localiser (higher IoU in
  11 of 14 conditions on two shards) at the cost of firing on 3× as many
  authentic photographs, so it is an option, not the default.

### Trusting the map per image beats trying to make it robust

Since the map fails confidently rather than quietly, the productive question is
whether the pipeline can tell when to ignore it. It can: gating on
`confidence_uncapped` — the pre-cap confidence `fusion.py` now records — at
0.8577 lifts mean AUC(tampered) across 14 degradation conditions from 0.692 to
**0.803** and the worst condition from 0.376 to **0.687**, on a shard the
threshold was not tuned on, with no measurable change on clean images
(+0.009, CI [−0.063, +0.080]). `dual_backend.py` implements it; **but note the
reported `confidence` cannot be used for this** — the uncalibrated cap saturates
it at 0.60 on 98% of images. See `EVALUATION_RESULTS.md` §10 for the three ways
this goes wrong if written as a bare `if`.

---

## Training a patch scorer (the server job)

This is the one intervention the evidence supports, and it is the reason
`scripts/train_patch_scorer.py` exists. Run it where the GPUs are.

**Nothing here ships the dataset.** SID-Set is ~123GB of train shards, it is
someone else's data (CC-BY-4.0, built on OpenImages/COCO/Flickr30k), and the
fetch is deterministic — reading fixed row groups in order — so the same command
reproduces byte-identical images anywhere. Fetch it on the machine that needs
it rather than pushing it through git.

```bash
cd server/model_03
pip install -r requirements.txt

# 1. training images (~1.9GB, ~20 min). Exactly the sample used locally:
python eval/fetch_sid_set.py --split train --shard 0 --shards 6 \
    --per_class 100 --out eval_data/sid_set_train

#    scale up on a server -- 40 shards is ~12k images:
#    python eval/fetch_sid_set.py --split train --shard 0 --shards 40 \
#        --per_class 250 --out eval_data/sid_set_train

# 2. fine-tune. Raise batch_size to fill a bigger card. --hard_negatives is the
#    lever on false positives; raise it if real images keep tripping.
python scripts/train_patch_scorer.py \
    --data eval_data/sid_set_train --out checkpoints/patch_scorer \
    --epochs 3 --per_image 4 --batch_size 32 --hard_negatives 512

# 2b. how should the map become a decision? The thresholds in configs/ were set
#     when the map had no spatial signal, so re-fit them against the objective
#     rather than trusting them on a map that now does.
python scripts/fit_decision.py --backend checkpoints/patch_scorer --limit 50

# 3. score it on the split it never saw
python eval/fetch_sid_set.py --split validation --shard 0 --per_class 120 \
    --out eval_data/sid_set_val
python eval/evaluate.py --backend checkpoints/patch_scorer \
    --out eval_results/patch_scorer.json
python eval/ablation.py eval_results/patch_scorer.json
```

`--backend` accepts a local directory, so a trained checkpoint drops in with no
other change.

**Judge it on AUC(real vs AI), not on patch accuracy and not on map AUC.** The
script now prints all three each epoch, and they can disagree sharply — that
disagreement is the most informative thing it produces:

| Metric | What it measures | Trap |
|---|---|---|
| Patch AUC | global patch ranking | a model can ace it and still flag every real photo |
| Per-pixel map AUC | ranking *within* one image | invariant to absolute level, so real images can float above the threshold while scoring 0.95 here |
| **AUC(real vs AI)** | **the objective** | — |

The first local run made that concrete: per-pixel map AUC went 0.460 → **0.955**,
and real photographs still averaged 0.537 against partially-AI images at 0.311.
Excellent within-image ranking, wrong absolute level. Hence
`--hard_negatives`, which re-shows the real-image patches the model scores
highest — every false positive is one of those.

Numbers from the 1,800-image local run, on 30 held-out images per class:

| Metric | base | patch-trained (1.8k images) |
|---|---|---|
| **AUC(real vs AI)** | 0.771 | **0.802** |
| ├ fully-AI subset | 0.893 | **1.000** |
| └ **partially-AI subset** | 0.648 | **0.604** ⚠ |
| mean score, real images | 0.539 | 0.470 |
| Per-pixel map AUC (diagnostic) | 0.460 | 0.955 |

Training fixed the easy half and lost ground on the hard half. **Getting the
partially-AI subset up is the open problem**, and it is what more data plus hard
negatives is expected to help with — verify it rather than assume it.

## The known ceiling — and what happened when it was lifted

Every *public* backend is trained on **whole images**, and model_03 asks it about
64px crops. That mismatch was the largest single source of error in the system,
and no amount of compute fixed it — the table above is what it looked like from
the inside.

The fix was a patch-scale detector, and it has since been built: SID-Set's
**train** split (210k images, with masks) is public, `scripts/train_patch_scorer.py`
samples patches, labels them by mask membership and fine-tunes a 224px backbone
on that distribution, and `--backend checkpoints/patch_scorer` drops it in with
no other change. On held-out shard 3 it takes AUC(real vs tampered) from 0.665 to
0.847 and localisation IoU from 0.041 to 0.416.

**It moved the ceiling rather than removing it.** The mismatch is now between a
patch detector trained on clean patches and the degraded patches it meets in
redistribution — see "Degradation moves the map's scores" above. The next
training run should carry the degradations of `eval/robustness.py` into the
augmentation pipeline and re-show the real-image patches that score highest under
them; that is the direct fix for the one success condition still unmet, and the
one item on this page where a stronger GPU pays for itself directly.

---

## Result files

`eval/report.py` regenerates all tables from the JSON, because numbers retyped
from a terminal drift from the runs that produced them. Each result file records
the backend (model, device, `id2label`, which index was read as AI and how that
was decided), the mapper config, and per-image rows — so a table can always be
traced back to the run that made it.

`eval/robustness.py` keeps the full per-image rows for every condition
(`per_image` in its output), which is what makes the gate replayable offline:
`eval/validate_gate.py`, `eval/report_thresholds.py` and
`scripts/fit_score_alignment.py` all read those rows rather than costing another
GPU hour. The other generators are `eval/report_robustness.py` (any class mix),
`eval/report_cifake.py` (CIFAKE, carries the windowing-collapse caveat) and
`eval/report_thresholds.py` (threshold modes and routing).
