# Evaluating model_03

model_03 has two very different kinds of check, and conflating them is how a
prototype ends up sounding validated when it isn't:

- **`tests/`** — 79 unit tests. They answer *does the machinery do what it says?*
  They run against a stub scorer, need no network and no GPU, and finish in
  seconds. They cannot tell you whether the maps land on real edits.
- **`eval/`** — this document. It answers *is it actually right?*, against
  ground-truth tamper masks. It needs the dataset, a GPU, and time.

Everything here is reproducible from the commands below. No number in this repo
should exist that you cannot regenerate.

---

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

# 2. fine-tune. Raise batch_size to fill a bigger card.
python scripts/train_patch_scorer.py \
    --data eval_data/sid_set_train --out checkpoints/patch_scorer \
    --epochs 3 --per_image 4 --batch_size 32

# 3. score it on the split it never saw
python eval/fetch_sid_set.py --split validation --shard 0 --per_class 120 \
    --out eval_data/sid_set_val
python eval/evaluate.py --backend checkpoints/patch_scorer \
    --out eval_results/patch_scorer.json
python eval/ablation.py eval_results/patch_scorer.json
```

`--backend` accepts a local directory, so a trained checkpoint drops in with no
other change.

**Judge it on per-pixel map AUC, not patch accuracy.** The script prints both and
refuses to endorse the model as a localiser if the map stays near chance. A
model can rank patches well globally and still produce a useless map, because
what the mapper needs is that patches *inside* an edit outrank patches *outside*
it **within the same image**. Patch accuracy is exactly the metric that would let
us declare success without having earned it. The base model's numbers to beat:

| Metric | base model |
|---|---|
| Per-pixel map AUC vs masks | **0.460** (below chance) |
| Patch AUC on mask-labelled patches | **0.335** (below chance) |
| Localisation mean IoU | 0.074 |
| Tampered detection AUC | 0.589 |

## The known ceiling

Every backend available is trained on **whole images**, and model_03 asks it
about 64px crops. That mismatch is the largest single source of error in the
system, and no amount of compute fixes it — the table above is what it looks
like from the inside.

The fix is a patch-scale detector, and SID-Set's **train** split (210k images,
with masks) is public and sufficient to fine-tune one: sample patches, label
them by mask membership, fine-tune a 224px backbone on that distribution, and
register it as a backend. `mapper/backends.py` is written so that dropping one
in means adding a class there and changing nothing else. That is the one item on
this page where a stronger GPU pays for itself directly.

---

## Result files

`eval/report.py` regenerates all tables from the JSON, because numbers retyped
from a terminal drift from the runs that produced them. Each result file records
the backend (model, device, `id2label`, which index was read as AI and how that
was decided), the mapper config, and per-image rows — so a table can always be
traced back to the run that made it.
