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

**Do not expect a bigger GPU alone to help much.** Measured on the reference
machine, the pipeline runs at **21% GPU utilisation** — it is bottlenecked on
CPU-side patch cropping and `AutoImageProcessor` resizing (~1250 PIL crops per
image), not on matrix multiplication. Raising `batch_size` and increasing
dataloader parallelism will buy more than more FLOPs. The genuinely
GPU-hungry item is not on this page: fine-tuning a patch-scale detector (see
"The known ceiling").

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
