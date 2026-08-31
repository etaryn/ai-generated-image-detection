# model_02 — frozen features → small classifier

A second architecture for the same problem model_01 attacks (robust detection of
AI-generated images under real-world transformations), built on the opposite bet.

**model_01** trains a CNN + Transformer end to end: it learns its own
representation of what a generated image looks like.
**model_02** learns no representation at all. It runs three *frozen*, complementary
descriptions of an image, concatenates them, and trains a small classifier on top:

```
                  ┌──────────────────────────────────────────┐
                  │  Step 1 — feature extraction (frozen)    │
  image ───┬──────│  DINOv2   → texture / structure          │──┐
           ├──────│  CLIP     → semantic meaning / logic     │──┼── concat ──┐
           └──────│  FFT math → microscopic pixel noise      │──┘            │
                  └──────────────────────────────────────────┘               │
                                                                             ▼
                                       ┌──────────────────────────────────────────┐
                                       │  Step 2 — classifier (the only trained   │
                                       │  part): MLP or XGBoost → 0 real / 1 AI   │
                                       └──────────────────────────────────────────┘
```

Why this is worth having alongside model_01:

- **Three independent kinds of evidence.** A generator has to fool a texture model,
  a semantic model *and* the noise floor simultaneously. Failing any one is enough.
- **It trains in minutes.** Extraction happens once and is cached to disk; after
  that, every experiment (classifier type, hyperparameters, feature ablation) reads
  a dense matrix, so iteration cost is measured in minutes, not GPU-hours.
- **It generalizes differently.** Frozen CLIP features are the strongest known
  baseline for *cross-generator* transfer (Ojha et al., CVPR 2023) — they degrade
  less on generator families absent from training, which is exactly what the
  challenge's held-out demo set probes.
- **It's auditable.** The FFT block's columns have names and meanings, so XGBoost's
  gain table says which physical artifact the detector is keying on
  (`eval/ablation.py`, and `--blocks` on `train.py`).

Trainable parameters: ~1.2M (the MLP head). Frozen: ~173M (DINOv2-base 86M + CLIP
ViT-B/16 vision tower 86M) — far under the challenge's 2B limit.

---

## Step 1: what the three branches actually measure

| Branch | Module | Default width | What it describes |
|---|---|---|---|
| DINOv2 | `features/dino.py` | 1536 (CLS 768 + patch-mean 768) | **Texture / structure.** Self-supervised, so it was never trained to name objects — its embedding encodes *how an image is built*: surface texture, material, part structure. Diffusion output tends to be over-smooth and structurally subtly-off exactly here. |
| CLIP | `features/clip.py` | 512 (ViT-B/16 projection) | **Semantics / logic.** Trained to align images with text, so it encodes what the picture *means* and whether that hangs together — implausible object combinations, non-language text, prompt-shaped composition. |
| FFT | `features/fft.py` | 130 | **Microscopic pixel noise.** Hand-derived, zero weights. A photo's high-frequency content is sensor noise: near-isotropic, near-1/f, with demosaicing-set channel correlations. A decoder manufactures its high frequencies by repeated upsampling, leaving periodic peaks at fractions of Nyquist, axis-aligned bias, and unnatural channel correlation. |

Default vector: **1536 + 512 + 130 = 2178** numbers per image.

The FFT block in detail (all computed on a high-pass residual — image minus its
blurred self — so it describes the noise floor rather than the scene):

| Count | Feature |
|---|---|
| 32 | radial log-power profile, mean-centered (isotropic spectrum shape) |
| 1 + 3 | mean log power; power-law fit slope / intercept / RMSE |
| 16 | azimuthal log-power profile, mean-centered (directional / grid bias) |
| 4 | half- and quarter-Nyquist peak scores, high-frequency band ratio, peakiness |
| 3 | residual std / skew / kurtosis |
| 4 | residual autocorrelation at lags 1 and 2, both axes |
| 3 | cross-channel residual correlation (R·G, R·B, G·B) |
| 64 | 8×8 block-DCT mean log-magnitude profile (JPEG grid + decoder fingerprint) |

## Step 2: the classifier

`classifier.type` selects one; both train on the same cache, so switching is a
config change:

- **`mlp`** — two hidden layers, BatchNorm + dropout, AdamW, early stopping on
  validation AUC (threshold-free, so the selected epoch isn't an artifact of a 0.5
  cutoff). Best when the dense embedding blocks dominate.
- **`xgboost`** — gradient-boosted trees. Usually stronger when the FFT block is
  carrying the signal (heterogeneous hand-built statistics on different scales are
  what axis-aligned splits are *for*), and it hands back per-feature gain
  importances that read directly against `features/fft.py`'s column names.

---

## Layout

```
model_02/
├── configs/
│   ├── default.yaml            # full run (CIFAKE, 32x32)
│   ├── generators.yaml         # full-resolution, multi-generator run (see below)
│   └── quick_smoke.yaml        # tiny end-to-end check (separate cache + checkpoint dirs)
├── features/
│   ├── base.py                 # extractor interface + shared resize/normalize
│   ├── dino.py                 # Step 1a  texture / structure
│   ├── clip.py                 # Step 1b  semantics / logic
│   ├── fft.py                  # Step 1c  pixel noise (no weights)
│   └── pipeline.py             # FeatureStack: build, concat, block spec
├── classifiers/
│   ├── mlp.py                  # torch MLP
│   └── xgb.py                  # XGBoost
├── eval/
│   ├── robustness_eval.py      # transform × severity matrix (CSV + heatmap)
│   └── ablation.py             # what each branch is actually worth
├── tests/                      # run these before a long extraction
├── data_io.py                  # canonical [0,1] tensor loading, group ids
├── shared.py                   # bridge to model_01's data/augmentation/metrics
├── prepare_generators.py       # Step 0 driver → multi-generator data at native resolution
├── extract_features.py         # Step 1 driver → cached .npz
├── train.py                    # Step 2 driver → checkpoint
└── infer.py                    # required deliverable: dir of images → predictions.json
```

`shared.py` loads model_01's `data/datasets.py`, `data/transforms.py` and
`eval/metrics.py` **by file path** rather than copying them. Both models therefore
train on the same folder layout, augment with the same transform family, and
report metrics computed by the same code — otherwise the two models' numbers
wouldn't be comparable. (File-path loading, not `sys.path`, because both models
have a top-level `eval/` package and would shadow each other.)

## Quickstart

Run everything from this directory. Datasets are read from `../../data/raw/` —
see `../model_01/data/prepare_data.py` for how to populate it.

```bash
pip install -r requirements.txt

# 0. verify the plumbing first (~30s, no weight downloads, no data needed)
python tests/test_fft_features.py
python tests/test_backbone_extractors.py
python tests/test_pipeline.py
python tests/test_train_roundtrip.py

# 1. extract features once (the expensive step; downloads DINOv2 + CLIP weights on
#    first run). --limit keeps a smoke run cheap.
python extract_features.py --config configs/quick_smoke.yaml --limit 2000
python extract_features.py --config configs/default.yaml

# 2. train the classifier on the cache (minutes, re-runnable)
python train.py --config configs/default.yaml
python train.py --config configs/default.yaml --classifier xgboost

# 3. score a directory of images -> predictions.json  (same schema as model_01)
python infer.py --input_dir /path/to/images --checkpoint checkpoints/best.pt \
    --output predictions.json

# 4. evidence for the writeup
python eval/ablation.py --config configs/default.yaml          # what each branch is worth
python eval/robustness_eval.py --config configs/default.yaml \
    --checkpoint checkpoints/best.pt --limit 2000              # severity matrix
```

`infer.py` output matches model_01's exactly, so the two are drop-in comparable:

```json
[
  {"image_path": "/path/to/images/img001.jpg", "pred": 0.93},
  {"image_path": "/path/to/images/img002.jpg", "pred": 0.07}
]
```

## Full-resolution, multi-generator data (DALL·E 3 / MidJourney / GAN)

CIFAKE is one generator family at 32×32. Cross-generator transfer — the thing the
challenge actually probes — cannot be measured on it at all. `prepare_generators.py`
builds three paired datasets at native resolution:

| Folder | Fake | Real | Source resolution |
|---|---|---|---|
| `gen_progan` | ProGAN (ForenSynths) | LSUN, matched pairs | 256² both sides |
| `gen_midjourney` | MidJourney (GenImage) | Open Images v7 | ~1024px both sides |
| `gen_dalle3` | DALL·E 3 | Open Images v7 (disjoint shards) | ~1024px both sides |

```bash
# Downloads + crops. Must run off the login node: `ulimit -v` there is 1GB and
# both the HF transfer backend and pyarrow exceed it instantly.
sbatch ../../prepare_generators.sbatch          # or: python prepare_generators.py --families all
```

**This dataset is adversarial to build, and most of the script is about that.**
The naive version is separable at AUC > 0.95 without looking at a single generator
artifact, via two independent shortcuts:

- **Resolution.** Every generated source is a fixed square (1024² or 128²); every
  real corpus is variable (~500×375, 640×480). So every image is written as a
  **center crop taken at native resolution** — never resized. Crop offsets are
  snapped to multiples of 8 to preserve JPEG grid phase, because 64 of `fft.py`'s
  130 columns are a block-DCT profile keyed to that grid.
- **Compression history.** GenImage's MidJourney split and the DALL·E 3 dataset both
  ship *losslessly as PNG*; every real-photo corpus ships as JPEG. Left alone, "was
  this ever JPEG-compressed?" is a near-perfect stand-in for the label — and it is
  exactly what those 64 columns measure. Every crop, both classes, is therefore
  written through one identical JPEG encode.

Scale matters as much as content when picking the real half. MidJourney is paired
with Open Images rather than ImageNet even though ImageNet is GenImage's own real
counterpart and the better *content* match: a 256px window covers 1/16 of a 1024²
render but most of a 500×375 photo, so the two differ in magnification and texture
density. Residual standard deviation alone separates MidJourney from ImageNet at
**AUC 0.976** — one trivial scalar, no generator evidence involved. Against Open
Images the same scalar drops to **0.64**.

Measured on the delivered data, 250 images per class, single-scalar AUC (0.50 = no
shortcut):

| Family | resid_std | hf_power | hf_ratio | detail | blockiness | worst |
|---|---|---|---|---|---|---|
| `gen_progan` | 0.512 | 0.552 | 0.563 | 0.547 | 0.541 | **0.563** |
| `gen_midjourney` | 0.643 | 0.600 | 0.553 | 0.595 | 0.684 | **0.684** |
| `gen_dalle3` | 0.662 | 0.745 | 0.776 | 0.667 | 0.505 | **0.776** |

ProGAN's matched LSUN pairs are effectively at chance, which is why it was chosen
over GenImage's 128px BigGAN. DALL·E 3 retains the most residual structure (0.776 on
`hf_ratio`); some of that is real generator signal and some is domain gap, and this
probe cannot separate the two — which is what `eval/ablation.py` is for.

Re-run the check after adding any source; the run also prints a per-class
source-format census so a new asymmetry can't slip in unnoticed:

```bash
python prepare_generators.py --families <name> --per-class 250
```

### Leave-one-generator-out

Extraction is the expensive step, so extract each family **once** and recombine the
caches at train time (`train.py --cache` takes several and offsets their group ids):

```bash
for f in progan midjourney dalle3; do
  python extract_features.py --config configs/generators.yaml \
      --datasets gen_$f --out features/cache/gen_$f.npz
done

# in-distribution
python train.py --config configs/generators.yaml \
    --cache features/cache/gen_{progan,midjourney,dalle3}.npz

# held out: train on two families, score the third
python train.py --config configs/generators.yaml \
    --cache features/cache/gen_{progan,midjourney}.npz --out checkpoints/loo_dalle3.pt
```

### Results (job 774674, MLP, clean rows only)

Each fold trains on two families and is scored on the third, which it has never seen.
Held-out scoring reads the third family's cache directly (`eval/score_cache.py`), so
no image is re-encoded.

| Held-out family | Held-out AUC | Held-out acc | F1 | FPR | In-dist. AUC | In-dist. acc |
|---|---|---|---|---|---|---|
| `gen_progan` | 0.686 | 0.511 | 0.103 | 0.034 | 1.000 | 0.9995 |
| `gen_midjourney` | 0.850 | 0.650 | 0.483 | 0.027 | 1.000 | 0.9989 |
| `gen_dalle3` | 0.822 | 0.562 | 0.242 | 0.015 | 1.000 | 0.9979 |

**The gap is the result.** In-distribution the classifier is essentially perfect —
AUC 1.0000 on all three families, and the fold's own validation split agrees
(0.996–0.997). On a generator it has never seen, AUC falls to 0.69–0.85 and accuracy
to near chance.

Read the F1 and FPR columns together and the failure mode is specific: FPR stays very
low (1.5–3.4%) while F1 collapses. The model is not confused on unseen generators — it
confidently calls them **real**. An unfamiliar generator's features simply don't
resemble the fakes it was trained on, so they land on the real side of the boundary.
That is the costly direction of error for moderation, and it is invisible in any
single-generator evaluation.

Two practical consequences:

- **The 0.5 threshold does not transfer.** In-distribution the 5%-FPR threshold
  calibrates to 0.05–0.12; on held-out generators the same cutoff is far too high.
  Ranking survives transfer considerably better than calibration does, which is why
  AUC (0.69–0.85) reads so much better than accuracy (0.51–0.65).
- **Near-perfect in-distribution numbers mean very little here.** The CIFAKE result
  (0.9979 val accuracy) and these 1.0000 AUCs are the same kind of number, and this
  table is what they are worth against an unseen generator.

ProGAN transfers worst (0.686), which fits: it is the only GAN in the set, so its
fold trains on two diffusion families and is tested on a different generator *class*
entirely. MidJourney transfers best (0.850) — its fold trains on DALL·E 3, the
nearest relative in the set.

## Design decisions worth knowing

**Robustness comes from augmented copies, not a consistency loss.** model_01 pairs
each image with an augmented view and penalizes disagreement between them. There's
no equivalent hook here — the extractors are frozen, so there's no end-to-end loss
to attach a consistency term to. Instead `features.train_aug_copies` writes N extra
rows per image, each an independently randomized pass of the same transform family
(JPEG 30–90, blur σ 0.5–2.0, 0.25×–1× resize, noise σ 0.02–0.10, ±20% color jitter,
80% center crop), so the classifier sees pristine and redistributed versions of the
same image and has to agree with itself.

**The train/val split is by source image, not by row.** Augmented copies inherit
their original's `group_id` and `train.py` splits on groups. Splitting on rows would
put a JPEG-recompressed copy of a training image into validation and report a val
score inflated by near-duplicate leakage.

**One decode, three extractors.** `data_io.py` produces a single canonical tensor
(float `[0,1]`, `canonical_size²`, un-normalized); DINOv2 and CLIP resample and
normalize from that themselves, and the FFT block reads it directly. Resampling
uses `antialias=True` throughout — without it, our own preprocessing would inject
the exact kind of high-frequency artifact the FFT block is meant to detect in the
*generator*.

**The checkpoint is self-contained.** It carries the scaler, the classifier, the
feature config, the block layout and the trained column selection, so `infer.py`
takes no feature flags: it rebuilds training-time preprocessing from the file.

**A threshold calibrated to a false-positive budget is saved alongside 0.5.**
Flagging real content as AI-generated is the costlier error for moderation, so
`train.py` also records the threshold meeting `eval.target_fpr` on validation;
`infer.py --use_calibrated_threshold` uses it.

## Verification status

Verified by running, in this repo's `.venv` (torch 2.13 + CUDA, transformers,
open_clip):

- All five test files pass (28 tests). The backbone tests run the real DINOv2 and
  open_clip architectures with random weights, so they check the plumbing without a
  1GB download; the FFT tests include a real-signal check (a nearest-neighbour
  upsampled image separates from a 1/f-spectrum image in the features designed to
  catch upsampling).
- Full pipeline end-to-end on real CIFAKE images with `dinov2`/`clip` disabled
  (FFT-only, no downloads): `extract_features.py` → `train.py` → `infer.py` →
  `eval/ablation.py` → `eval/robustness_eval.py` all run and produce output.
- **Full run with DINOv2 + CLIP enabled, on real weights** (job 774674, A100): all
  three generator families extracted (56,000 rows × 2178 features), four classifiers
  trained, and every leave-one-out fold scored against its held-out family. The
  numbers in the results table above come from that run.

**Watch out — `configs/default.yaml` has a CLIP defect.** It names
`backbone_name: "ViT-B-16"` with `pretrained: "openai"`. Since open_clip 2.24 the
plain entry builds the model with standard `nn.GELU`, while OpenAI's weights were
trained with QuickGELU, so open_clip warns `QuickGELU mismatch ... (quick_gelu=False)
and pretrained tag 'openai' (quick_gelu=True)` and silently degrades the 512 CLIP
dimensions rather than failing. `configs/generators.yaml` uses
`"ViT-B-16-quickgelu"` and `extract_generators.sbatch` preflights it. `default.yaml`
was deliberately left as-is so the existing CIFAKE checkpoint stays reproducible —
fix it there together with a re-extraction, and expect its numbers to move.

**Still not verified:** the ablation and robustness matrices on this data
(`eval/ablation.py`, `eval/robustness_eval.py`). The ablation matters most — it is
what separates genuine spectral evidence from the residual domain gap the
`gen_dalle3` / `gen_midjourney` pairings still carry.

## Limitations

- **The FFT branch is the fragile one.** JPEG recompression, blur and resizing
  directly overwrite the high-frequency evidence it reads. Expect it to degrade
  hardest in the robustness matrix; the honest way to show what it's worth after
  redistribution is to compare against an FFT-free checkpoint
  (`train.py --blocks dino clip`).
- **Frozen features cap the ceiling.** Nothing adapts to the task; if the artifact
  that separates a generator from a photo isn't visible in DINOv2's, CLIP's or the
  spectral descriptors' output, no classifier on top can recover it. That's the
  trade for the training speed and the cross-generator transfer.
- **Extraction dominates cost at inference too.** Scoring one image means two ViT
  forward passes plus the spectral statistics — heavier per image than model_01,
  even though training is far cheaper.
- **CIFAKE is 32×32.** Upsampling CIFAR-scale images to 224 leaves the spectral
  branch very little genuine high-frequency content to read. Treat CIFAKE numbers
  as a pipeline check. `configs/generators.yaml` is the fair test — 256px native
  crops across three generator families, where the FFT block finally has a real
  high-frequency band to read.
- **Residual content mismatch in two of the three families.** Cropping and uniform
  re-encoding remove the resolution and compression shortcuts, but MidJourney and
  DALL·E 3 are still paired with a real corpus of different subject matter, and their
  worst single-scalar AUC sits at 0.68 / 0.78 against ProGAN's 0.56. Some of that is
  genuine generator signal and some is domain gap; the two are not separated. Read
  `eval/ablation.py` against the FFT-free checkpoint before attributing a result to
  spectral evidence, and treat ProGAN as the cleanest of the three.
- **Only the fake half is generator-diverse.** Two of the three families draw their
  reals from the same corpus (disjoint Open Images shards). Generalization across
  *real* image domains — phone cameras, film scans, screenshots — is still untested.
