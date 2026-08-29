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
│   ├── default.yaml            # full run
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

- All four test files pass (22 tests). The backbone tests run the real DINOv2 and
  open_clip architectures with random weights, so they check the plumbing without a
  1GB download; the FFT tests include a real-signal check (a nearest-neighbour
  upsampled image separates from a 1/f-spectrum image in the features designed to
  catch upsampling).
- Full pipeline end-to-end on real CIFAKE images with `dinov2`/`clip` disabled
  (FFT-only, no downloads): `extract_features.py` → `train.py` → `infer.py` →
  `eval/ablation.py` → `eval/robustness_eval.py` all run and produce output.

**Not yet verified:** a real run with DINOv2 and CLIP enabled — that needs the
weight downloads, and no accuracy number in this README is claimed from one. Do
that first, then fill in the ablation and robustness tables.

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
  as a pipeline check; the FFT block only gets a fair test on full-resolution data
  (SID_Set / WildFake).
