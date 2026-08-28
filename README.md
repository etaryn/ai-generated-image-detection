# Robust AIGC Image Detection

A prototype that distinguishes AI-generated images from authentic photos, built to stay
accurate after the kind of post-processing real redistribution applies: JPEG
re-compression, blur, resize round-trips, noise, color jitter, and cropping.

Built for TikTok TechJam Challenge 5 ("Robust Detection of AI-Generated Images Under
Real-World Transformations"). See `docs/technical-approach.md` for the full design
rationale.

## Project overview

Most AIGC detectors report strong numbers on clean, in-distribution test images and then
fall apart once an image has been re-uploaded to a platform, thumbnailed, filtered, or
screenshotted. This project treats "robust under transform" as the primary objective,
not an afterthought:

- **Model**: a CNN + Transformer hybrid, trained fully end-to-end (team's current
  architecture of choice). A convolutional stem (`model/cnn_stem.py`) picks up the
  local, low-level artifacts synthetic images tend to leave (upsampling checkerboard
  patterns, decoder texture, blending seams); its output feature grid is then fed as
  tokens into a Transformer encoder (`model/transformer_encoder.py`) whose
  self-attention reasons globally across regions, catching inconsistencies a purely
  convolutional model would miss. An optional frequency-domain (DCT/FFT) branch can be
  fused in for extra robustness — see `model/freq_branch.py`. A frozen-CLIP-backbone +
  linear-head variant is kept as a switchable baseline (`model.architecture:
  clip_frozen` in the config) — see `model/backbone.py`.
- **Training**: the training pipeline applies the same family of transforms the
  evaluation uses (JPEG quality 30–90, Gaussian blur σ 0.5–2.0, 0.25×–1× resize,
  Gaussian noise σ 0.02–0.10, ±20% color jitter, down to 80% center crop), so the model
  learns invariance to them instead of memorizing pristine-image statistics.
- **Evaluation**: reports a transform × severity metrics matrix, not just a single
  clean-image number, plus false-positive rate at a fixed operating threshold (flagging
  real content as fake is the costlier error for a moderation use case).

The CNN + Transformer hybrid's default channel/depth schedule totals roughly 14M
trainable parameters — far under the 2B-parameter limit, and trainable end-to-end on a
single GPU, though (having no pretrained component to lean on) it will likely need more
data and epochs than the frozen-CLIP baseline to reach the same accuracy.

## Repo structure

```
aigc-detector/
├── configs/
│   └── default.yaml        # paths, hyperparameters, transform severities
├── data/
│   ├── datasets.py         # PyTorch Dataset classes for the labeled image folders
│   ├── transforms.py       # the robustness augmentation pipeline (this is the core of the "robust" story)
│   ├── prepare_data.py     # dataset download / layout instructions
│   └── download_cifake.py  # CIFAKE: kagglehub download + auto-reorganize into real/fake folders
├── model/
│   ├── cnn_stem.py         # ConvStem: CNN feature extractor (primary architecture)
│   ├── transformer_encoder.py  # TokenTransformer: self-attention over the CNN's tokens
│   ├── backbone.py         # frozen CLIP vision encoder wrapper (alternative baseline)
│   ├── freq_branch.py      # optional frequency-domain (DCT) feature branch
│   ├── head.py             # small trainable MLP classification head
│   └── detector.py         # picks architecture from config, combines it (+ optional freq branch) + head
├── train.py                 # end-to-end training loop with clean/augmented consistency loss
├── infer.py                 # CLI: image directory in -> JSON [{"image_path", "pred"}] out
├── eval/
│   ├── metrics.py           # accuracy / AUC / F1 / FPR-at-threshold
│   ├── robustness_eval.py   # builds the transform x severity evaluation matrix
│   ├── error_analysis.py    # pulls representative false positives / false negatives
│   └── attention_rollout.py # explainability: rolls up self-attention into a saliency map
├── demo/
│   └── app.py               # minimal Gradio dashboard for the demo video
├── tests/
│   ├── test_transforms.py   # torch-free tests for the augmentation pipeline (runs anywhere)
│   └── test_model_shapes.py # model forward/backward-pass + attention-rollout shape tests (needs torch)
└── notebooks/
    └── exploration.ipynb    # scratch space for data exploration (not required)
```

## Setup and installation

```bash
git clone <this-repo-url>
cd aigc-detector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA GPU is recommended for training the CNN + Transformer
hybrid end-to-end; CPU inference is feasible given the model's small size (~14M
params).

## Steps to reproduce results

1. **Get the data.** Run `python data/prepare_data.py --help` for pointers on laying out
   each dataset (SID_Set, CIFAKE, WildFake) into `data/raw/<dataset>/{real,fake}/`. See
   the TODOs in that file — dataset licenses/download mechanics differ, so this step is
   deliberately manual/semi-automated rather than a single script.
   - **CIFAKE** is the one exception with a fully automated path:
     `python data/download_cifake.py --out data/raw/cifake` downloads it via
     kagglehub and reorganizes it into the expected `real/`/`fake/` layout in one
     step (add `--copy` if your filesystem doesn't support symlinks).
   - Do **not** train on the WildFake (COCO val2017 / DALL·E Advanced) validation subset
     — it's reserved for demonstration/tracking only, per the challenge rules.
   - **No data yet, or just want to sanity-check the pipeline?** Run
     `python data/make_synthetic_dataset.py --out data/raw/synthetic --n_per_class 200`
     to generate a network-free synthetic dataset (smooth "scenes" as "real", the same
     scenes with a subtle periodic upsampling-style artifact added as "fake") and set
     `train_datasets: ["synthetic"]` in the config. This exercises the whole pipeline
     end-to-end but is **not a substitute for real data** — a model trained only on it
     will just learn to detect the synthetic artifact pattern, nothing more.
2. **Train the model.**
   ```bash
   python train.py --config configs/default.yaml
   ```
   With the default `model.architecture: cnn_transformer`, this trains the CNN stem +
   Transformer encoder + head end-to-end (and the optional frequency branch, if
   enabled). Each training sample is loaded as a genuine clean/augmented pair (see
   `data/datasets.py`'s `PairedViewDataset`): both views are classified, and a
   consistency loss penalizes the model for predicting differently on the clean vs.
   redistributed copy of the same image -- directly optimizing the "robust under
   transform" objective rather than hoping augmentation exposure alone produces it.
   Switch to `model.architecture: clip_frozen` in the config to instead freeze a
   pretrained CLIP backbone and train only the head.
3. **Run the robustness evaluation.**
   ```bash
   python eval/robustness_eval.py --config configs/default.yaml --checkpoint <path>
   ```
   Produces the clean-vs-transformed metrics matrix (`eval/robustness_matrix.csv` +
   a plotted heatmap) used in the Robustness Evaluation Summary deliverable.
4. **Run error analysis.**
   ```bash
   python eval/error_analysis.py --config configs/default.yaml --checkpoint <path>
   ```
   Dumps representative false positives / false negatives with their predicted scores.
5. **Run inference on a directory of images** (the required deliverable script):
   ```bash
   python infer.py --input_dir /path/to/images --checkpoint <path> --output predictions.json
   ```
   Output format:
   ```json
   [
     {"image_path": "/path/to/images/img001.jpg", "pred": 0.93},
     {"image_path": "/path/to/images/img002.jpg", "pred": 0.07}
   ]
   ```
   `pred` is the model's confidence that the image is AI-generated, in `[0, 1]`.

6. **(Optional) Explain a single prediction** via attention rollout (CNN + Transformer
   architecture only):
   ```bash
   python eval/attention_rollout.py --checkpoint <path> --image path/to/image.jpg --output rollout.png
   ```
   Rolls up the Transformer's per-layer self-attention into a single saliency map
   showing which image regions most influenced the [CLS] token used for the final
   decision (Abnar & Zuidema, "Quantifying Attention Flow in Transformers", 2020).

## Multi-machine distributed training (optional)

If you have multiple machines with GPUs (e.g. three teammates' laptops) and want to
combine them into **one faster training run** rather than running three separate
experiments, `train_ddp.py` wraps `train.py`'s training loop in PyTorch
`DistributedDataParallel` (DDP). Each machine runs one process, the dataset is split
across them with a `DistributedSampler` (no two machines train on the same images in
a given epoch), and gradients are synchronized every step so it behaves like one
larger-batch run on a single machine, just faster.

**Prerequisites (all machines):**

- The **same repo** (same commit) and the **same dataset** laid out at the same
  relative path (`data/raw/...`) on every machine.
- The **same Python/torch environment** (same `requirements.txt` installed) on every
  machine -- DDP is sensitive to version mismatches.
- All machines able to reach each other over the network on one TCP port (default
  `29500`). If they're not already on the same LAN/Wi-Fi, install a mesh VPN like
  Tailscale on all three laptops first -- it's free for personal use and gives each
  machine a stable IP that works across networks/firewalls without manual
  port-forwarding.
- Pick one machine as the "master" (rank 0) and note its Tailscale/LAN IP address.

**Running it.** On each of the 3 laptops, open a terminal in the repo and run (only
`--node_rank` changes between machines):

```bash
# Laptop 1 (master, rank 0) -- replace 100.x.y.z with its Tailscale/LAN IP
torchrun --nnodes=3 --nproc_per_node=1 --node_rank=0 \
  --master_addr=100.x.y.z --master_port=29500 train_ddp.py --config configs/default.yaml

# Laptop 2 (rank 1) -- same master_addr as above
torchrun --nnodes=3 --nproc_per_node=1 --node_rank=1 \
  --master_addr=100.x.y.z --master_port=29500 train_ddp.py --config configs/default.yaml

# Laptop 3 (rank 2) -- same master_addr as above
torchrun --nnodes=3 --nproc_per_node=1 --node_rank=2 \
  --master_addr=100.x.y.z --master_port=29500 train_ddp.py --config configs/default.yaml
```

Start the master (rank 0) first; the other two will wait until they can connect to it.
Only rank 0 evaluates, prints progress, writes `training_log.csv`, and saves
`checkpoints/best.pt` -- you only need to watch that machine's terminal for results.

Uses the `gloo` backend (not `nccl`) specifically so this works across mixed
OSes/GPU vendors (Windows, Mac, Linux, NVIDIA or not) rather than requiring all three
machines to be Linux+NVIDIA.

If this setup work sounds like more than it's worth for your timeline, the simpler
alternative is running three **independent** experiments (different configs/seeds, one
per laptop) with plain `train.py` and comparing results -- no networking required.

## Running the tests

```bash
python tests/test_transforms.py          # torch-free; verifies the augmentation pipeline itself
python tests/test_synthetic_dataset.py   # torch-free; verifies the network-free synthetic dataset generator
python tests/test_download_cifake.py     # torch-free; verifies the CIFAKE reorganization logic (no network needed)
python tests/test_model_shapes.py        # needs torch; forward/backward-pass + attention-rollout shape checks
```

Both scripts are also plain `pytest`-discoverable (`pytest tests/`) if you have pytest
installed. `test_transforms.py` has been run and passes in this project's dev
environment; `test_model_shapes.py` has only been verified by hand (shape/parameter-count
arithmetic) since no torch install was available when it was written -- run it first in
your real environment, before starting a full training run.

## Limitations and future work

- The CNN + Transformer hybrid trains fully from scratch, so it has no pretrained
  prior to fall back on -- expect it to need more data and/or epochs than the
  frozen-CLIP baseline to reach comparable accuracy, and to be more sensitive to
  training-set size and diversity. The `clip_frozen` config path is kept specifically
  as a fallback/comparison point if training data or time turns out to be too limited
  for the from-scratch model to converge well.
- The frequency-domain branch is implemented but not exhaustively tuned — it's included
  as an ablation/stretch component rather than a fully validated production feature.
- Robustness is evaluated against the transform list in the challenge brief; other
  real-world manipulations (heavy meme-text overlays, screenshots-of-screenshots,
  aggressive AI upscaling/inpainting touch-ups) are not covered and would need their own
  augmentation + eval passes.
- Given more time: calibrate the decision threshold against a target false-positive
  budget rather than a fixed 0.5 cutoff, expand training data diversity (more generator
  families beyond what's in SID_Set/CIFAKE/WildFake), and add a lightweight
  active-learning loop to mine hard examples from the error analysis step back into
  training.
- The model code was written and shape/parameter-count-verified by hand (and unit
  tested where torch-free, e.g. `tests/test_transforms.py`) in an environment without
  network access to install torch, so `tests/test_model_shapes.py` -- which actually
  runs a forward/backward pass -- has not yet been executed for real. Run it first
  thing in your training environment; see "Running the tests" above.

## Team member contributions

_Fill in per team member once roles are assigned, e.g.:_

| Member | Contribution |
|---|---|
| — | Data pipeline & augmentation |
| — | Model architecture & training |
| — | Evaluation harness & error analysis |
| — | Demo / dashboard / video |
