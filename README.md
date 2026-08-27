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
│   └── prepare_data.py     # dataset download / layout helpers
├── model/
│   ├── cnn_stem.py         # ConvStem: CNN feature extractor (primary architecture)
│   ├── transformer_encoder.py  # TokenTransformer: self-attention over the CNN's tokens
│   ├── backbone.py         # frozen CLIP vision encoder wrapper (alternative baseline)
│   ├── freq_branch.py      # optional frequency-domain (DCT) feature branch
│   ├── head.py             # small trainable MLP classification head
│   └── detector.py         # picks architecture from config, combines it (+ optional freq branch) + head
├── train.py                 # training loop (trains only the head / freq branch)
├── infer.py                 # CLI: image directory in -> JSON [{"image_path", "pred"}] out
├── eval/
│   ├── metrics.py          # accuracy / AUC / F1 / FPR-at-threshold
│   ├── robustness_eval.py  # builds the transform x severity evaluation matrix
│   └── error_analysis.py   # pulls representative false positives / false negatives
├── demo/
│   └── app.py               # minimal Gradio dashboard for the demo video
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
   - Do **not** train on the WildFake (COCO val2017 / DALL·E Advanced) validation subset
     — it's reserved for demonstration/tracking only, per the challenge rules.
2. **Train the model.**
   ```bash
   python train.py --config configs/default.yaml
   ```
   With the default `model.architecture: cnn_transformer`, this trains the CNN stem +
   Transformer encoder + head end-to-end (and the optional frequency branch, if
   enabled) with the robustness-augmentation pipeline from `data/transforms.py` applied
   on the fly. Switch to `model.architecture: clip_frozen` in the config to instead
   freeze a pretrained CLIP backbone and train only the head.
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

## Team member contributions

_Fill in per team member once roles are assigned, e.g.:_

| Member | Contribution |
|---|---|
| — | Data pipeline & augmentation |
| — | Model architecture & training |
| — | Evaluation harness & error analysis |
| — | Demo / dashboard / video |
