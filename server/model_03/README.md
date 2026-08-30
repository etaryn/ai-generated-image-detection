# model_03 — region-aware AI tamper forensics

**Find first, analyse second.** model_01 and model_02 both answer one question
about a whole image: *is this AI-generated?* model_03 asks a different one:
*where in this image is there evidence, what kind is it, and how much should we
trust it?*

That matters because AI tampering is usually local. A photograph with one
inpainted object is mostly authentic, so a whole-image detector averages the
evidence away — and, worse, it cannot tell a fully synthetic picture apart from
a real photograph with a generated patch in it, which are very different
findings for anyone who has to act on them.

```
Input image
   |
   v
[1] Multi-scale AI-likelihood mapper          mapper/
   |  calibrated dense heatmap: likely AI / likely non-AI / uncertain
   v
[2] Region proposal and confidence filtering  regions/
   |  connected suspicious areas, boundaries, region descriptors
   v
[3] Region-aware routing                      router.py, specialists/
   |--- generative-fill / inpainting specialist
   |--- whole-image synthesis specialist
   |--- face-edit specialist            (only when a face detector is installed)
   |--- conventional compositing specialist
   `--- general fallback for uncertain regions
   v
[4] Evidence fusion                           fusion.py
   |  regional scores + spatial context + uncertainty, conservatively
   v
Output: tamper map, regional labels, image verdict, confidence, explanation
```

model_03 **trains nothing and ships no weights.** It scores patches with
model_01 or model_02 and builds everything else on top. That is what makes it a
prototype of the *architecture* rather than a fourth detector.

---

## What is learned and what is not

This is the first thing to know about the prototype, because it determines what
its outputs are worth.

| Stage | What it is | Trained? |
|---|---|---|
| Patch scoring | model_01 or model_02, whole-image detectors asked about patches | **Yes** — but on whole images, not patches |
| Calibration | Platt / isotonic map fitted on patch scores | **Yes**, when you fit one (`scripts/calibrate_mapper.py`); identity otherwise |
| Blending, smoothing, labelling | Hann-weighted accumulation, guided filter, two thresholds | No — deterministic |
| Region proposal | Connected components + geometric descriptors | No |
| Routing | Six readable rules over region descriptors | No — deliberately, see `router.py` |
| Specialists | Hand-derived forensic statistics + detector re-scoring | No |
| Fusion | Shrinkage, support ceiling, noisy-OR across regions | No |

So the honest summary: **the localisation machinery is real and tested; the
specialists are classical forensics, not trained detectors; and the quality of
everything is bounded by the patch scorer underneath.**

The face specialist is the one to read most carefully. It is **not** a deepfake
detector — the project has no face-swap model, and substituting a general check
under that name would over-claim on exactly the finding users are most likely to
act on. It reports blending and noise-floor evidence on a face-shaped region,
labels its findings `face_region_edit_evidence` rather than "deepfake", and its
confidence is hard-capped at 0.55.

---

## Quick start

```bash
pip install -r server/model_03/requirements.txt   # numpy, Pillow (+ optional scipy, cv2)
pip install -r server/requirements.txt            # the model_01 backend: torch, torchvision

cd server/model_03
python infer.py --input_dir /path/to/images --output predictions.json
```

`predictions.json` matches model_01 and model_02 exactly, so all three are
drop-in comparable on the same harness:

```json
[
  {"image_path": "/path/to/images/img001.jpg", "pred": 0.93},
  {"image_path": "/path/to/images/img002.jpg", "pred": 0.07}
]
```

For the part model_03 exists for, ask for the region report and the overlays:

```bash
python infer.py --input_dir imgs --report report.json --render_dir overlays/
```

`report.json` carries, per image: the verdict, score, confidence, a written
explanation, and for each region its bounding box, label, probability, the
specialist it was routed to, **why** it was routed there, the descriptors that
decided it, and the evidence strings the specialist produced. `overlays/`
gets a heatmap, an outlined-regions image and a three-panel comparison per input.

In Python (the same API the siblings expose, plus one):

```python
from infer import load_model, predict_image, analyze_image

load_model()                      # warm the backend at startup, not on first upload
predict_image(pil_image)          # -> float, the fused P(AI-generated)
report = analyze_image(pil_image) # -> the whole analysis
report.verdict.verdict            # 'ai_generated' | 'ai_edited' | ...
report.verdict.explanation        # a paragraph you can check against the overlay
report.to_dict()                  # JSON-clean
```

The Streamlit bench (`client/app.py`) lists model_03 alongside the others; pick
it and the UI adds the likelihood map, the outlined regions, the
likely-AI/uncertain/likely-non-AI split, and each region's evidence.

### Choosing the patch scorer

```bash
python infer.py --input_dir imgs --backend model_02       # or: export AIGC_MODEL03_BACKEND=model_02
```

`model_01` is the default: its weights are in the repo and it is fast enough to
score ~1200 patches in about a second. `model_02` reads the noise floor directly
via its FFT block, which is the evidence a *local* edit actually leaves, but it
runs two ViT passes per patch — with it, drop the 64px scale (see below).

---

## The four layers

### [1] The AI-likelihood mapper (`mapper/`)

Windows at three scales with 50% overlap (`windows.py`), each patch scored by
the backend (`backends.py`), calibrated (`calibration.py`), then projected back
with a **centre-weighted Hann kernel** (`blend.py`) so a pixel is described
mostly by the patches that see it centrally. Flat-painting each patch instead
produces a heatmap made of rectangles, and region extraction then finds the
rectangles rather than the tampered object — the blocking is not cosmetic, it
corrupts everything downstream. Smoothing is a **guided filter** keyed on image
luminance (`smooth.py`), so window jitter is flattened without smearing the
region boundaries whose sharpness the router reads.

Two design decisions worth knowing:

**Scales combine with `max`, not `mean`.** A coarse window containing a 200px
edit inside a 224px field of authentic pixels reports their average — by
construction, not by disagreement. Averaging the scales treats that
structurally-guaranteed low reading as evidence *against* the edit. Measured on
the synthetic case in `tests/test_pipeline.py`: the fine scale peaks at 0.86
inside a 192px edit, the coarse scale cannot exceed 0.48, and their mean lands
at 0.63 — below any threshold worth setting, so the edit vanishes. A high score
at any scale is evidence; a low score at a coarser scale is not counter-evidence.

**Labels are three-valued and stay that way.** Above the high threshold is
"likely AI", below the low one is "likely non-AI", and in between is
`uncertain` — carried forward rather than rounded off. Pixels the windows barely
covered (the frame edge) are demoted to uncertain rather than trusted.

### [2] Region proposals (`regions/`)

Connected components over the confident core, grown into the adjacent uncertain
band (never into confident non-AI), then described: area, score statistics,
per-scale profile, scale disagreement, boundary sharpness, compactness, fill
ratio, uncertain halo, border contact, window support. Regions are ranked by
evidence mass — area × strength — so one large moderate region outranks three
speckles that happened to peak.

### [3] Routing and specialists (`router.py`, `specialists/`)

The router is six rules on cheap descriptors, and it is rules on purpose:
routing errors are expensive (measure a region against the wrong hypothesis and
the evidence means nothing), and a rule you can read is a rule you can check
against a heatmap you are looking at. Every decision carries the reason string
that lands in the report.

Each specialist returns a probability, a **confidence**, evidence strings, and
optionally a tighter mask. Confidence is the load-bearing field — it is what
lets fusion be conservative without being deaf. Their forensic measurements
(`forensics.py`) are all *local* contrasts, region against the ring immediately
outside it: residual noise level, fine-detail energy, JPEG 8×8 lattice
coherence, error-level analysis, cross-channel noise correlation, rim
straightness. Comparing against the whole image instead would flag every dark
corner.

The compositing specialist exists so the system can say "manipulated, but this
looks like conventional editing rather than a generator" — calling a Photoshop
splice "AI-generated" is the expensive false positive here.

### [4] Fusion (`fusion.py`)

Three mechanisms enforce the report's rule that an uncertain map must not become
a definitive verdict because one specialist shouted:

1. **Shrinkage** — a specialist moves the region's score in proportion to its
   own confidence. At confidence 0 the map's opinion stands.
2. **A ceiling from map support** — a region the map was never confident about
   has a hard cap, whatever the specialist says.
3. **`max`, not noisy-OR, between hypotheses** — "wholly generated" and
   "locally edited" are competing explanations of the same pixels; OR-ing them
   would manufacture confidence from two moderate readings. Separate *regions*
   do combine with a noisy-OR, since they are genuinely independent evidence.

Verdicts: `ai_generated`, `ai_edited`, `manipulated_not_necessarily_ai`,
`likely_authentic`, `likely_authentic_with_open_questions`, `uncertain`.

---

## Calibration

The map thresholds a patch score. Untouched, that is a cut on a number nobody
measured: both siblings were trained on whole images and saturate hard, so their
patch scores are neither calibrated nor comparable across backends.

```bash
python scripts/calibrate_mapper.py --data_dir data/raw/cifake \
    --backend model_01 --out configs/calibration_model_01.json
# then set mapper.calibration_path in your config
```

It reports held-out ECE before and after, and warns if the fit made calibration
worse. Until you fit one, the pipeline runs with `Calibrator.identity()`, **caps
reported confidence at 0.60, and says so in the explanation and in the report's
`notes`** — an uncalibrated map should not be claiming confidence it has not
earned.

One limitation stated plainly: fitting on a whole-image dataset like CIFAKE
labels every patch of a "fake" image as fake. That is right for fully generated
images and wrong for locally edited ones. Pass `--mask_dir` with tamper masks to
label patches by their centre pixel, which is the only honest way to calibrate
the locally-edited case.

---

## Tests

All four are torch-free and run in seconds — the pipeline test drives the whole
system through a stub scorer that measures real image content (it keys on
smoothness, as an over-smoothed generative fill would present), so nothing tells
the pipeline where the tampered square is.

```bash
cd server/model_03
python tests/test_windows.py        # coverage, edge clamping, degenerate sizes
python tests/test_blend.py          # correct averages, and no square artefacts
python tests/test_regions.py        # connected components, morphology, scipy/numpy parity
python tests/test_calibration.py    # monotonicity, ECE improvement, bad-fit refusal
python tests/test_pipeline.py       # end to end + the two conservatism rules in fusion
```

They are `pytest`-discoverable too (`pytest tests/`). All 41 pass in this
project's dev environment (Python 3.13, numpy 2.5, Pillow 12), and the pipeline
has additionally been run end to end against the real model_01 checkpoint.

---

## Known weaknesses

Ordered by how much they should change your reading of an output.

- **The patch scorer is the ceiling, and the shipped one is small.** model_01's
  bundled checkpoint is trained on CIFAKE at **32×32**, so every 64/128/224px
  patch is downsampled to 32px before scoring — lossy for exactly the fine
  blending seams this pipeline hunts. The map is only as good as the answers it
  stitches together. A patch-level training run is the single highest-value next
  step; the mapper is written so that dropping one in means adding a backend to
  `mapper/backends.py` and nothing else.
- **The finest scale sets the smallest findable edit.** A region is confidently
  mapped only where windows fit *inside* it. Measured on the 192px synthetic
  edit: scales `[64, 128, 224]` recover it at IoU 0.66, `[128, 224]` at IoU 0.17
  — the edit is still found and still correctly centred, but its outline shrinks
  toward the middle. An edit much smaller than the finest scale is diluted below
  threshold and missed entirely. Cost scales as 1/scale², so this is the knob to
  turn when the backend is expensive.
- **The specialists are classical forensics, not trained detectors.** Each of
  their measurements is individually beatable, and several are simply
  unavailable on some inputs: ELA and JPEG-grid evidence mean nothing on a PNG
  or a re-saved image. They are weighted by how much signal the image actually
  offers, and report low confidence rather than a confident 0.5, but they are
  corroboration — not a verdict.
- **No face-swap model.** See above; the face route under-claims by design, and
  is not taken at all when OpenCV is absent.
- **The router is untuned.** Its thresholds are reasoned, not fitted — there is
  no routing-labelled data in this project to fit them on. A learned router is
  the obvious successor once such data exists.
- **Robustness under the challenge's transforms is untested for the *map*.**
  model_01 and model_02 have transform×severity matrices; model_03 does not yet.
  Heavy JPEG or downscaling attacks precisely the local statistics the
  specialists read, so the expected failure is confidence collapsing to
  `uncertain` — which is the right failure, but it should be measured, not
  assumed.
- **Runtime** is ~1.2s for a 1600×1200 upload with the model_01 backend (932
  windows at 1024px working resolution), dominated by patch scoring. Without
  scipy, region extraction adds ~2s on whole-image verdicts.

---

## Relationship to model_01 and model_02

| | model_01 | model_02 | model_03 |
|---|---|---|---|
| Question | is this image AI? | is this image AI? | *where*, *what kind*, *how sure*? |
| Trains | CNN+Transformer end to end | a small classifier on frozen features | nothing |
| Output | one score | one score | map + regions + verdict + explanation |
| Depends on | — | — | one of the other two |

model_03 is not a replacement for either. It is a layer that turns either of
them into a localiser, and it inherits their strengths and their blind spots.
