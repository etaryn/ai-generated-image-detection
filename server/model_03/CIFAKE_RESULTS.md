# Results — model_03 on CIFAKE (real vs synthetic, no partially-AI subset)

**Dataset:** {'real': 100, 'synthetic': 100} · source `birdy654/cifake-real-and-ai-generated-synthetic-images`
**Base run:** backend `hf:Organika/sdxl-detector` · 100/class · max_side 768
**Patch-scorer run:** backend `hf:checkpoints/patch_scorer` · 100/class · max_side 768

> CIFAKE ships at native 32x32. `mapper/windows.py` clamps the [64, 128, 224]
> window scales down to the image's own short side and drops the resulting
> duplicates, so on these images the multi-scale region machinery collapses to
> a single whole-image window (see `eval/fetch_cifake.py`). Any gap between
> "with" and "without localisation" below comes from `fuse()`'s
> max()-of-hypotheses logic on that one region, not from spatial evidence --
> read it as a check on whether the fusion step itself ever hurts on
> wholly-generated images, not as a localisation result.

## 1. Headline: plain baseline vs the full upgrade

| Condition | baseline: base backend, no localisation | shipped: patch scorer + localisation | delta |
|---|---|---|---|
| clean | 0.617 | 0.669 | +0.052 |
| jpeg_q90 | 0.615 | 0.672 | +0.056 |
| jpeg_q60 | 0.620 | 0.645 | +0.025 |
| jpeg_q30 | 0.588 | 0.616 | +0.028 |
| blur_s0.5 | 0.597 | 0.709 | +0.112 |
| blur_s1.0 | 0.583 | 0.701 | +0.118 |
| blur_s2.0 | 0.546 | 0.614 | +0.068 |
| downscale_0.5 | 0.582 | 0.611 | +0.029 |
| downscale_0.25 | 0.520 | 0.585 | +0.065 |
| noise_s0.02 | 0.579 | 0.582 | +0.003 |
| noise_s0.05 | 0.532 | 0.604 | +0.072 |
| crop_0.9 | 0.610 | 0.701 | +0.091 |
| crop_0.8 | 0.548 | 0.641 | +0.093 |
| saturate_1.5 | 0.618 | 0.643 | +0.025 |

## 2. Isolating localisation, per backend and condition

**Base backend** (`AUC(real vs AI)`, with localisation -> without):

| Condition | with | without | delta |
|---|---|---|---|
| clean | 0.617 | 0.617 | +0.000 |
| jpeg_q90 | 0.615 | 0.615 | +0.000 |
| jpeg_q60 | 0.620 | 0.620 | +0.000 |
| jpeg_q30 | 0.588 | 0.588 | +0.000 |
| blur_s0.5 | 0.597 | 0.597 | +0.000 |
| blur_s1.0 | 0.583 | 0.583 | +0.000 |
| blur_s2.0 | 0.546 | 0.546 | +0.000 |
| downscale_0.5 | 0.582 | 0.582 | +0.000 |
| downscale_0.25 | 0.520 | 0.520 | +0.000 |
| noise_s0.02 | 0.579 | 0.579 | +0.000 |
| noise_s0.05 | 0.532 | 0.532 | +0.000 |
| crop_0.9 | 0.610 | 0.610 | +0.000 |
| crop_0.8 | 0.548 | 0.548 | +0.000 |
| saturate_1.5 | 0.618 | 0.618 | +0.000 |

**Patch scorer** (`AUC(real vs AI)`, with localisation -> without):

| Condition | with | without | delta |
|---|---|---|---|
| clean | 0.669 | 0.669 | +0.000 |
| jpeg_q90 | 0.672 | 0.672 | +0.000 |
| jpeg_q60 | 0.645 | 0.645 | +0.000 |
| jpeg_q30 | 0.616 | 0.616 | +0.000 |
| blur_s0.5 | 0.709 | 0.709 | +0.000 |
| blur_s1.0 | 0.701 | 0.701 | +0.000 |
| blur_s2.0 | 0.614 | 0.614 | +0.000 |
| downscale_0.5 | 0.611 | 0.611 | +0.000 |
| downscale_0.25 | 0.585 | 0.585 | +0.000 |
| noise_s0.02 | 0.582 | 0.582 | +0.000 |
| noise_s0.05 | 0.604 | 0.604 | +0.000 |
| crop_0.9 | 0.701 | 0.701 | +0.000 |
| crop_0.8 | 0.641 | 0.641 | +0.000 |
| saturate_1.5 | 0.643 | 0.643 | +0.000 |

## 3. Confidence and verdict stability

**Base backend**:

| Condition | mean confidence | verdict stability vs clean | mean regions/img |
|---|---|---|---|
| clean | 0.52 | 1.00 | 0.12 |
| jpeg_q90 | 0.52 | 0.95 | 0.12 |
| jpeg_q60 | 0.55 | 0.84 | 0.08 |
| jpeg_q30 | 0.51 | 0.81 | 0.18 |
| blur_s0.5 | 0.51 | 0.79 | 0.22 |
| blur_s1.0 | 0.51 | 0.27 | 0.69 |
| blur_s2.0 | 0.59 | 0.13 | 0.98 |
| downscale_0.5 | 0.49 | 0.41 | 0.53 |
| downscale_0.25 | 0.58 | 0.14 | 0.95 |
| noise_s0.02 | 0.56 | 0.80 | 0.06 |
| noise_s0.05 | 0.56 | 0.78 | 0.04 |
| crop_0.9 | 0.51 | 0.90 | 0.15 |
| crop_0.8 | 0.50 | 0.80 | 0.18 |
| saturate_1.5 | 0.56 | 0.85 | 0.10 |

**Patch scorer**:

| Condition | mean confidence | verdict stability vs clean | mean regions/img |
|---|---|---|---|
| clean | 0.59 | 1.00 | 0.00 |
| jpeg_q90 | 0.58 | 0.99 | 0.00 |
| jpeg_q60 | 0.60 | 0.98 | 0.00 |
| jpeg_q30 | 0.59 | 0.98 | 0.00 |
| blur_s0.5 | 0.59 | 0.99 | 0.01 |
| blur_s1.0 | 0.59 | 0.98 | 0.00 |
| blur_s2.0 | 0.57 | 0.94 | 0.00 |
| downscale_0.5 | 0.60 | 0.98 | 0.00 |
| downscale_0.25 | 0.60 | 0.98 | 0.00 |
| noise_s0.02 | 0.60 | 0.98 | 0.00 |
| noise_s0.05 | 0.60 | 0.98 | 0.00 |
| crop_0.9 | 0.60 | 0.98 | 0.00 |
| crop_0.8 | 0.60 | 0.98 | 0.00 |
| saturate_1.5 | 0.59 | 0.99 | 0.00 |

---

Regenerate from the two `eval/robustness.py` runs this was built from:

```bash
python eval/report_cifake.py eval_results/robustness_cifake_base.json eval_results/robustness_cifake_patch_scorer.json --out CIFAKE_RESULTS.md
```
