# Results — model_03 robustness on saberzl/SID_Set

**Dataset:** {'real': 50, 'synthetic': 50, 'tampered': 50}
**Base run:** backend `hf:Organika/sdxl-detector` · 50/class · max_side 768
**Patch-scorer run:** backend `hf:checkpoints/patch_scorer` · 50/class · max_side 768

Both arms are paired: same images, same conditions, only the backend (and
whether localisation is applied) differs. `whole_image_score`/`*_whole_image`
metrics are the backend's own single-pass score on the full image (the "no
localisation" arm); the plain metrics are the fused, region-aware score.

## 1. Headline: plain baseline vs the full upgrade

### All AI (real vs synthetic+tampered)

| Condition | baseline: base backend, no localisation | shipped: patch scorer + localisation | delta |
|---|---|---|---|
| clean | 0.759 | 0.919 | +0.160 |
| jpeg_q90 | 0.752 | 0.920 | +0.168 |
| jpeg_q60 | 0.647 | 0.683 | +0.037 |
| jpeg_q30 | 0.735 | 0.430 | -0.304 |
| blur_s0.5 | 0.763 | 0.911 | +0.149 |
| blur_s1.0 | 0.754 | 0.753 | -0.001 |
| blur_s2.0 | 0.743 | 0.678 | -0.065 |
| downscale_0.5 | 0.749 | 0.772 | +0.024 |
| downscale_0.25 | 0.739 | 0.665 | -0.074 |
| noise_s0.02 | 0.761 | 0.879 | +0.117 |
| noise_s0.05 | 0.769 | 0.839 | +0.070 |
| crop_0.9 | 0.764 | 0.929 | +0.165 |
| crop_0.8 | 0.790 | 0.925 | +0.135 |
| saturate_1.5 | 0.772 | 0.948 | +0.175 |

### Tampered only (real vs partially-AI)

| Condition | baseline: base backend, no localisation | shipped: patch scorer + localisation | delta |
|---|---|---|---|
| clean | 0.587 | 0.854 | +0.267 |
| jpeg_q90 | 0.592 | 0.872 | +0.280 |
| jpeg_q60 | 0.387 | 0.594 | +0.207 |
| jpeg_q30 | 0.555 | 0.367 | -0.188 |
| blur_s0.5 | 0.590 | 0.835 | +0.245 |
| blur_s1.0 | 0.576 | 0.554 | -0.023 |
| blur_s2.0 | 0.562 | 0.501 | -0.061 |
| downscale_0.5 | 0.568 | 0.575 | +0.007 |
| downscale_0.25 | 0.544 | 0.411 | -0.134 |
| noise_s0.02 | 0.589 | 0.775 | +0.186 |
| noise_s0.05 | 0.630 | 0.726 | +0.096 |
| crop_0.9 | 0.587 | 0.872 | +0.285 |
| crop_0.8 | 0.635 | 0.871 | +0.235 |
| saturate_1.5 | 0.603 | 0.901 | +0.298 |


## 2. Isolating localisation, per backend and condition

**Base backend** (`AUC(real vs all AI)`, with localisation -> without):

| Condition | with | without | delta |
|---|---|---|---|
| clean | 0.739 | 0.759 | -0.020 |
| jpeg_q90 | 0.740 | 0.752 | -0.011 |
| jpeg_q60 | 0.634 | 0.647 | -0.013 |
| jpeg_q30 | 0.723 | 0.735 | -0.012 |
| blur_s0.5 | 0.723 | 0.763 | -0.040 |
| blur_s1.0 | 0.665 | 0.754 | -0.088 |
| blur_s2.0 | 0.760 | 0.743 | +0.017 |
| downscale_0.5 | 0.657 | 0.749 | -0.091 |
| downscale_0.25 | 0.729 | 0.739 | -0.011 |
| noise_s0.02 | 0.699 | 0.761 | -0.063 |
| noise_s0.05 | 0.725 | 0.769 | -0.044 |
| crop_0.9 | 0.745 | 0.764 | -0.019 |
| crop_0.8 | 0.755 | 0.790 | -0.035 |
| saturate_1.5 | 0.754 | 0.772 | -0.018 |

**Patch scorer** (`AUC(real vs all AI)`, with localisation -> without):

| Condition | with | without | delta |
|---|---|---|---|
| clean | 0.919 | 0.782 | +0.137 |
| jpeg_q90 | 0.920 | 0.818 | +0.102 |
| jpeg_q60 | 0.683 | 0.756 | -0.072 |
| jpeg_q30 | 0.430 | 0.781 | -0.351 |
| blur_s0.5 | 0.911 | 0.783 | +0.128 |
| blur_s1.0 | 0.753 | 0.784 | -0.032 |
| blur_s2.0 | 0.678 | 0.777 | -0.099 |
| downscale_0.5 | 0.772 | 0.787 | -0.015 |
| downscale_0.25 | 0.665 | 0.776 | -0.111 |
| noise_s0.02 | 0.879 | 0.844 | +0.035 |
| noise_s0.05 | 0.839 | 0.837 | +0.002 |
| crop_0.9 | 0.929 | 0.793 | +0.136 |
| crop_0.8 | 0.925 | 0.806 | +0.119 |
| saturate_1.5 | 0.948 | 0.811 | +0.137 |

**Base backend** (`AUC(real vs tampered)`, with localisation -> without):

| Condition | with | without | delta |
|---|---|---|---|
| clean | 0.584 | 0.587 | -0.003 |
| jpeg_q90 | 0.598 | 0.592 | +0.006 |
| jpeg_q60 | 0.375 | 0.387 | -0.012 |
| jpeg_q30 | 0.539 | 0.555 | -0.016 |
| blur_s0.5 | 0.556 | 0.590 | -0.034 |
| blur_s1.0 | 0.479 | 0.576 | -0.098 |
| blur_s2.0 | 0.620 | 0.562 | +0.059 |
| downscale_0.5 | 0.474 | 0.568 | -0.094 |
| downscale_0.25 | 0.606 | 0.544 | +0.062 |
| noise_s0.02 | 0.545 | 0.589 | -0.044 |
| noise_s0.05 | 0.611 | 0.630 | -0.019 |
| crop_0.9 | 0.591 | 0.587 | +0.003 |
| crop_0.8 | 0.606 | 0.635 | -0.029 |
| saturate_1.5 | 0.616 | 0.603 | +0.013 |

**Patch scorer** (`AUC(real vs tampered)`, with localisation -> without):

| Condition | with | without | delta |
|---|---|---|---|
| clean | 0.854 | 0.574 | +0.279 |
| jpeg_q90 | 0.872 | 0.656 | +0.216 |
| jpeg_q60 | 0.594 | 0.531 | +0.063 |
| jpeg_q30 | 0.367 | 0.605 | -0.238 |
| blur_s0.5 | 0.835 | 0.576 | +0.259 |
| blur_s1.0 | 0.554 | 0.580 | -0.026 |
| blur_s2.0 | 0.501 | 0.572 | -0.071 |
| downscale_0.5 | 0.575 | 0.584 | -0.009 |
| downscale_0.25 | 0.411 | 0.569 | -0.158 |
| noise_s0.02 | 0.775 | 0.707 | +0.068 |
| noise_s0.05 | 0.726 | 0.723 | +0.002 |
| crop_0.9 | 0.872 | 0.594 | +0.278 |
| crop_0.8 | 0.871 | 0.624 | +0.246 |
| saturate_1.5 | 0.901 | 0.628 | +0.272 |


## 3. Confidence, verdict stability, localisation

**Base backend**:

| Condition | mean confidence | verdict stability vs clean | mean regions/img | loc IoU | loc recall |
|---|---|---|---|---|---|
| clean | 0.60 | 1.00 | 1.65 | 0.041 | 0.076 |
| jpeg_q90 | 0.60 | 0.93 | 1.70 | 0.040 | 0.077 |
| jpeg_q60 | 0.60 | 0.80 | 1.81 | 0.021 | 0.040 |
| jpeg_q30 | 0.60 | 0.65 | 2.71 | 0.043 | 0.105 |
| blur_s0.5 | 0.60 | 0.94 | 1.73 | 0.042 | 0.077 |
| blur_s1.0 | 0.60 | 0.78 | 2.58 | 0.069 | 0.165 |
| blur_s2.0 | 0.60 | 0.65 | 1.71 | 0.109 | 0.621 |
| downscale_0.5 | 0.60 | 0.76 | 2.50 | 0.064 | 0.153 |
| downscale_0.25 | 0.60 | 0.70 | 2.53 | 0.093 | 0.358 |
| noise_s0.02 | 0.60 | 0.49 | 1.17 | 0.001 | 0.001 |
| noise_s0.05 | 0.60 | 0.30 | 0.23 | 0.000 | 0.000 |
| crop_0.9 | 0.60 | 0.89 | 1.96 | 0.051 | 0.113 |
| crop_0.8 | 0.60 | 0.79 | 2.16 | 0.052 | 0.106 |
| saturate_1.5 | 0.60 | 0.83 | 1.58 | 0.028 | 0.052 |

**Patch scorer**:

| Condition | mean confidence | verdict stability vs clean | mean regions/img | loc IoU | loc recall |
|---|---|---|---|---|---|
| clean | 0.60 | 1.00 | 0.94 | 0.416 | 0.523 |
| jpeg_q90 | 0.60 | 0.80 | 1.05 | 0.408 | 0.503 |
| jpeg_q60 | 0.60 | 0.61 | 1.67 | 0.422 | 0.526 |
| jpeg_q30 | 0.60 | 0.50 | 2.55 | 0.372 | 0.887 |
| blur_s0.5 | 0.60 | 0.97 | 0.95 | 0.412 | 0.526 |
| blur_s1.0 | 0.60 | 0.75 | 1.50 | 0.317 | 0.436 |
| blur_s2.0 | 0.60 | 0.63 | 2.26 | 0.256 | 0.665 |
| downscale_0.5 | 0.60 | 0.75 | 1.35 | 0.237 | 0.271 |
| downscale_0.25 | 0.60 | 0.63 | 1.29 | 0.123 | 0.149 |
| noise_s0.02 | 0.60 | 0.68 | 0.61 | 0.136 | 0.179 |
| noise_s0.05 | 0.60 | 0.34 | 0.09 | 0.005 | 0.006 |
| crop_0.9 | 0.60 | 0.93 | 0.92 | 0.413 | 0.568 |
| crop_0.8 | 0.60 | 0.88 | 0.95 | 0.342 | 0.562 |
| saturate_1.5 | 0.60 | 0.91 | 0.91 | 0.421 | 0.519 |

---

Regenerate from the two `eval/robustness.py` runs this was built from:

```bash
python eval/report_robustness.py eval_results/robustness_sidset_shard4_base.json eval_results/robustness_sidset_shard4_patch_scorer.json --out SID_SET_RESULTS.md
```
