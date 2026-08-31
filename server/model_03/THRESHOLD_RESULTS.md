# model_03 - threshold modes and localisation routing

Arms: base/absolute, patch_scorer/absolute, patch_scorer/median_shift, patch_scorer/quantile

## 1. Threshold modes: does removing the common-mode shift help?

`AUC(real vs tampered)`, and the share of *real* images that grew a
region -- localisation's false-positive rate, which is what the
aggregate AUC hides.

| condition | absolute | quantile | median_shift |
|---|---|---|---|
| **clean** | 0.854 (fp 14%) | 0.851 (fp 38%) | 0.867 (fp 42%) |
| jpeg_q90 | 0.872 (fp 18%) | 0.854 (fp 40%) | 0.886 (fp 42%) |
| jpeg_q60 | 0.595 (fp 56%) | 0.543 (fp 82%) | 0.522 (fp 86%) |
| **jpeg_q30** | 0.368 (fp 96%) | 0.372 (fp 98%) | 0.454 (fp 76%) |
| blur_s0.5 | 0.835 (fp 12%) | 0.850 (fp 42%) | 0.862 (fp 46%) |
| blur_s1.0 | 0.557 (fp 70%) | 0.535 (fp 92%) | 0.511 (fp 90%) |
| blur_s2.0 | 0.500 (fp 98%) | 0.505 (fp 100%) | 0.668 (fp 38%) |
| downscale_0.5 | 0.574 (fp 52%) | 0.578 (fp 82%) | 0.531 (fp 86%) |
| **downscale_0.25** | 0.411 (fp 60%) | 0.396 (fp 76%) | 0.378 (fp 66%) |
| noise_s0.02 | 0.775 (fp 6%) | 0.818 (fp 20%) | 0.820 (fp 24%) |
| **noise_s0.05** | 0.726 (fp 0%) | 0.730 (fp 2%) | 0.740 (fp 2%) |
| crop_0.9 | 0.871 (fp 14%) | 0.874 (fp 44%) | 0.890 (fp 46%) |
| crop_0.8 | 0.871 (fp 20%) | 0.879 (fp 50%) | 0.845 (fp 50%) |
| saturate_1.5 | 0.901 (fp 14%) | 0.910 (fp 32%) | 0.936 (fp 32%) |

## 2. Confidence, before and after the uncalibrated cap

`mean_confidence` saturates at UNCALIBRATED_CONFIDENCE_CAP (0.60), so it
cannot show the system losing certainty. The pre-cap value can.

| run | condition | reported | pre-cap | capped |
|---|---|---|---|---|
| patch_scorer/absolute | clean | 0.599 | 0.875 | 99% |
| patch_scorer/absolute | jpeg_q30 | 0.598 | 0.784 | 95% |
| patch_scorer/absolute | noise_s0.05 | 0.599 | 0.866 | 99% |
| patch_scorer/absolute | downscale_0.25 | 0.600 | 0.858 | 99% |
| patch_scorer/quantile | clean | 0.599 | 0.852 | 98% |
| patch_scorer/quantile | jpeg_q30 | 0.600 | 0.799 | 100% |
| patch_scorer/quantile | noise_s0.05 | 0.598 | 0.838 | 97% |
| patch_scorer/quantile | downscale_0.25 | 0.600 | 0.839 | 98% |
| patch_scorer/median_shift | clean | 0.599 | 0.860 | 97% |
| patch_scorer/median_shift | jpeg_q30 | 0.589 | 0.763 | 89% |
| patch_scorer/median_shift | noise_s0.05 | 0.599 | 0.854 | 95% |
| patch_scorer/median_shift | downscale_0.25 | 0.598 | 0.845 | 97% |
| base/absolute | clean | 0.600 | 0.846 | 99% |
| base/absolute | jpeg_q30 | 0.600 | 0.829 | 100% |
| base/absolute | noise_s0.05 | 0.600 | 0.897 | 100% |
| base/absolute | downscale_0.25 | 0.600 | 0.828 | 100% |

## 3. Routing localisation trust, per image

| condition | always fuse | always whole-image | best pure | gate | gate picks no-loc |
|---|---|---|---|---|---|
| clean | 0.854 | 0.575 | 0.854 | 0.833 | 38% |
| jpeg_q90 | 0.872 | 0.656 | 0.872 | 0.871 | 31% |
| jpeg_q60 | 0.595 | 0.531 | 0.595 | 0.588 | 39% |
| jpeg_q30 | 0.368 | 0.605 | 0.605 | 0.444 | 59% |
| blur_s0.5 | 0.835 | 0.576 | 0.835 | 0.816 | 38% |
| blur_s1.0 | 0.557 | 0.580 | 0.580 | 0.567 | 37% |
| blur_s2.0 | 0.500 | 0.572 | 0.572 | 0.736 | 69% |
| downscale_0.5 | 0.574 | 0.584 | 0.584 | 0.599 | 34% |
| downscale_0.25 | 0.411 | 0.568 | 0.568 | 0.465 | 39% |
| noise_s0.02 | 0.775 | 0.708 | 0.775 | 0.774 | 10% |
| noise_s0.05 | 0.726 | 0.723 | 0.726 | 0.726 | 0% |
| crop_0.9 | 0.871 | 0.594 | 0.871 | 0.861 | 37% |
| crop_0.8 | 0.871 | 0.624 | 0.871 | 0.848 | 39% |
| saturate_1.5 | 0.901 | 0.628 | 0.901 | 0.894 | 36% |

Mean gate gain over always-fuse: **+0.022** AUC.
Worst condition for the gate: `crop_0.8` at -0.022.
