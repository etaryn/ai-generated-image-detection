#!/bin/bash
#SBATCH --job-name=aigc-diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Post-mortem on a finished training run. Answers three questions in one job:
# what the checkpoint predicts, whether the code can memorize a tiny set, and
# whether a stock ResNet-18 can learn the data at all.
#
#   DATA_ARCHIVE=$HOME/cifake.tar sbatch scripts/diagnose.sbatch
#
# Runs on the short `gpu` partition, not gpu-long -- it should finish in minutes.

set -euo pipefail

REPO="${REPO:-$HOME/ai-generated-image-detection}"
cd "$REPO"
source scripts/_bootstrap.sh

# stdout only: tqdm is off here, so the .out file stays readable.
# CONFIG / DATASETS are overridable so a config's MODEL can be run against a
# different dataset -- e.g. the 224px smoke model against real CIFAKE, which is
# what tells a model-config bug apart from data that is simply too subtle.
#   CONFIG=configs/smoke.yaml DATASETS=cifake PART=overfit \
#     DATA_ARCHIVE=$HOME/cifake.tar sbatch scripts/diagnose.sbatch
srun python scripts/diagnose.py \
    --part "${PART:-all}" \
    --config "${CONFIG:-configs/default.yaml}" \
    ${DATASETS:+--datasets $DATASETS} \
    ${OVERFIT_STEPS:+--overfit_steps $OVERFIT_STEPS} \
    --data_root "$DATA_ROOT"
