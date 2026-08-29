#!/bin/bash
#SBATCH --job-name=aigc-train
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#
# Full training run.
#   DATA_ARCHIVE=$HOME/cifake.tar sbatch scripts/train.sbatch
#
# train.py writes checkpoints/last.pt every epoch and --resume picks it up, so
# hitting the walltime limit (or a requeue) costs one epoch, not the run.
#
# Walltime is 8h rather than the old 2 days: at CIFAKE's native 32px an epoch is
# minutes, not the ~7 minutes/epoch the 224px upsampling cost in job 768468.
#
# NOTE on --resume: train.py refuses to resume from a checkpoint whose model config
# differs from the current one. The 32px switch changed model.cnn_transformer, so
# checkpoints from job 768468 and earlier will (correctly) be rejected -- they were
# moved to checkpoints/archive-768468-224px/ when the config changed.

set -euo pipefail

REPO="${REPO:-$HOME/ai-generated-image-detection}"
cd "$REPO"
source scripts/_bootstrap.sh

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---------------------------------------------------------------------------- #
# Gate: prove the model can memorize 256 images before spending the allocation.
# ---------------------------------------------------------------------------- #
# Job 768468 ran 1h49m on a model that was mathematically dead by epoch 4. This
# check takes ~1 minute and exits non-zero if the model cannot fit a tiny set with
# augmentation and the consistency loss switched off -- which no correctly-wired
# classifier fails. `set -e` then aborts the job before training starts.
echo "=== pre-flight gate: overfit 256 images ==="
srun python scripts/diagnose.py \
    --part overfit \
    --gate \
    --config configs/default.yaml \
    --data_root "$DATA_ROOT"

echo "=== gate passed, starting full run ==="
srun python train.py \
    --config configs/default.yaml \
    --data_root "$DATA_ROOT" \
    --resume
