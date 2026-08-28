#!/bin/bash
#SBATCH --job-name=aigc-train
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#
# Full training run.
#   DATA_ARCHIVE=$HOME/cifake.tar sbatch scripts/train.sbatch
#
# gpu-long allows 3 days; we ask for 2 to leave resubmit headroom. train.py writes
# checkpoints/last.pt every epoch and --resume picks it up, so hitting the walltime
# limit (or a requeue) costs one epoch, not the run.

set -euo pipefail

REPO="${REPO:-$HOME/ai-generated-image-detection}"
cd "$REPO"
source scripts/_bootstrap.sh

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun python train.py \
    --config configs/default.yaml \
    --data_root "$DATA_ROOT" \
    --resume
