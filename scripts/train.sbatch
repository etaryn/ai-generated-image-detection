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
# Full training run on the SoC cluster.
#   sbatch scripts/train.sbatch
#
# gpu-long gives 3 days; we ask for 2 so there's headroom to resubmit. train.py
# writes checkpoints/last.pt every epoch and --resume picks it up, so hitting the
# walltime limit (or being requeued) costs at most one epoch, not the whole run.

set -euo pipefail

# Override either at submit time, e.g.:  REPO=~/work/aigc sbatch scripts/train.sbatch
REPO="${REPO:-$HOME/ai-generated-image-detection}"
VENV="${VENV:-$HOME/.venv}"

cd "$REPO"
mkdir -p logs
source "$VENV/bin/activate"

echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# /home is CFS network storage. The dataset is ~120k small files, and reading them
# per-batch over NFS will starve the GPU. Stage to the node's local disk first and
# point the run at that copy. $TMPDIR is per-job and cleaned up automatically.
DATA_ROOT="data/raw"
if [[ -n "${STAGE_DATA:-}" ]]; then
    echo "staging data/raw -> $TMPDIR/raw"
    mkdir -p "$TMPDIR/raw"
    cp -r data/raw/. "$TMPDIR/raw/"
    DATA_ROOT="$TMPDIR/raw"
fi

srun python train.py \
    --config configs/default.yaml \
    --data_root "$DATA_ROOT" \
    --resume
