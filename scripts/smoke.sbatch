#!/bin/bash
#SBATCH --job-name=aigc-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Cheap correctness pass before burning a gpu-long allocation. Runs the forward/
# backward shape test (which has never actually been executed -- see README) and
# then one epoch on the network-free synthetic dataset.
#
#   sbatch scripts/smoke.sbatch

set -euo pipefail

REPO="${REPO:-$HOME/ai-generated-image-detection}"
VENV="${VENV:-$HOME/.venv}"

cd "$REPO"
mkdir -p logs
source "$VENV/bin/activate"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python tests/test_model_shapes.py
python data/make_synthetic_dataset.py --out data/raw/synthetic --n_per_class 200
python train.py --config configs/smoke.yaml
