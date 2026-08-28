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
# Cheap correctness pass before spending a gpu-long allocation:
#   1. the forward/backward shape test, which has never actually been executed
#   2. one epoch on the network-free synthetic dataset
#
#   sbatch scripts/smoke.sbatch

set -euo pipefail

REPO="${REPO:-$HOME/ai-generated-image-detection}"
cd "$REPO"
source scripts/_bootstrap.sh

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python tests/test_model_shapes.py

# Generated onto node-local /tmp: 400 images in $HOME would be 400 inodes.
python data/make_synthetic_dataset.py --out "$LOCAL_ROOT/raw/synthetic" --n_per_class 200
python train.py --config configs/smoke.yaml --data_root "$LOCAL_ROOT/raw"
