#!/usr/bin/env bash
#SBATCH --job-name=aigc-train
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/train_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/train_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv

# Route every cache (pip, kagglehub, matplotlib, etc.) to node-local /tmp --
# $HOME on this cluster has a very small quota and cannot hold the CIFAKE
# dataset or the CUDA/torch package cache.
export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export MPLCONFIGDIR=/tmp/mpl_cache
export KAGGLEHUB_CACHE=/tmp/kagglehub_cache
# Credentials come from ~/.kaggle/kaggle.json (kagglehub reads it automatically) --
# do not hardcode KAGGLE_USERNAME/KAGGLE_KEY here, this file gets committed.

cd "$REPO/server/model_01"

echo "=== building venv at $VENV (node-local, no home quota) ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

echo "=== installing torch (pinned cu124 build) ==="
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== installing remaining requirements ==="
grep -v -iE "^torch" ../requirements.txt > /tmp/requirements_notorch.txt
pip install -r /tmp/requirements_notorch.txt

echo "=== torch/cuda sanity check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== downloading CIFAKE via kagglehub ==="
python data/download_cifake.py --out data/raw/cifake

echo "=== starting full training run (configs/default.yaml) ==="
python train.py --config configs/default.yaml

echo "=== TRAINING COMPLETE ==="
ls -la checkpoints/
