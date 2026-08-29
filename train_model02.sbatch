#!/usr/bin/env bash
#SBATCH --job-name=aigc-model02
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model02_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model02_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_model02

# Route every cache (pip, HF, torch hub, kagglehub, matplotlib, etc.) to
# node-local /tmp -- $HOME on this cluster has a very small quota and cannot
# hold the CIFAKE dataset, DINOv2/CLIP weights, or the CUDA/torch package cache.
export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache
export KAGGLEHUB_CACHE=/tmp/kagglehub_cache
# Credentials come from ~/.kaggle/kaggle.json (kagglehub reads it automatically) --
# do not hardcode KAGGLE_USERNAME/KAGGLE_KEY here, this file gets committed.

cd "$REPO/server/model_02"

echo "=== building venv at $VENV (node-local, no home quota) ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

echo "=== installing torch (pinned cu124 build) ==="
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== installing remaining requirements (model_02 + shared server reqs) ==="
grep -h -v -iE "^torch" ../requirements.txt requirements.txt > /tmp/requirements_notorch_m02.txt
pip install -r /tmp/requirements_notorch_m02.txt

echo "=== torch/cuda sanity check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== plumbing tests (no weight downloads, no data needed) ==="
python tests/test_fft_features.py
python tests/test_backbone_extractors.py
python tests/test_pipeline.py
python tests/test_train_roundtrip.py

echo "=== downloading CIFAKE into repo-root data/raw (model_02 reads ../../data/raw, distinct from model_01's server/model_01/data/raw) ==="
mkdir -p "$REPO/data/raw"
python ../model_01/data/download_cifake.py --out "$REPO/data/raw/cifake"

echo "=== quick smoke run: extraction + train on 2000 images (validates DINOv2/CLIP weight downloads + full pipeline) ==="
python extract_features.py --config configs/quick_smoke.yaml --limit 2000
python train.py --config configs/quick_smoke.yaml

echo "=== full feature extraction (expensive step; cached to disk afterwards) ==="
python extract_features.py --config configs/default.yaml

echo "=== full classifier training (mlp) ==="
python train.py --config configs/default.yaml

echo "=== MODEL_02 TRAINING COMPLETE ==="
ls -la checkpoints/
