#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-patch
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /tmp | tail -1

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_model03

# Route every cache (pip, HF, torch hub, matplotlib) to node-local /tmp --
# $HOME on this cluster has a very small quota and cannot hold the SID-Set
# sample, the detector weights, or the CUDA/torch package cache.
export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

# The SID-Set sample also lives on node-local /tmp for the same reason. It is
# not an artifact worth keeping: eval/fetch_sid_set.py reads fixed row groups in
# order, so re-running the same command reproduces byte-identical images.
DATA=/tmp/aigc_m03_data/sid_set_train

cd "$REPO/server/model_03"

echo "=== building venv at $VENV (node-local, no home quota) ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

echo "=== installing torch (pinned cu124 build) ==="
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== installing model_03 requirements ==="
grep -v -iE "^torch" requirements.txt > /tmp/requirements_notorch_m03.txt
pip install -r /tmp/requirements_notorch_m03.txt
# Not in requirements.txt: eval/fetch_sid_set.py streams parquet row groups over
# HTTP range requests (pyarrow), and train_patch_scorer.py uses tqdm for its
# per-epoch progress bar when it is importable.
pip install "pyarrow>=15" "tqdm>=4.66"

echo "=== torch/cuda sanity check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== plumbing tests (no weight downloads, no data needed) ==="
python tests/test_pipeline.py
python tests/test_fast_preprocess.py

echo "=== fetching SID-Set train sample -> $DATA ==="
# 6 shards x 100 per class: exactly the sample EVALUATION.md used locally, so
# the resulting numbers are comparable to the recorded base-model baseline.
mkdir -p "$DATA"
python eval/fetch_sid_set.py --split train --shard 0 --shards 6 \
    --per_class 100 --out "$DATA"
du -sh "$DATA"

echo "=== fine-tuning the patch scorer ==="
python scripts/train_patch_scorer.py \
    --data "$DATA" --out checkpoints/patch_scorer \
    --epochs 3 --per_image 4 --batch_size 32

echo "=== MODEL_03 PATCH SCORER TRAINING COMPLETE ==="
ls -la checkpoints/patch_scorer/
cat checkpoints/patch_scorer/training.json
