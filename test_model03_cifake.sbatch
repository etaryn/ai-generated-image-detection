#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-cifake
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_cifake_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_cifake_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03cifake

# Route every cache to node-local /tmp -- $HOME quota can't hold CIFAKE + the
# torch/CUDA package cache (same discipline as train_cifake.sbatch / m03robust).
export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache
export KAGGLEHUB_CACHE=/tmp/kagglehub_cache
# Kaggle credentials come from ~/.kaggle/kaggle.json (kagglehub reads it
# automatically) -- do not hardcode KAGGLE_USERNAME/KAGGLE_KEY here.

cd "$REPO/server/model_03"

DATA=/tmp/aigc_m03_cifake/cifake_val

echo "=== venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -iE "^torch" requirements.txt > /tmp/req_m03cifake.txt
pip install -r /tmp/req_m03cifake.txt
pip install "kagglehub" "tqdm>=4.66"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "=== fetching CIFAKE sample (100/class, real + synthetic, no in-between) ==="
mkdir -p "$DATA"
python eval/fetch_cifake.py --per_class 100 --seed 0 --out "$DATA"

mkdir -p eval_results

echo "=== [1/2] robustness: BASE backend, no patch scorer (limit 100/class, max_side 768) ==="
python eval/robustness.py --data_dir "$DATA" \
    --backend hf:Organika/sdxl-detector \
    --limit 100 --max_side 768 \
    --out eval_results/robustness_cifake_base.json

echo "=== [2/2] robustness: TUNED patch-scorer backend, same data, paired ==="
python eval/robustness.py --data_dir "$DATA" \
    --backend checkpoints/patch_scorer \
    --limit 100 --max_side 768 \
    --out eval_results/robustness_cifake_patch_scorer.json

echo "=== building the [backend] vs [backend + localisation + patch scorer] report ==="
python eval/report_cifake.py \
    eval_results/robustness_cifake_base.json \
    eval_results/robustness_cifake_patch_scorer.json \
    --out CIFAKE_RESULTS.md

echo "=== MODEL_03 CIFAKE TEST COMPLETE ==="
ls -la eval_results/robustness_cifake_*.json CIFAKE_RESULTS.md
