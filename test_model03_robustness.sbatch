#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-robust
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_robust_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_robust_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03robust

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_03"

# Shard 3: untouched by anything so far (0 = eval+decision-fit, 1 = calibration,
# 2 = the held-out test). Not load-bearing for robustness the way it is for
# decision-threshold fitting, but keeping the same discipline costs nothing.
DATA=/tmp/aigc_m03_robust/sid_set_val_shard3

echo "=== venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -iE "^torch" requirements.txt > /tmp/req_m03robust.txt
pip install -r /tmp/req_m03robust.txt
pip install "pyarrow>=15" "tqdm>=4.66"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "=== fetching SID-Set VALIDATION shard 3, 15/class (matches the published robustness sample size) ==="
mkdir -p "$DATA"
python eval/fetch_sid_set.py --split validation --shard 3 --per_class 15 --out "$DATA"

mkdir -p eval_results

echo "=== [1/2] robustness: BASE backend, same settings as the published table (limit 15, max_side 768) ==="
python eval/robustness.py --data_dir "$DATA" \
    --backend hf:Organika/sdxl-detector \
    --limit 15 --max_side 768 \
    --out eval_results/robustness_base.json

echo "=== [2/2] robustness: TUNED patch-scorer backend, same data, paired ==="
python eval/robustness.py --data_dir "$DATA" \
    --backend checkpoints/patch_scorer \
    --limit 15 --max_side 768 \
    --out eval_results/robustness_patch_scorer.json

echo "=== MODEL_03 ROBUSTNESS TEST COMPLETE ==="
ls -la eval_results/robustness_*.json
