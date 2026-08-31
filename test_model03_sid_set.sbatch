#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-sidset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_sidset_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_sidset_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03sidset

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_03"

# Shard 4: every prior model_03 robustness run (778211/778685/778939, feeding
# EVALUATION_RESULTS.md S6/S7) used shard 3 at 15/class. This run is a fresh
# shard at a bigger sample (50/class = 150 images vs 45) for a tighter CI,
# written to its own output files so it doesn't clobber the numbers already
# cited in EVALUATION_RESULTS.md.
DATA=/tmp/aigc_m03_sidset/sid_set_val_shard4

echo "=== venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -iE "^torch" requirements.txt > /tmp/req_m03sidset.txt
pip install -r /tmp/req_m03sidset.txt
pip install "pyarrow>=15" "tqdm>=4.66"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "=== fetching SID-Set VALIDATION shard 4, 50/class ==="
mkdir -p "$DATA"
python eval/fetch_sid_set.py --split validation --shard 4 --per_class 50 --out "$DATA"

mkdir -p eval_results

echo "=== [1/2] robustness: BASE backend (limit 50, max_side 768) ==="
python eval/robustness.py --data_dir "$DATA" \
    --backend hf:Organika/sdxl-detector \
    --limit 50 --max_side 768 \
    --out eval_results/robustness_sidset_shard4_base.json

echo "=== [2/2] robustness: TUNED patch-scorer backend, same data, paired ==="
python eval/robustness.py --data_dir "$DATA" \
    --backend checkpoints/patch_scorer \
    --limit 50 --max_side 768 \
    --out eval_results/robustness_sidset_shard4_patch_scorer.json

echo "=== building the [backend] vs [backend + localisation + patch scorer] report ==="
python eval/report_robustness.py \
    eval_results/robustness_sidset_shard4_base.json \
    eval_results/robustness_sidset_shard4_patch_scorer.json \
    --out SID_SET_RESULTS.md

echo "=== MODEL_03 SID-SET TEST COMPLETE ==="
ls -la eval_results/robustness_sidset_shard4_*.json SID_SET_RESULTS.md
