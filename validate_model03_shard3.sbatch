#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-val3
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_val3_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_val3_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03val3

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_03"

# HELD-OUT VALIDATION for the two findings tuned on shard 4 (job 779811):
#
#   1. gate localisation on the pipeline's own pre-cap confidence, falling back
#      to the whole-image score when confidence_uncapped < 0.858. The threshold
#      was chosen by leave-one-condition-out on shard 4 and is applied here
#      FROZEN -- nothing is refitted on shard 3, or this stops being a test.
#   2. median_shift cuts, which beat absolute on localisation IoU in 11 of 14
#      shard-4 conditions while being mixed on detection AUC.
#
# Shard 3 is a different parquet file from shard 4 (validation-00003 vs
# -00004 of 00034), so the images are disjoint. Prior shard-3 runs
# (778211/778685/778939, feeding EVALUATION_RESULTS.md S6/S7) used 15/class and
# the uninstrumented harness; this uses 50/class for a tighter interval and
# writes to its own files, so nothing already cited is overwritten.
DATA=/tmp/aigc_m03_val3/sid_set_val_shard3

echo "=== venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -iE "^torch" requirements.txt > /tmp/req_m03val3.txt
pip install -r /tmp/req_m03val3.txt
pip install "pyarrow>=15" "tqdm>=4.66"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== fetching SID-Set VALIDATION shard 3, 50/class (disjoint from shard 4) ==="
mkdir -p "$DATA"
python eval/fetch_sid_set.py --split validation --shard 3 --per_class 50 --out "$DATA"

mkdir -p eval_results
COMMON="--data_dir $DATA --limit 50 --max_side 768"

echo "=== [1/2] shard 3: patch scorer, ABSOLUTE cuts (the arm the gate was tuned on) ==="
python eval/robustness.py $COMMON \
    --backend checkpoints/patch_scorer \
    --threshold_mode absolute \
    --out eval_results/val3_shard3_ps_absolute.json

echo "=== [2/2] shard 3: patch scorer, MEDIAN_SHIFT cuts (ref from shard 4 clean) ==="
python eval/robustness.py $COMMON \
    --backend checkpoints/patch_scorer \
    --threshold_mode median_shift --adaptive_ref_median 0.3660738479313538 \
    --out eval_results/val3_shard3_ps_median_shift.json

echo "=== MODEL_03 SHARD-3 VALIDATION COMPLETE ==="
ls -la eval_results/val3_shard3_*.json
