#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-dual
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_dual_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_dual_%j.err

set -uo pipefail   # NOT -e: the test suite must be allowed to fail and still report

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03dual

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_03"

# Two jobs in one:
#
#   [1] Run the unit tests. fusion.py, mapper/heatmap.py and eval/robustness.py
#       were all modified without ever being tested -- the login node has
#       neither torch nor pytest, so test_labels/test_pipeline/test_regions have
#       not seen the new threshold_mode code path at all. This is the gap.
#
#   [2] First real-backend run of DualBackendAnalyzer. Everything measured about
#       the gate so far was replayed offline from per-image JSON; this exercises
#       the actual class, on shard 3, with the shard-4 threshold frozen.
DATA=/tmp/aigc_m03_dual/sid_set_val_shard3

echo "=== venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -iE "^torch" requirements.txt > /tmp/req_m03dual.txt
pip install -r /tmp/req_m03dual.txt
pip install "pyarrow>=15" "tqdm>=4.66" pytest

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo
echo "=================== [1/3] UNIT TESTS ==================="
python -m pytest tests/ -q --no-header -rf 2>&1 | tail -40
TEST_STATUS=${PIPESTATUS[0]}
echo "=== pytest exit status: $TEST_STATUS ==="

echo
echo "=== fetching SID-Set VALIDATION shard 3, 50/class ==="
mkdir -p "$DATA"
python eval/fetch_sid_set.py --split validation --shard 3 --per_class 50 --out "$DATA"

mkdir -p eval_results
COMMON="--data_dir $DATA --limit 50 --max_side 768 --backend checkpoints/patch_scorer"

echo
echo "=========== [2/3] DUAL, self-fallback (the default) ==========="
# Threshold frozen from shard 4; nothing is refitted here.
python eval/robustness.py $COMMON --dual --fallback_backend self \
    --out eval_results/dual_shard3_self.json

echo
echo "=========== [3/3] DUAL, separate sdxl-detector fallback ==========="
# Measured worse offline (0.745 vs 0.806 mean AUC on shard 4); run here so the
# comparison is made with the real class rather than replayed from JSON.
python eval/robustness.py $COMMON --dual --fallback_backend hf:Organika/sdxl-detector \
    --out eval_results/dual_shard3_separate.json

echo
echo "=== MODEL_03 DUAL-BACKEND RUN COMPLETE (pytest exit was $TEST_STATUS) ==="
ls -la eval_results/dual_shard3_*.json
