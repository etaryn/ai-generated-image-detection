#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-heldout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_heldout_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_heldout_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03heldout

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_03"

# Validation shard 2: touched by nothing so far. Shard 0 was already used by
# train_model03_scaled.sbatch's decision sweep (scripts/fit_decision.py) --
# scoring on it again would be reporting threshold-fitting error as a result,
# exactly what EVALUATION.md warns against. Shard 1 is the conventional
# calibration shard. Neither is a clean held-out set for this checkpoint.
DATA=/tmp/aigc_m03_heldout/sid_set_val_shard2

echo "=== venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -iE "^torch" requirements.txt > /tmp/req_m03heldout.txt
pip install -r /tmp/req_m03heldout.txt
pip install "pyarrow>=15" "tqdm>=4.66"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "=== fetching SID-Set VALIDATION shard 2, 120/class (untouched by training or decision-fitting) ==="
mkdir -p "$DATA"
python eval/fetch_sid_set.py --split validation --shard 2 --per_class 120 --out "$DATA"

mkdir -p eval_results

echo "=== [1/4] shipped pipeline, tuned checkpoint, current configs/default.yaml thresholds ==="
python eval/evaluate.py --data_dir "$DATA" \
    --backend checkpoints/patch_scorer \
    --out eval_results/heldout_patch_scorer.json

echo "=== [2/4] shipped pipeline, base backend, same data, paired ==="
python eval/evaluate.py --data_dir "$DATA" \
    --backend hf:Organika/sdxl-detector \
    --out eval_results/heldout_base.json

echo "=== [3/4] ablation: does region-awareness beat whole-image, on held-out data? ==="
python eval/ablation.py eval_results/heldout_patch_scorer.json \
    --out eval_results/heldout_tables_patch_scorer.md \
    --json_out eval_results/heldout_ablation_patch_scorer.json
python eval/ablation.py eval_results/heldout_base.json \
    --out eval_results/heldout_tables_base.md \
    --json_out eval_results/heldout_ablation_base.json

echo "=== [4/4] does the fit_decision.py sweep's 'max' result generalise, or was it fit to shard 0? ==="
python scripts/fit_decision.py \
    --backend checkpoints/patch_scorer \
    --manifest "$DATA/manifest.json" \
    --limit 120 \
    --out eval_results/heldout_decision_sweep.json

echo "=== MODEL_03 HELD-OUT TEST COMPLETE ==="
echo "--- tuned vs base, shard-0-fitted config ---"
cat eval_results/heldout_ablation_patch_scorer.json 2>/dev/null | head -60
echo "--- decision sweep on never-before-seen shard 2 ---"
cat eval_results/heldout_decision_sweep.json
