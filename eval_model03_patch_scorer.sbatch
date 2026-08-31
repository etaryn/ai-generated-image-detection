#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-eval
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_eval_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_eval_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03eval

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

# Validation sample on node-local /tmp (deterministic fetch, regenerable).
# The checkpoint and the JSON results live in $HOME and persist.
DATA=/tmp/aigc_m03_eval/sid_set_val

cd "$REPO/server/model_03"

echo "=== venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -iE "^torch" requirements.txt > /tmp/req_m03eval.txt
pip install -r /tmp/req_m03eval.txt
pip install "pyarrow>=15" "tqdm>=4.66"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "=== fetching SID-Set VALIDATION shard 0, 120/class ==="
# Exactly the sample EVALUATION_RESULTS.md used, so the tuned numbers are
# directly comparable to the recorded base-model run. train was shards 0-5 of
# the *train* split -- disjoint from validation, so no leakage.
mkdir -p "$DATA"
python eval/fetch_sid_set.py --split validation --shard 0 --per_class 120 --out "$DATA"

mkdir -p eval_results

echo "=== [1/2] evaluating TUNED patch scorer ==="
python eval/evaluate.py --data_dir "$DATA" \
    --backend checkpoints/patch_scorer \
    --out eval_results/patch_scorer.json

echo "=== [2/2] evaluating BASE model (same env, same data, paired) ==="
python eval/evaluate.py --data_dir "$DATA" \
    --backend hf:Organika/sdxl-detector \
    --out eval_results/base_rerun.json

echo "=== ablation: tuned ==="
python eval/ablation.py eval_results/patch_scorer.json \
    --out eval_results/tables_patch_scorer.md \
    --json_out eval_results/ablation_patch_scorer.json

echo "=== ablation: base ==="
python eval/ablation.py eval_results/base_rerun.json \
    --out eval_results/tables_base.md \
    --json_out eval_results/ablation_base.json

echo "=== MODEL_03 EVAL COMPLETE ==="
ls -la eval_results/
