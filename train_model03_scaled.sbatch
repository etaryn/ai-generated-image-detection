#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-scaled
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_scaled_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_scaled_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03scaled
SCRATCH=/mnt/scratch/b/bing2812/aigc_m03

# The venv and the HF weight cache stay node-local (/tmp, ~9.8G, fast).
export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

# The data does NOT. 10 shards x 150/class is ~4.5k images, ~4.7G extrapolating
# from the 1.9G that 6 shards x 100/class produced. That would fit /tmp, but it
# still goes on /mnt/scratch so a resumed/rerun job doesn't have to refetch it.
DATA=$SCRATCH/sid_set_train
VAL=$SCRATCH/sid_set_val

if [ ! -d /mnt/scratch/b/bing2812 ]; then
    echo "FATAL: /mnt/scratch not mounted on $(hostname); refusing to fill /tmp" >&2
    exit 1
fi
mkdir -p "$DATA" "$VAL"
echo "=== scratch free ===" && df -h /mnt/scratch | tail -1
echo "=== /tmp free ===" && df -h /tmp | tail -1

cd "$REPO/server/model_03"

# The existing checkpoint (patch_auc_after=0.019, flip=true) is about to be
# overwritten in place. Keep it -- it is the only record of that run.
if [ -f checkpoints/patch_scorer/model.safetensors ]; then
    echo "=== backing up previous checkpoint -> $SCRATCH/patch_scorer_prev ==="
    rm -rf "$SCRATCH/patch_scorer_prev"
    cp -r checkpoints/patch_scorer "$SCRATCH/patch_scorer_prev"
fi

echo "=== building venv at $VENV ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

echo "=== installing torch (pinned cu124 build) ==="
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== installing model_03 requirements ==="
grep -v -iE "^torch" requirements.txt > /tmp/req_m03scaled.txt
pip install -r /tmp/req_m03scaled.txt
# Not in requirements.txt: pyarrow for the parquet range-request fetch, tqdm for
# the per-epoch progress bar.
pip install "pyarrow>=15" "tqdm>=4.66"

echo "=== torch/cuda sanity check ==="
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== plumbing tests ==="
python tests/test_pipeline.py
python tests/test_fast_preprocess.py

echo "=== [1/4] fetching SID-Set TRAIN, 10 shards x 150/class -> $DATA ==="
python eval/fetch_sid_set.py --split train --shard 0 --shards 10 \
    --per_class 150 --out "$DATA"
du -sh "$DATA"

echo "=== [2/4] fetching SID-Set VALIDATION shard 0, 120/class -> $VAL ==="
# fit_decision.py needs a manifest, and it must not be the split the scorer was
# fitted on. Same validation sample the recorded baseline used, so the swept
# numbers stay comparable to EVALUATION_RESULTS.md.
python eval/fetch_sid_set.py --split validation --shard 0 --per_class 120 --out "$VAL"

echo "=== [3/4] fine-tuning the patch scorer ==="
python scripts/train_patch_scorer.py \
    --data "$DATA" --out checkpoints/patch_scorer \
    --epochs 3 --batch_size 32 --hard_negatives 512

echo "=== checkpoint written ==="
ls -la checkpoints/patch_scorer/
cat checkpoints/patch_scorer/training.json

echo "=== [4/4] sweeping the decision rule ==="
mkdir -p eval_results
python scripts/fit_decision.py \
    --backend checkpoints/patch_scorer \
    --manifest "$VAL/manifest.json" \
    --limit 50 \
    --cache "$SCRATCH/decision_cache.npz" \
    --out eval_results/decision_sweep.json

echo "=== MODEL_03 SCALED RUN COMPLETE ==="
cat eval_results/decision_sweep.json
