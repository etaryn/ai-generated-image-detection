#!/usr/bin/env bash
#SBATCH --job-name=aigc-augckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/r/reyes/ai-generated-image-detection/logs/augckpt-%j.out
#SBATCH --error=/home/r/reyes/ai-generated-image-detection/logs/augckpt-%j.err

# Job 772957 measured that mixture augmentation recovers ~75% of the oracle
# robustness headroom, but eval/fft_degradation_ablation.py fits a head per
# (block-combo, condition), scores it and DISCARDS it -- it writes metrics, not
# models. So there is no augmented checkpoint to serve. This builds one.
#
# Two checkpoints, not one. checkpoints/best.pt was written at 09:59 on 2026-08-30,
# BEFORE the QuickGELU fix (cc2fa22, 11:23), so its CLIP features were extracted
# with the wrong activation -- infer.py::_warn_if_quickgelu_stale exists for exactly
# that case. A/B-ing augmented-vs-best.pt would therefore confound augmentation with
# the QuickGELU fix. Both caches below were extracted after the fix (12:10 and 13:13),
# so fitting a clean-cache baseline here makes the comparison one-variable.
#
# Feature extraction is NOT repeated: both caches are already on disk. This only
# fits the classifier head, so it is minutes of GPU, not the 20m of job 772957.

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/r/reyes/ai-generated-image-detection
VENV=/tmp/aigc_venv_augckpt

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_02"
CACHE=features/cache

for f in full_clean full_augtrain; do
    [ -f "$CACHE/$f.npz" ] || { echo "ERROR: missing $CACHE/$f.npz" >&2; exit 1; }
    echo "  ok $f.npz"
done

echo "=== building venv at $VENV ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -h -v -iE "^torch" ../requirements.txt requirements.txt > /tmp/requirements_notorch_augckpt.txt
pip install -q -r /tmp/requirements_notorch_augckpt.txt

# train.py group-splits on the cache's own `groups`, so the augmented cache holds out
# whole source images together with their augmented copies -- no leakage across the
# split, and the same held-out images as the sweep (same --limit and seed).
#
# --out is passed explicitly in both cases so neither run touches checkpoints/best.pt;
# the frontend keeps serving the current model until you point it elsewhere.

echo ""
echo "=== baseline: fit on CLEAN cache (post-QuickGELU-fix) ==="
python train.py --config configs/default.yaml \
    --cache "$CACHE/full_clean.npz" \
    --out checkpoints/best_clean12k.pt

echo ""
echo "=== treatment: fit on AUGMENTED mixture cache ==="
python train.py --config configs/default.yaml \
    --cache "$CACHE/full_augtrain.npz" \
    --out checkpoints/best_augtrain.pt

echo ""
echo "=== DONE. checkpoints: ==="
ls -la checkpoints/
echo ""
echo "Serve either one from the Streamlit bench without editing code:"
echo "  export AIGC_MODEL02_CHECKPOINT=$REPO/server/model_02/checkpoints/best_augtrain.pt"
echo "  streamlit run $REPO/client/app.py"
