#!/usr/bin/env bash
#SBATCH --job-name=aigc-augtrain
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/r/reyes/ai-generated-image-detection/logs/augtrain-%j.out
#SBATCH --error=/home/r/reyes/ai-generated-image-detection/logs/augtrain-%j.err

# Follow-up to job 772779. That run showed a +0.21..+0.39 balanced-accuracy gap
# between training on clean features and refitting per degradation condition --
# but refitting per condition is an ORACLE: it assumes the test-time degradation
# is known. The realistic middle is training on a MIXTURE of degradations, which
# is what model_02's `train_aug_copies` already does and what my clean-only
# caches deliberately switched off.
#
# So: build ONE augmented training cache (--aug-copies 1, i.e. model_02's own
# default regime) and re-run the sweep against the SIX eval caches job 772779
# already wrote. Only the training cache is new; extraction of the eval caches
# is not repeated.

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

REPO=/home/r/reyes/ai-generated-image-detection
VENV=/tmp/aigc_venv_augtrain
N_IMAGES=${N_IMAGES:-12000}

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_02"
CACHE=features/cache

echo "=== checking the eval caches from job 772779 are still on disk ==="
for f in full_clean full_resize_0.9 full_resize_0.7 full_resize_0.5 full_resize_0.3 full_resize_0.2; do
    if [ ! -f "$CACHE/$f.npz" ]; then
        echo "ERROR: missing $CACHE/$f.npz -- re-run fft_degradation_sweep.sbatch first." >&2
        exit 1
    fi
    echo "  ok $f.npz"
done

echo "=== building venv at $VENV ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -h -v -iE "^torch" ../requirements.txt requirements.txt > /tmp/requirements_notorch_augtrain.txt
pip install -q -r /tmp/requirements_notorch_augtrain.txt
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

# --aug-copies 1 -> 2 rows per source image (clean + one randomised RobustnessAugment
# pass). Same --limit and seed as job 772779, so the group ids match the eval caches
# and the held-out source images are identical.
echo "=== extracting AUGMENTED training cache (${N_IMAGES} images x 2 copies) ==="
python extract_features.py --config configs/default.yaml \
    --limit "$N_IMAGES" --aug-copies 1 --out "$CACHE/full_augtrain.npz"

run_sweep () {
    local prefix=$1; shift
    python eval/fft_degradation_ablation.py --config configs/default.yaml \
        --train-cache "$CACHE/full_augtrain.npz" \
        --clean-cache "$CACHE/full_clean.npz" \
        --degraded "resize_0.9=$CACHE/full_resize_0.9.npz" \
        --degraded "resize_0.7=$CACHE/full_resize_0.7.npz" \
        --degraded "resize_0.5=$CACHE/full_resize_0.5.npz" \
        --degraded "resize_0.3=$CACHE/full_resize_0.3.npz" \
        --degraded "resize_0.2=$CACHE/full_resize_0.2.npz" \
        --out-prefix "$prefix" "$@"
}

echo "=== sweep: MLP, fit on augmented mixture ==="
run_sweep fft_augtrain

echo "=== sweep: XGBoost, fit on augmented mixture ==="
run_sweep fft_augtrain --classifier xgboost

echo ""
echo "=== DONE. outputs: ==="
ls -la eval/outputs/
