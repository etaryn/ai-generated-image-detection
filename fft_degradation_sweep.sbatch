#!/usr/bin/env bash
#SBATCH --job-name=aigc-fftdeg
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/r/reyes/ai-generated-image-detection/logs/fftdeg-%j.out
#SBATCH --error=/home/r/reyes/ai-generated-image-detection/logs/fftdeg-%j.err

# Does model_02's FFT branch decay faster than DINOv2/CLIP under the RealDeg
# resize sweep? See server/model_02/eval/fft_degradation_ablation.py for the
# argument; this script just produces the caches it needs and runs it.
#
# Structure: build env -> SMOKE pass on 400 images (catches bugs in minutes) ->
# FULL pass on $N_IMAGES. The smoke pass is not optional: extraction is the
# expensive step and a typo found after it is a wasted allocation.

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

REPO=/home/r/reyes/ai-generated-image-detection
VENV=/tmp/aigc_venv_fftdeg
N_IMAGES=${N_IMAGES:-12000}

# $HOME on this cluster has a small quota and cannot hold the torch wheels or the
# DINOv2/CLIP weights -- route every cache to node-local /tmp.
export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_02"

echo "=== checking data is present (this job does NOT download it) ==="
for split in real fake; do
    n=$(ls "$REPO/data/raw/cifake/$split" | wc -l)
    echo "  cifake/$split: $n images"
    if [ "$n" -lt 1000 ]; then
        echo "ERROR: $REPO/data/raw/cifake/$split has only $n images." >&2
        exit 1
    fi
done

echo "=== building venv at $VENV (node-local, no home quota) ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

echo "=== installing torch (pinned cu124 build) ==="
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== installing remaining requirements ==="
grep -h -v -iE "^torch" ../requirements.txt requirements.txt > /tmp/requirements_notorch_fftdeg.txt
pip install -q -r /tmp/requirements_notorch_fftdeg.txt

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

CACHE=features/cache
mkdir -p "$CACHE"

# ---------------------------------------------------------------------------
# Extract one cache per condition. --aug-copies 0 everywhere: these caches must
# be one row per source image so the clean and degraded caches stay row-aligned
# (fft_degradation_ablation.py enforces this).
# ---------------------------------------------------------------------------
extract_sweep () {
    local tag=$1 limit=$2
    echo "=== [$tag] extracting CLEAN (${limit} images) ==="
    python extract_features.py --config configs/default.yaml \
        --limit "$limit" --aug-copies 0 --out "$CACHE/${tag}_clean.npz"

    for scale in 0.9 0.7 0.5 0.3 0.2; do
        echo "=== [$tag] extracting resize@${scale} ==="
        python extract_features.py --config configs/default.yaml \
            --limit "$limit" --aug-copies 0 \
            --realdeg-op resize --realdeg-strength "$scale" \
            --out "$CACHE/${tag}_resize_${scale}.npz"
    done
}

run_ablation () {
    local tag=$1 prefix=$2
    shift 2
    python eval/fft_degradation_ablation.py --config configs/default.yaml \
        --clean-cache "$CACHE/${tag}_clean.npz" \
        --degraded "resize_0.9=$CACHE/${tag}_resize_0.9.npz" \
        --degraded "resize_0.7=$CACHE/${tag}_resize_0.7.npz" \
        --degraded "resize_0.5=$CACHE/${tag}_resize_0.5.npz" \
        --degraded "resize_0.3=$CACHE/${tag}_resize_0.3.npz" \
        --degraded "resize_0.2=$CACHE/${tag}_resize_0.2.npz" \
        --out-prefix "$prefix" "$@"
}

echo ""
echo "############################################################"
echo "# SMOKE PASS (400 images) -- validates the whole path"
echo "############################################################"
extract_sweep smoke 400
run_ablation smoke fft_degradation_smoke
echo "=== SMOKE PASS OK ==="

echo ""
echo "############################################################"
echo "# FULL PASS ($N_IMAGES images)"
echo "############################################################"
extract_sweep full "$N_IMAGES"

echo "=== ablation: MLP (train on clean, score every condition) ==="
run_ablation full fft_degradation

echo "=== ablation: XGBoost (trees suit the hand-built FFT stats better) ==="
run_ablation full fft_degradation --classifier xgboost

echo "=== ablation: MLP, also retraining per condition (upper bound per strength) ==="
run_ablation full fft_degradation_retrained --retrain-per-condition

echo ""
echo "=== DONE. outputs: ==="
ls -la eval/outputs/
