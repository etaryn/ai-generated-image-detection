#!/usr/bin/env bash
#SBATCH --job-name=aigc-genext
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/r/reyes/ai-generated-image-detection/logs/genext_%j.out
#SBATCH --error=/home/r/reyes/ai-generated-image-detection/logs/genext_%j.err

# Step 1 + Step 2 for model_02 on the multi-generator data prepared by
# prepare_generators.sbatch.
#
#   1. one feature cache per generator family  (the expensive part: DINOv2 + CLIP
#      + FFT over 28k images x 2 augmentation copies)
#   2. an in-distribution classifier over all three families
#   3. three leave-one-generator-out classifiers
#   4. each LOO checkpoint scored against the family it never saw
#
# Extraction runs once per family and every fold reuses those caches, so adding a
# fold costs seconds rather than another pass through the backbones.
#
# Prerequisites:
#   sbatch prepare_generators.sbatch                     # writes data/raw/gen_*
#   tar cf ~/gen_datasets.tar -C data/raw gen_progan gen_midjourney gen_dalle3

set -euo pipefail

REPO=/home/r/reyes/ai-generated-image-detection
LOCAL_ROOT="/tmp/$USER"
VENV="$LOCAL_ROOT/venv"
VENV_ARCHIVE="${VENV_ARCHIVE:-$HOME/venv.tar.gz}"
DATA_ARCHIVE="${DATA_ARCHIVE:-$HOME/gen_datasets.tar}"

echo "=== node: $(hostname) job=${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

mkdir -p "$LOCAL_ROOT" "$REPO/logs"

# Caches to node-local disk: $HOME is inode-quota'd on this cluster and the HF
# weight cache is many small files.
export HF_HOME="$LOCAL_ROOT/hf_cache"
export TORCH_HOME="$LOCAL_ROOT/torch_cache"
export XDG_CACHE_HOME="$LOCAL_ROOT/xdg_cache"
export MPLCONFIGDIR="$LOCAL_ROOT/mpl_cache"
export TMPDIR="$LOCAL_ROOT/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

# --- venv -------------------------------------------------------------------- #
# Same restore-from-tarball trick as model_01/scripts/_bootstrap.sh. The .ready
# marker is written only after a complete extract, so a venv another job on this
# node is still unpacking is treated as absent rather than used half-built.
if [[ ! -f "$VENV/.ready" ]]; then
    if [[ ! -f "$VENV_ARCHIVE" ]]; then
        echo "ERROR: $VENV_ARCHIVE not found (build it with model_01/scripts/build_env.sbatch)" >&2
        exit 1
    fi
    echo "=== restoring venv: $VENV_ARCHIVE -> $VENV ==="
    rm -rf "$VENV"
    tar xzf "$VENV_ARCHIVE" -C "$LOCAL_ROOT"
    touch "$VENV/.ready"
fi
source "$VENV/bin/activate"

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# --- dataset ----------------------------------------------------------------- #
# 28k small files read over NFS is the slow part of extraction; unpack locally.
DATA_ROOT="$REPO/data/raw"
if [[ -f "$DATA_ARCHIVE" ]]; then
    echo "=== restoring dataset: $DATA_ARCHIVE -> $LOCAL_ROOT/raw ==="
    mkdir -p "$LOCAL_ROOT/raw"
    tar xf "$DATA_ARCHIVE" -C "$LOCAL_ROOT/raw"
    DATA_ROOT="$LOCAL_ROOT/raw"
else
    echo "=== $DATA_ARCHIVE absent, reading from $DATA_ROOT over NFS (slower) ==="
fi
for f in gen_progan gen_midjourney gen_dalle3; do
    echo "  $f: real=$(ls "$DATA_ROOT/$f/real" | wc -l) fake=$(ls "$DATA_ROOT/$f/fake" | wc -l)"
done

cd "$REPO/server/model_02"
CFG=configs/generators.yaml
CACHE=features/cache
mkdir -p "$CACHE" checkpoints

# --- plumbing gate ----------------------------------------------------------- #
# Cheap, and it fails in seconds rather than after a 40-minute extraction.
echo "=== plumbing tests ==="
python tests/test_fft_features.py
# Runs the real DINOv2 and open_clip architectures with random weights, so it
# catches a backbone API break (this venv ships transformers 5) in seconds
# rather than after the weight download at the start of extraction.
python tests/test_backbone_extractors.py
python tests/test_pipeline.py
python tests/test_canonical_crop.py

# Verify the configured CLIP backbone actually exists with the requested weights
# BEFORE the long extraction. open_clip silently degrades a QuickGELU mismatch
# instead of failing, so the name in the config is load-bearing.
python - <<'PYCHK'
import yaml, open_clip
cfg = yaml.safe_load(open("configs/generators.yaml"))["features"]["clip"]
name, tag = cfg["backbone_name"], cfg["pretrained"]
assert name in open_clip.list_models(), f"{name!r} not in this open_clip build"
tags = open_clip.list_pretrained_tags_by_model(name)
assert tag in tags, f"{tag!r} not available for {name!r}; have {tags}"
if tag == "openai" and not name.endswith("-quickgelu"):
    raise SystemExit(f"{name!r} + openai weights = QuickGELU mismatch; use {name}-quickgelu")
print(f"clip backbone OK: {name} / {tag}")
PYCHK

# --- Step 1: one cache per family -------------------------------------------- #
for f in progan midjourney dalle3; do
    if [[ -f "$CACHE/gen_$f.npz" ]]; then
        echo "=== cache for gen_$f exists, skipping extraction ==="
        continue
    fi
    echo "=== extracting gen_$f ==="
    python extract_features.py --config "$CFG" \
        --datasets "gen_$f" \
        --data-root "$DATA_ROOT" \
        --out "$CACHE/gen_$f.npz"
done
ls -lh "$CACHE"/gen_*.npz

# --- Step 2a: in-distribution ------------------------------------------------ #
echo "=== training on all three families (in-distribution) ==="
python train.py --config "$CFG" \
    --cache "$CACHE"/gen_progan.npz "$CACHE"/gen_midjourney.npz "$CACHE"/gen_dalle3.npz \
    --out checkpoints/generators_all.pt

# --- Step 2b: leave-one-generator-out ---------------------------------------- #
# The number that actually matters. Each fold trains on two families and is then
# scored against the third, which it has never seen.
for held in progan midjourney dalle3; do
    train_caches=()
    for f in progan midjourney dalle3; do
        [[ "$f" != "$held" ]] && train_caches+=("$CACHE/gen_$f.npz")
    done
    echo "=== LOO fold: holding out gen_$held ==="
    python train.py --config "$CFG" \
        --cache "${train_caches[@]}" \
        --out "checkpoints/loo_$held.pt"
done

# --- Step 3: score every LOO checkpoint on its held-out family ---------------- #
echo ""
echo "########################################################################"
echo "# LEAVE-ONE-GENERATOR-OUT: held-out transfer"
echo "########################################################################"
for held in progan midjourney dalle3; do
    python eval/score_cache.py \
        --checkpoint "checkpoints/loo_$held.pt" \
        --cache "$CACHE/gen_$held.npz" \
        --json-out "eval/outputs/loo_$held.json"
done

echo ""
echo "########################################################################"
echo "# IN-DISTRIBUTION: all-families checkpoint on each family"
echo "########################################################################"
python eval/score_cache.py \
    --checkpoint checkpoints/generators_all.pt \
    --cache "$CACHE"/gen_progan.npz "$CACHE"/gen_midjourney.npz "$CACHE"/gen_dalle3.npz \
    --per-cache --json-out eval/outputs/in_distribution.json

echo ""
echo "=== EXTRACTION + TRAINING COMPLETE ==="
ls -la checkpoints/
