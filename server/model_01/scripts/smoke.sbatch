#!/bin/bash
#SBATCH --job-name=aigc-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Cheap correctness pass before spending a gpu-long allocation:
#   1. the forward/backward shape test, which has never actually been executed
#   2. the overfit gate that scripts/train.sbatch depends on
#   3. one epoch on the network-free synthetic dataset
#
# Runs at 224px (configs/smoke.yaml) while the real run is 32px, so between the two
# both stem/resolution configurations get executed.
#
#   sbatch scripts/smoke.sbatch

set -euo pipefail

REPO="${REPO:-$HOME/ai-generated-image-detection}"
cd "$REPO"
source scripts/_bootstrap.sh

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python tests/test_model_shapes.py

# Generated onto node-local /tmp: 400 images in $HOME would be 400 inodes.
# Two variants:
#   synthetic         -- realistic, subtle artifact (amplitude at the noise floor).
#                        What train.py's one-epoch run uses.
#   synthetic_strong  -- same generator, obvious artifact. What the GATE uses.
# The gate is a wiring check, so it needs data whose signal is unambiguous: the
# subtle default is genuinely near-unlearnable in a few hundred steps, and pointing
# the gate at it produced a false "REAL BUG" verdict (0.5039) for code that scored
# 0.9961 on real CIFAKE minutes earlier.
python data/make_synthetic_dataset.py --out "$LOCAL_ROOT/raw/synthetic" --n_per_class 200
python data/make_synthetic_dataset.py --out "$LOCAL_ROOT/raw/synthetic_strong" \
    --n_per_class 200 --amplitude 20 30

# The same pre-flight gate scripts/train.sbatch uses, run here so the gate itself
# is exercised before it guards a real allocation.
python scripts/diagnose.py \
    --part overfit --gate --gate_min_acc 0.90 \
    --datasets synthetic_strong \
    --config configs/smoke.yaml --data_root "$LOCAL_ROOT/raw"

python train.py --config configs/smoke.yaml --data_root "$LOCAL_ROOT/raw"
