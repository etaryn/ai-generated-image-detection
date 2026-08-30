#!/usr/bin/env bash
#SBATCH --job-name=aigc-prov
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/r/reyes/ai-generated-image-detection/logs/prov-%j.out
#SBATCH --error=/home/r/reyes/ai-generated-image-detection/logs/prov-%j.err

# infer.py's staleness guard compared the checkpoint's REQUESTED clip backbone
# ("ViT-B-16", straight out of the config) against the QuickGELU-resolved one. That
# differs for every checkpoint ever written, pre-fix and post-fix alike, so the guard
# warned on all of them and discriminated nothing. The fact that actually settles it --
# which tower ran at extraction time -- was never recorded: features/clip.py set
# self.resolved_backbone_name and nothing read it.
#
# The code now stamps `resolved_backbones` into cache metadata, carries it into the
# checkpoint, and compares it against the live serving stack, treating a MISSING field
# as stale (only pre-fix caches lack it). This re-extracts the two caches the frontend
# A/B needs so they carry the field, then refits both checkpoints from them.
#
# The features themselves are unchanged: both caches were already extracted after the
# QuickGELU fix (12:10 and 13:13 on 2026-08-30) and the only edit to the extraction
# path is the metadata field. Same --limit and seed, so the rows are reproduced, not
# altered -- this buys correct provenance, not different numbers.

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/r/reyes/ai-generated-image-detection
VENV=/tmp/aigc_venv_prov
N_IMAGES=${N_IMAGES:-12000}

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_02"
CACHE=features/cache

echo "=== building venv at $VENV ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -h -v -iE "^torch" ../requirements.txt requirements.txt > /tmp/requirements_notorch_prov.txt
pip install -q -r /tmp/requirements_notorch_prov.txt

# Extract to a sidecar name and move over the original only on success, so a failure
# partway cannot leave a truncated 190MB cache where a good one used to be.
echo "=== re-extracting full_clean (${N_IMAGES} images, aug-copies 0) ==="
python extract_features.py --config configs/default.yaml \
    --limit "$N_IMAGES" --aug-copies 0 --out "$CACHE/full_clean.stamped.npz"

echo "=== re-extracting full_augtrain (${N_IMAGES} images, aug-copies 1) ==="
python extract_features.py --config configs/default.yaml \
    --limit "$N_IMAGES" --aug-copies 1 --out "$CACHE/full_augtrain.stamped.npz"

echo "=== confirming the new field is present, then swapping in ==="
python - <<'PY'
import json, numpy as np, sys
for n in ["full_clean", "full_augtrain"]:
    m = json.loads(str(np.load(f"features/cache/{n}.stamped.npz", allow_pickle=True)["meta"]))
    rb = m.get("resolved_backbones")
    print(f"  {n}: resolved_backbones={rb}")
    if not rb or "clip" not in rb:
        sys.exit(f"ERROR: {n} did not record resolved_backbones -- not swapping it in.")
PY
mv "$CACHE/full_clean.stamped.npz"     "$CACHE/full_clean.npz"
mv "$CACHE/full_augtrain.stamped.npz"  "$CACHE/full_augtrain.npz"

echo ""
echo "=== refitting baseline checkpoint (clean cache) ==="
python train.py --config configs/default.yaml \
    --cache "$CACHE/full_clean.npz" --out checkpoints/best_clean12k.pt

echo ""
echo "=== refitting augmented checkpoint (mixture cache) ==="
python train.py --config configs/default.yaml \
    --cache "$CACHE/full_augtrain.npz" --out checkpoints/best_augtrain.pt

# End-to-end check of the guard itself: the old best.pt lacks the field and must warn,
# the two new ones carry it and must stay silent. A guard that cannot tell these apart
# is the bug this job exists to fix, so assert it rather than eyeball it.
echo ""
echo "=== verifying the staleness guard discriminates ==="
python - <<'PY'
import warnings, sys
from infer import load_model

expected = {"checkpoints/best.pt": True,
            "checkpoints/best_clean12k.pt": False,
            "checkpoints/best_augtrain.pt": False}
bad = []
for ckpt, should_warn in expected.items():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_model(ckpt, device="cpu")
    hits = [w for w in caught if issubclass(w.category, RuntimeWarning)
            and "backbone" in str(w.message).lower()]
    got = bool(hits)
    print(f"  {ckpt}: warned={got} (expected {should_warn})")
    if hits:
        print(f"      -> {str(hits[0].message)[:150]}")
    if got != should_warn:
        bad.append(ckpt)
if bad:
    sys.exit(f"GUARD CHECK FAILED for: {bad}")
print("  guard discriminates correctly")
PY

echo ""
echo "=== DONE ==="
ls -la checkpoints/
