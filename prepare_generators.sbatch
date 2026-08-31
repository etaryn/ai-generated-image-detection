#!/usr/bin/env bash
#SBATCH --job-name=aigc-prepgen
#SBATCH --partition=long
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/r/reyes/ai-generated-image-detection/logs/prepgen_%j.out
#SBATCH --error=/home/r/reyes/ai-generated-image-detection/logs/prepgen_%j.err

# Downloads the multi-generator training data (MidJourney / DALL-E 3 / ProGAN,
# each paired with a matched real corpus) into data/raw/.
#
# This runs as a batch job rather than on the login node for a hard reason: the
# login node enforces `ulimit -v` = 1GB of virtual memory, and both the Hugging
# Face transfer backend and pyarrow's parquet reader blow through that instantly
# ("memory allocation of 67088607 bytes failed"). Compute nodes have no such cap.
#
# CPU-only -- this is network + JPEG decode + crop, no GPU involved. `long` is
# used over `normal` because a ~20GB pull can exceed the 3h limit on a slow link.

set -euo pipefail

echo "=== node: $(hostname) ==="
echo "=== ulimit -v: $(ulimit -v) (login node is 1048576; should be unlimited here) ==="
free -g | head -2

REPO=/home/r/reyes/ai-generated-image-detection
VENV="$REPO/.venv-data"

# Caches go to node-local /tmp: $HOME/.cache is on the shared filesystem and the
# login node's /tmp carries a 50MB per-user quota. Shards are deleted as they are
# consumed, so this only ever holds ~one shard at a time.
export TMPDIR=/tmp/$USER/prepgen_tmp
export HF_HOME=/tmp/$USER/hf_cache
export XDG_CACHE_HOME=/tmp/$USER/xdg_cache
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$REPO/logs"

# The venv lives on the shared filesystem (built once on the login node) so it is
# already visible here. Rebuild node-locally only if that turns out not to work.
if ! "$VENV/bin/python" -c "import huggingface_hub, pyarrow, PIL, py7zr" 2>/dev/null; then
    echo "=== shared venv unusable, building node-local ==="
    VENV=/tmp/$USER/prepgen_venv
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q huggingface_hub pyarrow pillow py7zr tqdm
fi

"$VENV/bin/python" -c "
import huggingface_hub, pyarrow, PIL
print('hf_hub', huggingface_hub.__version__, '| pyarrow', pyarrow.__version__, '| pillow', PIL.__version__)"

cd "$REPO/server/model_02"

# Order matters only for how fast you get something usable: progan is the small,
# fully-matched pair and lands in minutes, so a failure in the big pulls still
# leaves one complete family on disk.
echo "=== preparing generator families (per-class target: ${PER_CLASS:-5000}) ==="
"$VENV/bin/python" prepare_generators.py \
    --families progan midjourney dalle3 \
    --per-class "${PER_CLASS:-5000}" \
    --crop-size 256

echo "=== resulting layout ==="
for d in "$REPO"/data/raw/gen_*; do
    [ -d "$d" ] || continue
    echo "  $(basename "$d"): real=$(find "$d/real" -type f \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | wc -l) fake=$(find "$d/fake" -type f \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | wc -l)  $(du -sh "$d" | cut -f1)"
done

echo "=== PREPARE COMPLETE ==="
