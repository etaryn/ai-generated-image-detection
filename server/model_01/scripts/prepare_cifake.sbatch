#!/bin/bash
#SBATCH --job-name=aigc-data
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Turns the downloaded CIFAKE zip into the single-file dataset archive the
# training jobs expect:
#
#   sbatch scripts/prepare_cifake.sbatch
#   DATA_ARCHIVE=$HOME/cifake.tar sbatch scripts/train.sbatch
#
# Same $HOME-is-inode-quota'd constraint as the venv (see build_env.sbatch):
# CIFAKE is ~120k tiny files, so it cannot be laid out in $HOME. It is unpacked
# and reorganized on the node's local /tmp and only the finished tar -- one
# inode -- lands in $HOME.
#
# No GPU and no network: the zip is fetched separately (the login node has
# outbound access; compute nodes are not guaranteed to), so this job only
# unpacks, relabels and repacks. That's why it can sit on the CPU partition and
# start immediately instead of queueing behind GPU work.

set -euo pipefail

REPO="${REPO:-$HOME/ai-generated-image-detection}"
ZIP="${ZIP:-$HOME/cifake.zip}"
ARCHIVE="${ARCHIVE:-$HOME/cifake.tar}"

WORK="/tmp/$USER/dataprep"
SRC="$WORK/src"
OUT="$WORK/out"

cd "$REPO"
mkdir -p logs

if [[ ! -f "$ZIP" ]]; then
    echo "ERROR: $ZIP not found. Download it first (from a node with outbound access):" >&2
    echo "  curl -sSL -o \$HOME/cifake.zip \\" >&2
    echo "    https://www.kaggle.com/api/v1/datasets/download/birdy654/cifake-real-and-ai-generated-synthetic-images" >&2
    exit 1
fi

echo "node=$(hostname) job=${SLURM_JOB_ID:-none}"
rm -rf "$WORK"
mkdir -p "$SRC" "$OUT"

echo "unzipping $ZIP -> $SRC"
unzip -q "$ZIP" -d "$SRC"

# Reuse the repo's own layout function rather than reimplementing the REAL/FAKE
# walk here, so this path stays in sync with data/download_cifake.py (and with
# the unit tests that cover it). symlink=False because the result gets tarred:
# symlinks into $SRC would be dangling everywhere else.
echo "laying out real/fake -> $OUT/cifake"
python3 - "$SRC" "$OUT/cifake" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, ".")
from data.download_cifake import layout_from_source

counts = layout_from_source(Path(sys.argv[1]), Path(sys.argv[2]), symlink=False)
print(f"laid out {counts['real']} real + {counts['fake']} fake images")
PY

# Write to .tmp then rename, matching build_env.sbatch: a job killed mid-tar
# would otherwise leave a truncated archive every later job fails to unpack.
echo "packing $ARCHIVE"
tar cf "$ARCHIVE.tmp" -C "$OUT" cifake
mv "$ARCHIVE.tmp" "$ARCHIVE"

rm -rf "$WORK"
ls -lh "$ARCHIVE"
echo "done -- run training with: DATA_ARCHIVE=$ARCHIVE sbatch scripts/train.sbatch"
