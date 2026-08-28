#!/bin/bash
#SBATCH --job-name=aigc-buildenv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Build the Python environment ONCE and store it in $HOME as a single tarball.
#
#   sbatch scripts/build_env.sbatch
#
# Home is quota'd on INODES, not bytes: 500MB writes instantly, but a torch
# install unpacks ~50k files and blows the file-count limit. So the venv is built
# on the compute node's local /tmp (9.8G, unquota'd) and archived to one file.
# Every later job untars it back to the SAME absolute path.
#
# The path must be identical on build and restore -- a venv's activate script and
# console-script shebangs hardcode absolute paths, so a venv unpacked somewhere
# else than it was created points its shebangs at a python that isn't there.

set -euo pipefail

REPO="${REPO:-$HOME/ai-generated-image-detection}"
VENV_ROOT="/tmp/$USER"
VENV="$VENV_ROOT/venv"
ARCHIVE="${ARCHIVE:-$HOME/venv.tar.gz}"

cd "$REPO"
mkdir -p logs "$VENV_ROOT"
rm -rf "$VENV"

echo "node=$(hostname) building $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"

# --no-cache-dir throughout: the pip cache is pure inode waste here, and it would
# be discarded with the node's /tmp anyway.
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -r requirements.txt

python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# Write to .tmp then rename: a job killed mid-tar would otherwise leave a
# truncated archive that every later job would fail to unpack.
tar czf "$ARCHIVE.tmp" -C "$VENV_ROOT" venv
mv "$ARCHIVE.tmp" "$ARCHIVE"
ls -lh "$ARCHIVE"
echo "done -- later jobs restore this to $VENV"
