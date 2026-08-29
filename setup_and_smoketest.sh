#!/usr/bin/env bash
set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

REPO=~/ai-generated-image-detection
VENV=/tmp/aigc_venv
cd "$REPO/server/model_01"

echo "=== building venv at $VENV (node-local, no home quota) ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

echo "=== installing torch (pinned, self-contained cu124 build -- avoids the split-cudnn version-mismatch bug in newer torch releases) ==="
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== installing remaining requirements ==="
grep -v -iE "^torch" ../requirements.txt > /tmp/requirements_notorch.txt
pip install -r /tmp/requirements_notorch.txt

echo "=== torch/cuda sanity check (including an actual cudnn conv2d forward+backward on GPU) ==="
python -c "
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
x = torch.randn(2, 3, 32, 32, device='cuda', requires_grad=True)
conv = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
y = conv(x)
y.sum().backward()
print('cudnn conv2d forward+backward OK, output shape:', tuple(y.shape))
"

echo "=== running unit tests ==="
python tests/test_transforms.py
python tests/test_synthetic_dataset.py
python tests/test_download_cifake.py
python tests/test_model_shapes.py

echo "=== generating synthetic dataset ==="
python data/make_synthetic_dataset.py --out data/raw/synthetic --n_per_class 200

echo "=== quick smoke-test training run (synthetic data) ==="
python train.py --config configs/smoke_synthetic.yaml

echo "=== SMOKE TEST COMPLETE ==="
