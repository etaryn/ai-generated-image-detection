#!/usr/bin/env bash
#SBATCH --job-name=aigc-m03-diag
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_diag_%j.out
#SBATCH --error=/home/b/bing2812/ai-generated-image-detection/slurm_logs/model03_diag_%j.err

set -euo pipefail

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

REPO=/home/b/bing2812/ai-generated-image-detection
VENV=/tmp/aigc_venv_m03diag

export PIP_CACHE_DIR=/tmp/pip_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/mpl_cache

cd "$REPO/server/model_03"

# Same shard 4 / 50-per-class data as job 779368, so every arm here is directly
# comparable to eval_results/robustness_sidset_shard4_*.json.
#
# Job 779368 showed the region stage failing three different ways under
# degradation: over-firing at jpeg_q30 (2.55 regions/img, 147/150 images called
# AI), switching off at noise_s0.05 (0.09 regions/img, 139/150 called
# authentic), and firing at a normal rate in the wrong place at downscale_0.25
# (IoU 0.416 -> 0.123). The first two are common-mode shifts of the map's score
# distribution against the *absolute* cuts threshold_lo/hi, so this run tests
# whether removing that shift fixes them -- and what it costs on clean data.
DATA=/tmp/aigc_m03_sidset/sid_set_val_shard4

echo "=== venv ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -iE "^torch" requirements.txt > /tmp/req_m03diag.txt
pip install -r /tmp/req_m03diag.txt
pip install "pyarrow>=15" "tqdm>=4.66"

python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== fetching SID-Set VALIDATION shard 4, 50/class (same as 779368) ==="
mkdir -p "$DATA"
python eval/fetch_sid_set.py --split validation --shard 4 --per_class 50 --out "$DATA"

mkdir -p eval_results
COMMON="--data_dir $DATA --limit 50 --max_side 768"

echo "=== [1/4] patch scorer, ABSOLUTE cuts (instrumented baseline) ==="
python eval/robustness.py $COMMON \
    --backend checkpoints/patch_scorer \
    --threshold_mode absolute \
    --out eval_results/diag_shard4_ps_absolute.json

# median_shift corrects toward a reference median. Take it from this run's own
# clean condition rather than hardcoding one, so the constant is measured on the
# same backend and working resolution it will be applied at.
REF=$(python - <<'PY'
import json
d = json.load(open("eval_results/diag_shard4_ps_absolute.json"))
rows = d["per_image"]["clean"]
vals = [r["map"]["median_score"] for r in rows]
print(sum(vals) / len(vals))
PY
)
echo "=== clean-corpus reference median for median_shift: $REF ==="

echo "=== [2/4] patch scorer, QUANTILE cuts (shift-invariant) ==="
python eval/robustness.py $COMMON \
    --backend checkpoints/patch_scorer \
    --threshold_mode quantile \
    --out eval_results/diag_shard4_ps_quantile.json

echo "=== [3/4] patch scorer, MEDIAN_SHIFT cuts (ref=$REF) ==="
python eval/robustness.py $COMMON \
    --backend checkpoints/patch_scorer \
    --threshold_mode median_shift --adaptive_ref_median "$REF" \
    --out eval_results/diag_shard4_ps_median_shift.json

echo "=== [4/4] base backend, ABSOLUTE cuts (paired reference arm) ==="
python eval/robustness.py $COMMON \
    --backend hf:Organika/sdxl-detector \
    --threshold_mode absolute \
    --out eval_results/diag_shard4_base_absolute.json

echo "=== MODEL_03 THRESHOLD DIAGNOSTIC COMPLETE ==="
ls -la eval_results/diag_shard4_*.json
