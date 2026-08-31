"""Helpers for downloading and laying out the challenge's datasets.

Each dataset has a different license/download mechanism, so this is a set of
documented, semi-manual steps rather than one fully automated pipeline — run with
`--dataset <name>` to print (and, where possible, execute) the steps for that
dataset. All datasets should land in:

    data/raw/<dataset_name>/real/*.jpg
    data/raw/<dataset_name>/fake/*.jpg

so that `data/datasets.py`'s `RealFakeImageDataset` can load them uniformly.

No network access yet, or just want to sanity-check the pipeline mechanics first?
See `data/make_synthetic_dataset.py` for a network-free synthetic stand-in dataset
(not a substitute for real data -- see its docstring).

IMPORTANT: the WildFake "Validation Dataset" subset named in the challenge brief
(COCO val2017 as real, DALL-E Advanced as fake) is for demonstration/tracking only
and must NOT be placed under a train_datasets root — keep it under
`data/raw/wildfake_demo/` (see configs/default.yaml's `demo_eval_set`) and never
pass that path to training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

RAW_DIR = Path("data/raw")

INSTRUCTIONS = {
    "generators": """
Multi-generator, full-resolution data (DALL-E 3 / MidJourney / ProGAN).

This one IS automated -- see server/model_02/prepare_generators.py:

    sbatch prepare_generators.sbatch        # from the repo root
    # or, off the login node:
    python server/model_02/prepare_generators.py --families all --per-class 5000

It writes three paired datasets in the usual layout, and both models can train on
them (model_01 would need its 224px stem schedule; see configs/default.yaml):

    data/raw/gen_progan/{real,fake}/        ProGAN vs LSUN      (ForenSynths, matched pairs)
    data/raw/gen_midjourney/{real,fake}/    MidJourney vs Open Images
    data/raw/gen_dalle3/{real,fake}/        DALL-E 3 vs Open Images

Do NOT run it on the login node: `ulimit -v` is 1GB there and both the Hugging
Face transfer backend and pyarrow blow through that immediately.

Every image is a native-resolution center crop, never a resize, with one uniform
JPEG re-encode across both classes. Both of those exist to kill shortcuts that
otherwise make the data separable at AUC>0.95 without any generator signal --
server/model_02/README.md has the measurements.
""",
    "sid_set": """
SID_Set (Hugging Face): https://huggingface.co/datasets/saberzl/SID_Set

    pip install huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='saberzl/SID_Set', repo_type='dataset',
                   local_dir='data/raw/_sid_set_hf')
"

Then reorganize the downloaded files into:
    data/raw/sid_set/real/*.jpg
    data/raw/sid_set/fake/*.jpg
(check the dataset card for its label field names — this varies by release).
""",
    "cifake": """
CIFAKE (Kaggle): https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

    python data/download_cifake.py --out data/raw/cifake

This downloads via kagglehub (prompts for Kaggle credentials once, then caches --
no manual kaggle.json setup needed) and automatically reorganizes CIFAKE's own
train/test x REAL/FAKE folders into:
    data/raw/cifake/real/*.jpg
    data/raw/cifake/fake/*.jpg
merging both splits together (this project does its own train/val split on top).
Pass --copy instead of the default symlinks if your filesystem doesn't support
them (e.g. Windows without Developer Mode enabled).
""",
    "wildfake": """
WildFake (ModelScope): https://modelscope.cn/datasets/hy2628982280/WildFake/summary

NOTE: per the dataset page, click the translation button on the ModelScope page
before browsing/downloading if you're not reading Chinese.

    pip install modelscope
    python -c "
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('hy2628982280/WildFake')
"

Reorganize the subset you choose to train on into:
    data/raw/wildfake/real/*.jpg
    data/raw/wildfake/fake/*.jpg

For the challenge's held-out demonstration subset specifically (COCO val2017 as
real, DALL-E Advanced as fake — 4998 / 8843 images), lay those out separately
under:
    data/raw/wildfake_demo/coco_val2017/*.jpg
    data/raw/wildfake_demo/dalle_advanced/*.jpg
and reference them via configs/default.yaml's `data.demo_eval_set`, never via
`train_datasets`.
""",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(INSTRUCTIONS.keys()) + ["all"],
        default="all",
        help="Which dataset's setup instructions to print.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    datasets = INSTRUCTIONS.keys() if args.dataset == "all" else [args.dataset]
    for name in datasets:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        print(INSTRUCTIONS[name])


if __name__ == "__main__":
    main()
