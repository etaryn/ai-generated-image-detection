"""Verify a patch-scorer backend before trusting a single map it produces.

`mapper/labels.py` resolves which output means "AI-generated" from the model's
own `id2label`, and refuses to guess when the labels are unfamiliar. That
removes one failure mode and not the other: **a model whose uploaded config
labels its classes in the wrong order will resolve "correctly" and score
backwards.** Nothing in the resolution logic can detect that, because the config
is the only statement of intent available.

There is one reliable way to find out, and it is empirical: score images whose
labels you already know and check which way the scores point. This script does
that.

    python scripts/check_backend.py --data_dir data/raw/cifake --backend hf

It reports mean score per class and AUC, then a verdict:

    CORRECT             AI images score higher. Use it.
    INVERTED            AI images score *lower*. Either the model's config has
                        its labels reversed, or the model does not work on this
                        kind of image at all. Do not use it until you know
                        which -- `backend.positive_index` can flip it, but only
                        do that once you are sure the cause is the config.
    NON-DISCRIMINATIVE  AUC near 0.5. The model has no signal on this data;
                        the map it produces would be noise.

Measured on this repo's CIFAKE samples, 60 images per class. Read these as a
demonstration of the check, *not* as a ranking of the models: CIFAKE is 32x32,
which is far out of distribution for every one of these 224px detectors.

    backend                             fake   real   AUC    verdict
    Organika/sdxl-detector              0.342  0.131  0.696  CORRECT
    dima806/ai_vs_real_image_detection  0.991  0.025  1.000  CORRECT
    Ateeqq/ai-vs-human-image-detector   0.581  0.764  0.454  NON-DISCRIMINATIVE

Two things in that table are worth more than the ordering:

* dima806's labels are *reversed* relative to the others ({0: REAL, 1: FAKE}),
  and it still comes out CORRECT -- name-based resolution handled the flip. A
  hard-coded index would have produced a clean-looking AUC of 0.000 here.
  Its perfect score should be treated with suspicion rather than admiration,
  though: a 1.000 on a public benchmark most often means that benchmark was in
  the training set, and tells you nothing about held-out generators.
* Ateeqq is not broken; it saturates (both medians ~0.997) because 32px
  upscaled thumbnails are nothing like its training data. On real photographs it
  may well be the better model. That is exactly why this check has to be run on
  data resembling what you will actually analyse -- a backend that passes on
  thumbnails can fail on photographs, and vice versa.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapper.backends import build_backend  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path, limit: int):
    paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]
    return paths[:limit]


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney U / rank-based AUC, ties counted as half. No sklearn needed."""
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = np.empty_like(combined, dtype=np.float64)
    order = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[order]

    # Average ranks within tied groups, so a saturated detector is not flattered.
    i = 0
    current = np.empty(combined.size, dtype=np.float64)
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        current[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks[order] = current

    n_pos = positive.size
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * negative.size))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Folder with real/ and fake/ subdirectories")
    parser.add_argument("--backend", default="hf",
                        help="'hf', 'hf:<hub id>', a bare Hub id, or 'model_01' / 'model_02'")
    parser.add_argument("--limit", type=int, default=100, help="Images per class")
    parser.add_argument("--min_auc", type=float, default=0.60,
                        help="Below this the backend is reported NON-DISCRIMINATIVE")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fake_paths = iter_images(data_dir / "fake", args.limit)
    real_paths = iter_images(data_dir / "real", args.limit)
    if not fake_paths or not real_paths:
        raise SystemExit(
            f"Need images in both {data_dir / 'real'} and {data_dir / 'fake'}; "
            f"found {len(real_paths)} real and {len(fake_paths)} fake."
        )

    scorer = build_backend(args.backend)
    describe = getattr(scorer, "describe", None)
    if describe:
        info = describe()
        print(f"backend  : {info['backend']}")
        print(f"labels   : {info['id2label']}")
        print(f"AI class : {info['positive_indices']} -- {info['positive_resolved_by']}")
    else:
        print(f"backend  : {getattr(scorer, 'name', args.backend)}")

    def score(paths):
        images = []
        for path in paths:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        return np.array(scorer.score_patches(images), dtype=np.float64)

    print(f"scoring {len(fake_paths)} fake + {len(real_paths)} real images ...")
    fake_scores, real_scores = score(fake_paths), score(real_paths)

    area = auc(fake_scores, real_scores)
    print()
    print(f"  AI-labelled images   mean {fake_scores.mean():.3f}  median {np.median(fake_scores):.3f}")
    print(f"  real images          mean {real_scores.mean():.3f}  median {np.median(real_scores):.3f}")
    print(f"  AUC                  {area:.3f}")
    print()

    if area >= args.min_auc:
        print("  VERDICT: CORRECT -- AI images score higher. Safe to use.")
        return 0
    if area <= 1.0 - args.min_auc:
        print(
            "  VERDICT: INVERTED -- AI images score LOWER than real ones.\n"
            "  Either this model's uploaded config lists its labels in the wrong order,\n"
            "  or the model does not work on this kind of image. Find out which before\n"
            "  using it: backend.positive_index can flip the mapping, but flipping it to\n"
            "  paper over a model that simply has no signal here would be worse than not\n"
            "  using the model at all."
        )
        return 1
    print(
        f"  VERDICT: NON-DISCRIMINATIVE -- AUC {area:.3f} is near chance. This backend has\n"
        f"  no signal on this data, so any map built from it would be noise. Try another\n"
        f"  backend, or test on data closer to what you will actually analyse."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
