"""Pull a bounded evaluation sample from SID-Set, with tamper masks.

model_03's whole claim is *localisation* -- that it finds where an edit is, not
just that one happened. Nothing in the unit tests can check that claim, because
they run against a stub scorer on a synthetic square. Checking it needs images
that were really edited by a real generator, together with ground-truth masks of
what was edited.

SID-Set (Huang et al., "SIDA: Social Media Image Deepfake Detection,
Localization and Explanation") is the dataset this repo's own README already
names, and it is the right shape for all three of model_03's open questions:

    label 0  real          authentic photographs (from OpenImages V7)
    label 1  full_synthetic  wholly generated images
    label 2  tampered      real photographs with a generated region, **plus a
                           binary mask of that region**

Three classes, not two, is what makes it usable here: model_03 is built around
the distinction between "this image was generated" and "this photograph was
locally edited", and a two-class dataset cannot measure that distinction at all.

**Which split, and why it matters.** The official test split is gated to prevent
leakage, so this uses **validation**. For model_03 itself that is sound -- it
trains on nothing, so no split is "seen" -- but the *backend* is a third-party
detector whose training data is unknown, and SID-Set's real images come from
OpenImages, which is in very many training sets. So read backend detection
numbers as an upper bound, and prefer the localisation numbers, which no
whole-image detector could have memorised the answer to.

The shards are ~500MB each and there are 34 of them; this reads parquet row
groups over HTTP range requests and stops as soon as it has enough, so a
few-hundred-image sample costs a few hundred MB rather than 17GB.

    python eval/fetch_sid_set.py --per_class 120 --out eval_data/sid_set_val
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

CLASS_NAMES = {0: "real", 1: "synthetic", 2: "tampered"}


SPLITS = {"validation": ("validation", 34), "train": ("train", 249)}


def fetch(
    out_dir: Path,
    per_class: int,
    max_row_groups: int,
    shard: int,
    split: str = "validation",
    max_side: int = 1024,
) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image

    if split not in SPLITS:
        raise SystemExit(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    prefix, shard_count = SPLITS[split]

    fs = HfFileSystem()
    path = (
        f"datasets/saberzl/SID_Set/data/"
        f"{prefix}-{shard:05d}-of-{shard_count:05d}.parquet"
    )
    handle = fs.open(path, "rb")
    parquet = pq.ParquetFile(handle)

    for name in CLASS_NAMES.values():
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in CLASS_NAMES.values()}
    manifest = []

    for group in range(min(max_row_groups, parquet.metadata.num_row_groups)):
        if all(c >= per_class for c in counts.values()):
            break
        table = parquet.read_row_group(group).to_pydict()

        for i, label in enumerate(table["label"]):
            name = CLASS_NAMES.get(int(label))
            if name is None or counts[name] >= per_class:
                continue

            img_id = table["img_id"][i]
            stem = f"{name}_{img_id}"
            image = Image.open(io.BytesIO(table["image"][i]["bytes"])).convert("RGB")
            if max_side and max(image.size) > max_side:
                # Training sets run to thousands of images; PNG at full size is
                # far more disk than the patches sampled from them need. The
                # mapper works at 1024 anyway, so anything above that is thrown
                # away at inference time too.
                factor = max_side / max(image.size)
                image = image.resize(
                    (max(1, round(image.width * factor)), max(1, round(image.height * factor))),
                    Image.BICUBIC,
                )
            image_path = out_dir / name / f"{stem}.png"
            image.save(image_path)

            row = {
                "stem": stem,
                "img_id": img_id,
                "label": int(label),
                "class": name,
                "image": str(image_path.relative_to(out_dir)).replace("\\", "/"),
                "size": list(image.size),
                "mask": None,
            }

            # Only tampered rows carry a mask; a real or fully-synthetic row has
            # nothing to localise, which is itself the ground truth those two
            # classes contribute (no region, and the whole frame respectively).
            mask_cell = table["mask"][i]
            if mask_cell is not None and mask_cell.get("bytes"):
                mask = Image.open(io.BytesIO(mask_cell["bytes"])).convert("L")
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.NEAREST)
                mask_path = out_dir / "masks" / f"{stem}.png"
                mask.save(mask_path)
                row["mask"] = str(mask_path.relative_to(out_dir)).replace("\\", "/")

            manifest.append(row)
            counts[name] += 1

        print(f"  row group {group}: {counts}")

    # Accumulate across shards: a multi-shard fetch calls this repeatedly into
    # the same directory, and a manifest that only described the last shard
    # would silently hide most of the data from every consumer.
    manifest_path = out_dir / "manifest.json"
    existing, shards = [], []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        existing = previous.get("items", [])
        shards = previous.get("summary", {}).get("shards", [])
        seen = {row["stem"] for row in existing}
        manifest = existing + [row for row in manifest if row["stem"] not in seen]

    summary = {
        "source": "saberzl/SID_Set",
        "split": split,
        "shards": sorted(set(shards + [shard])),
        "counts": {
            name: sum(1 for r in manifest if r["class"] == name)
            for name in CLASS_NAMES.values()
        },
        "with_masks": sum(1 for r in manifest if r["mask"]),
        "note": "the official test split is gated, so validation is used for scoring and "
                "train for fitting -- never both. SID-Set's real images come from "
                "OpenImages, which is widely used in training sets, so detection numbers "
                "are an upper bound; the localisation numbers are the ones to trust.",
    }
    manifest_path.write_text(json.dumps({"summary": summary, "items": manifest}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="eval_data/sid_set_val", help="Where to write the sample")
    parser.add_argument("--per_class", type=int, default=120, help="Images per class (real/synthetic/tampered)")
    parser.add_argument("--max_row_groups", type=int, default=9, help="Row groups to read before giving up")
    parser.add_argument("--shard", type=int, default=0, help="Which shard (validation 0-33, train 0-248)")
    parser.add_argument("--split", default="validation", choices=sorted(SPLITS),
                        help="train is for fitting a patch scorer; validation is for scoring it. "
                             "Never fit and score on the same split.")
    parser.add_argument("--shards", type=int, default=1,
                        help="Consecutive shards to read, starting at --shard")
    parser.add_argument("--max_side", type=int, default=1024,
                        help="Downscale long edge on save (0 disables)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    summaries = []
    for offset in range(args.shards):
        shard = args.shard + offset
        print(f"streaming SID-Set {args.split} shard {shard} -> {out_dir}", flush=True)
        summaries.append(
            fetch(out_dir, args.per_class, args.max_row_groups, shard, args.split, args.max_side)
        )
    print(json.dumps(summaries[-1] if len(summaries) == 1 else summaries, indent=2))


if __name__ == "__main__":
    main()
