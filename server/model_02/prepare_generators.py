"""Downloads and lays out full-resolution, multi-generator training data for model_02.

Why this exists
---------------
Both models currently only have CIFAKE on disk: 120k images, all 32x32, all fakes
from a single Stable-Diffusion family. That measures nothing about cross-generator
transfer, which is what the challenge actually probes. This script builds three
*paired* real/fake datasets at native resolution, one per generator family:

    gen_midjourney/   MidJourney (GenImage)  vs  ImageNet
    gen_dalle3/       DALL-E 3               vs  Open Images v7
    gen_progan/       ProGAN (ForenSynths)   vs  LSUN

laid out in the same `<root>/real/*.png` + `<root>/fake/*.png` shape that
`model_01/data/datasets.py`'s `RealFakeImageDataset` already reads, so both models
consume them without any loader changes.

The resolution trap this script exists to avoid
-----------------------------------------------
Every AI-generated source is a fixed square -- MidJourney and DALL-E 3 are 1024x1024,
BigGAN is 128x128 -- while every real photo source is variable (~500x375 for ImageNet,
640x480 for COCO). Point a classifier at that as-is and it scores ~100% off aspect
ratio and pixel count alone, having learned nothing whatsoever about generators.

model_02 is the more exposed of the two models here, because `data_io.canonical_transform`
resizes everything to a square `canonical_size`. A 1024->256 downscale and a 128->256
upscale leave completely different high-frequency signatures, and `features/fft.py`
reads exactly that band. The FFT block would end up describing *our own preprocessing*
rather than the generator -- the same failure the top-level README already flags for
upsampled CIFAKE.

So every image written by this script is a **center crop taken at native resolution**:
no scaling, no resampling, no interpolation of any kind. A 1024x1024 MidJourney render
and a 500x375 ImageNet photo both contribute a genuine 256x256 window of their own
pixels, and the two are then indistinguishable on every trivial cue. Images whose
short side is below the crop size are skipped rather than upscaled (upscaling is the
exact artifact we are trying not to manufacture); the count is reported per source.

Crop offsets are snapped to a multiple of 8 so each crop inherits its source's JPEG
8x8 grid phase. `features/fft.py` spends 64 of its 130 columns on a block-DCT profile
that assumes the grid begins at pixel 0, and an off-phase crop smears every block
boundary across all 64 of them.

The compression trap this script also avoids
--------------------------------------------
The second label-correlated cue in this data is compression history. GenImage's
MidJourney split and the DALL-E 3 dataset both ship **losslessly as PNG** inside their
parquet shards, while every real-photo corpus ships as JPEG. Leave that alone and "has
this image ever been JPEG-compressed?" becomes a near-perfect stand-in for the label --
and it is precisely what those 64 block-DCT columns measure. The classifier would score
brilliantly, generalize to nothing, and fall over the instant an image was re-encoded.

So by default every crop, real and fake alike, is written through one identical JPEG
encode (`--recompress jpeg`, quality 95, no chroma subsampling). The already-JPEG reals
end up double-compressed against single-compressed fakes, which is a far weaker residue
than present-vs-absent and is the same compromise the standard benchmarks in this area
make. `--recompress none` writes lossless PNG instead, which is only safe where both
classes share a source format (ForenSynths is PNG on both sides). Either way the run
prints the per-class source-format census, so this asymmetry can never silently return
when a new source is added.

One crop per source image, never several tiles from the same picture: `train.py` splits
train/val by `group_id`, which is assigned per file, so sibling tiles would land on
opposite sides of the split and inflate validation through near-duplicate leakage.

Usage
-----
    # everything, ~5k real + ~5k fake per family
    python prepare_generators.py --families all --per-class 5000

    # one family, smaller, to check the plumbing
    python prepare_generators.py --families midjourney --per-class 200

Parquet shards are streamed one at a time and deleted after use unless
`--keep-shards` is passed, so peak disk stays near one shard (~600MB) rather than
the full multi-GB download. Re-running is safe: existing outputs are counted and
the script tops up to `--per-class` rather than starting over.

Every written file is recorded in `<out>/<family>/manifest.csv` with its source repo,
shard, row index, original dimensions and original format, so any result can be traced
back to the exact bytes it came from.
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image

# Pillow refuses very large files by default as a decompression-bomb guard. The
# generator sources are legitimately 1024x1024+, which is nowhere near dangerous.
Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "raw"

# --------------------------------------------------------------------------- #
# Source specifications
#
# `real` and `fake` each name a Hugging Face dataset repo plus the column holding
# the image bytes. Pairings are deliberate, not arbitrary:
#
#  - MidJourney is paired with Open Images, NOT with ImageNet, despite ImageNet
#    being GenImage's own real counterpart and the better content match. Measured
#    on a 40-image sample, the ImageNet pairing is separable at AUC 0.976 on
#    residual standard deviation alone -- a single trivial scalar. The cause is
#    scale, not content: a 256px window covers 1/16 of a 1024x1024 render but
#    most of a 500x375 photo, so the crops differ in magnification and therefore
#    in texture density. Open Images is ~1024px native, which matches MidJourney's
#    scale and drops that same AUC to 0.734. A content mismatch the semantic
#    backbones can see through beats a scale cue that any classifier wins on.
#  - DALL-E 3 is paired with Open Images rather than COCO on purpose. The
#    challenge's held-out demo set is COCO val2017, and `bitmind/MS-COCO` is
#    COCO-2014 under Karpathy splits, which overlaps val2017 by image id. Using a
#    disjoint real corpus keeps the demo set genuinely held out instead of
#    relying on an id filter to stay correct.
#  - ProGAN needs no pairing decision: ForenSynths ships matched `0_real`/`1_fake`
#    folders drawn from the same LSUN scenes. It is measurably the cleanest of the
#    three (AUC 0.51-0.56 on those same trivial scalars) and the reason it was
#    chosen over GenImage's 128px BigGAN.
#
# `shard_offset` keeps the two families that share a real corpus reading from
# disjoint parts of it, so combining them in one training set doesn't silently
# duplicate the same real photographs across both.
# --------------------------------------------------------------------------- #
FAMILIES: dict[str, dict] = {
    "midjourney": {
        "out_name": "gen_midjourney",
        "kind": "parquet_pair",
        "fake": {"repo": "bitmind/GenImage_MidJourney", "column": "image"},
        "real": {"repo": "bitmind/open-images-v7-subset", "column": "image",
                 "shard_offset": 0},
        "note": "MidJourney (GenImage) 1024x1024 vs native-resolution Open Images v7",
    },
    "dalle3": {
        "out_name": "gen_dalle3",
        "kind": "parquet_pair",
        "fake": {"repo": "OpenDatasets/dalle-3-dataset", "column": "image"},
        "real": {"repo": "bitmind/open-images-v7-subset", "column": "image",
                 "shard_offset": 100},
        "note": "DALL-E 3 vs native-resolution Open Images v7 (disjoint shards)",
    },
    "progan": {
        "out_name": "gen_progan",
        "kind": "forensynths_zip",
        "repo": "sywang/CNNDetection",
        "archive": "progan_val.zip",
        "note": "ProGAN vs LSUN, matched pairs from ForenSynths (256x256 native)",
    },
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #
def center_crop_native(img: Image.Image, size: int) -> Image.Image | None:
    """Center `size`x`size` crop taken at the image's own resolution.

    Returns None if the image is too small, which the caller records as a skip.
    Deliberately never falls back to resizing: an upscale would manufacture the
    very high-frequency artifact `features/fft.py` is built to detect in the
    generator, and a downscale would hand the classifier a resolution cue that
    correlates perfectly with the label.

    Crop offsets are snapped down to a multiple of 8 so the crop keeps the JPEG
    8x8 grid phase of its source. 64 of `features/fft.py`'s 130 columns are a
    block-DCT profile that assumes the grid starts at pixel 0; an off-phase crop
    smears the block boundaries across all 64 and turns a real, measurable
    signal into noise. It also silently corrupts any attempt to measure
    compression symmetry between the two classes, since crop parity would differ
    per source (a 1024px fake lands on 384, an odd-sized real on 59).
    """
    w, h = img.size
    if min(w, h) < size:
        return None
    left = ((w - size) // 2) & ~7
    top = ((h - size) // 2) & ~7
    return img.crop((left, top, left + size, top + size))


class Writer:
    """Writes cropped images into `<root>/<label>/` and records provenance.

    Counts existing files on construction so a re-run tops up toward the target
    instead of redoing work already on disk.

    `recompress` controls the single most dangerous confound in this dataset.
    The generated sources ship losslessly (GenImage MidJourney and DALL-E 3 are
    both stored as PNG in their parquet shards) while every real-photo corpus
    ships as JPEG. Left alone, that makes "was this ever JPEG-compressed?" an
    almost perfect proxy for the label -- and `features/fft.py` spends 64 of its
    130 columns on exactly that question. A classifier would score near-perfectly
    while learning nothing about generators, and would collapse the moment
    anything re-encoded the images.

    With `recompress="jpeg"` (the default) every crop, real and fake alike, is
    written through one identical JPEG encode at `jpeg_quality`. That leaves the
    already-JPEG reals double-compressed against single-compressed fakes, which
    is a far subtler residue than present-vs-absent, and it is the same trade the
    standard benchmarks in this area make. `recompress="none"` restores lossless
    PNG output for sources already known to be format-symmetric (ForenSynths is
    PNG on both sides), or for deliberately inspecting the raw asymmetry.
    """

    def __init__(self, root: Path, crop_size: int, recompress: str = "jpeg",
                 jpeg_quality: int = 95):
        self.root = root
        self.crop_size = crop_size
        self.recompress = recompress
        self.jpeg_quality = jpeg_quality
        self.suffix = ".jpg" if recompress == "jpeg" else ".png"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = root / "manifest.csv"
        self._new_manifest = not self.manifest_path.exists()
        self.manifest = self.manifest_path.open("a", newline="")
        self.csv = csv.writer(self.manifest)
        if self._new_manifest:
            self.csv.writerow(
                ["dest", "label", "source_repo", "source_file", "row_index",
                 "orig_width", "orig_height", "orig_format", "crop_size",
                 "stored_format", "jpeg_quality"]
            )
        self.counts = {"real": self._existing("real"), "fake": self._existing("fake")}
        self.skipped_small = {"real": 0, "fake": 0}
        self.skipped_error = {"real": 0, "fake": 0}
        # Tracked per class so the run can report whether the source formats were
        # asymmetric -- the thing that must never silently return with a new source.
        self.src_formats: dict[str, dict[str, int]] = {"real": {}, "fake": {}}

    def _existing(self, label: str) -> int:
        d = self.root / label
        return sum(1 for p in d.glob(f"*{self.suffix}")) if d.is_dir() else 0

    def write(self, img_bytes: bytes, label: str, repo: str, source_file: str, row_index: int) -> bool:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            orig_format = img.format or "?"
            img = img.convert("RGB")
        except Exception:
            self.skipped_error[label] += 1
            return False

        orig_w, orig_h = img.size
        crop = center_crop_native(img, self.crop_size)
        if crop is None:
            self.skipped_small[label] += 1
            return False

        dest_dir = self.root / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Name encodes the source so a file is traceable even without the manifest.
        stem = f"{repo.split('/')[-1]}_{Path(source_file).stem}_{row_index:06d}"
        dest = dest_dir / f"{stem}{self.suffix}"
        if self.recompress == "jpeg":
            crop.save(dest, format="JPEG", quality=self.jpeg_quality, subsampling=0)
            stored, q = "JPEG", self.jpeg_quality
        else:
            crop.save(dest, format="PNG", optimize=False)
            stored, q = "PNG", ""

        self.src_formats[label][orig_format] = self.src_formats[label].get(orig_format, 0) + 1
        self.csv.writerow([str(dest.relative_to(self.root)), label, repo, source_file,
                           row_index, orig_w, orig_h, orig_format, self.crop_size,
                           stored, q])
        self.counts[label] += 1
        return True

    def format_warning(self) -> str | None:
        """Non-None if the two classes came from different source formats."""
        r, f = set(self.src_formats["real"]), set(self.src_formats["fake"])
        if r and f and r != f:
            note = (f"source formats differ: real={sorted(r)} fake={sorted(f)}")
            if self.recompress == "jpeg":
                return note + " -- neutralized by uniform JPEG re-encode"
            return note + " -- NOT neutralized (--recompress none); this is a label-correlated cue"
        return None

    def close(self):
        self.manifest.close()


# --------------------------------------------------------------------------- #
# Parquet sources
# --------------------------------------------------------------------------- #
def list_parquet_shards(repo: str) -> list[str]:
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    if not files:
        raise RuntimeError(f"No parquet shards found in {repo}")
    return sorted(files)


def fill_from_parquet(
    writer: Writer,
    label: str,
    repo: str,
    column: str,
    target: int,
    keep_shards: bool,
    shard_offset: int = 0,
) -> None:
    """Stream shards from `repo` until `writer` holds `target` images for `label`.

    One shard is on disk at a time (deleted after use unless --keep-shards), so a
    200GB source costs ~600MB of working space rather than 200GB.

    `shard_offset` skips the first N shards, so two families drawing reals from
    the same corpus get disjoint images instead of the same ones twice.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if writer.counts[label] >= target:
        print(f"    {label}: already have {writer.counts[label]}/{target}, skipping")
        return

    shards = list_parquet_shards(repo)
    if shard_offset:
        if shard_offset >= len(shards):
            raise RuntimeError(
                f"shard_offset {shard_offset} exceeds {repo}'s {len(shards)} shards"
            )
        shards = shards[shard_offset:]
    print(f"    {label}: {repo} ({len(shards)} shards"
          f"{f', offset {shard_offset}' if shard_offset else ''}) -> need "
          f"{target - writer.counts[label]} more")

    for shard in shards:
        if writer.counts[label] >= target:
            break
        try:
            local = hf_hub_download(repo, shard, repo_type="dataset")
        except Exception as exc:
            print(f"      ! failed to fetch {shard}: {exc}")
            continue

        try:
            pf = pq.ParquetFile(local)
            if column not in pf.schema_arrow.names:
                raise RuntimeError(
                    f"column '{column}' not in {repo} schema {pf.schema_arrow.names}"
                )
            row_index = 0
            for batch in pf.iter_batches(batch_size=64, columns=[column]):
                for value in batch.column(0).to_pylist():
                    if writer.counts[label] >= target:
                        break
                    # HF Image features arrive as {"bytes": ..., "path": ...};
                    # some repos store raw bytes directly.
                    data = value.get("bytes") if isinstance(value, dict) else value
                    if data:
                        writer.write(data, label, repo, shard, row_index)
                    row_index += 1
                if writer.counts[label] >= target:
                    break
            print(f"      {shard}: {label} now {writer.counts[label]}/{target}")
        finally:
            if not keep_shards:
                # Remove the blob the symlink points at, not just the link.
                try:
                    real = Path(local).resolve()
                    Path(local).unlink(missing_ok=True)
                    real.unlink(missing_ok=True)
                except OSError:
                    pass

    if writer.counts[label] < target:
        print(f"      ! exhausted {repo}: {writer.counts[label]}/{target} for {label}")


# --------------------------------------------------------------------------- #
# ForenSynths source (matched real/fake pairs in one zip)
# --------------------------------------------------------------------------- #
def fill_from_forensynths(
    writer: Writer, repo: str, archive: str, target: int, keep_shards: bool
) -> None:
    """ForenSynths lays out `<class>/0_real/*.png` and `<class>/1_fake/*.png`.

    Both labels come out of one archive, drawn from the same LSUN scenes, so the
    real/fake pairing is matched by construction rather than by our choice.
    """
    from huggingface_hub import hf_hub_download

    if writer.counts["real"] >= target and writer.counts["fake"] >= target:
        print(f"    already have {writer.counts}, skipping")
        return

    print(f"    fetching {repo}:{archive} (~0.8GB)")
    local = hf_hub_download(repo, archive, repo_type="dataset")

    try:
        with zipfile.ZipFile(local) as zf:
            members = [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_SUFFIXES]
            buckets: dict[str, list[str]] = {"real": [], "fake": []}
            for name in members:
                parts = Path(name).parts
                if "0_real" in parts:
                    buckets["real"].append(name)
                elif "1_fake" in parts:
                    buckets["fake"].append(name)
            if not buckets["real"] or not buckets["fake"]:
                raise RuntimeError(
                    f"{archive} has no 0_real/1_fake folders -- layout changed; "
                    f"inspect with `unzip -l` and adjust fill_from_forensynths()"
                )

            for label in ("real", "fake"):
                # Interleave classes so the sample spans all LSUN categories
                # instead of exhausting the alphabetically-first one.
                names = sorted(buckets[label])
                by_class: dict[str, list[str]] = {}
                for n in names:
                    by_class.setdefault(Path(n).parts[0], []).append(n)
                ordered: list[str] = []
                streams = list(by_class.values())
                for i in range(max(len(s) for s in streams)):
                    for s in streams:
                        if i < len(s):
                            ordered.append(s[i])

                for row_index, name in enumerate(ordered):
                    if writer.counts[label] >= target:
                        break
                    writer.write(zf.read(name), label, repo, archive, row_index)
                print(f"      {label}: {writer.counts[label]}/{target} "
                      f"(pool {len(ordered)} across {len(by_class)} classes)")
    finally:
        if not keep_shards:
            try:
                real = Path(local).resolve()
                Path(local).unlink(missing_ok=True)
                real.unlink(missing_ok=True)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
def prepare_family(name: str, out_root: Path, per_class: int, crop_size: int,
                   keep_shards: bool, recompress: str, jpeg_quality: int) -> dict:
    spec = FAMILIES[name]
    root = out_root / spec["out_name"]
    print(f"\n{'=' * 72}\n{name}  ->  {root}\n  {spec['note']}\n{'=' * 72}")

    writer = Writer(root, crop_size, recompress=recompress, jpeg_quality=jpeg_quality)
    try:
        if spec["kind"] == "parquet_pair":
            fill_from_parquet(writer, "fake", spec["fake"]["repo"],
                              spec["fake"]["column"], per_class, keep_shards,
                              spec["fake"].get("shard_offset", 0))
            fill_from_parquet(writer, "real", spec["real"]["repo"],
                              spec["real"]["column"], per_class, keep_shards,
                              spec["real"].get("shard_offset", 0))
        elif spec["kind"] == "forensynths_zip":
            fill_from_forensynths(writer, spec["repo"], spec["archive"],
                                  per_class, keep_shards)
        else:
            raise RuntimeError(f"unknown kind {spec['kind']}")
    finally:
        writer.close()

    warning = writer.format_warning()
    summary = {
        "family": name,
        "root": str(root),
        "real": writer.counts["real"],
        "fake": writer.counts["fake"],
        "skipped_small": dict(writer.skipped_small),
        "skipped_error": dict(writer.skipped_error),
        "src_formats": {k: dict(v) for k, v in writer.src_formats.items()},
        "warning": warning,
    }
    print(f"  -> real={summary['real']} fake={summary['fake']} "
          f"skipped_small={summary['skipped_small']} "
          f"skipped_error={summary['skipped_error']}")
    print(f"     source formats: real={summary['src_formats']['real']} "
          f"fake={summary['src_formats']['fake']}")
    if warning:
        print(f"     [format] {warning}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--families", nargs="+", default=["all"],
                        choices=list(FAMILIES) + ["all"],
                        help="Which generator families to prepare.")
    parser.add_argument("--per-class", type=int, default=5000,
                        help="Target images per class per family (default 5000, "
                             "so ~10k per family).")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Native-resolution center crop size (default 256, "
                             "matching model_02's canonical_size).")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Dataset root (default {DEFAULT_OUT}).")
    parser.add_argument("--keep-shards", action="store_true",
                        help="Keep downloaded parquet/zip blobs in the HF cache. "
                             "Off by default so peak disk stays near one shard.")
    parser.add_argument("--recompress", choices=["jpeg", "none"], default="jpeg",
                        help="Write every crop through one identical JPEG encode "
                             "(default) so compression history cannot correlate "
                             "with the label. 'none' writes lossless PNG and is "
                             "only safe when both classes share a source format.")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="Quality for --recompress jpeg (default 95).")
    args = parser.parse_args()

    families = list(FAMILIES) if "all" in args.families else args.families
    out_root = Path(args.out)

    summaries = [
        prepare_family(f, out_root, args.per_class, args.crop_size, args.keep_shards,
                       args.recompress, args.jpeg_quality)
        for f in families
    ]

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for s in summaries:
        print(f"  {s['family']:<12} real={s['real']:<7} fake={s['fake']:<7} {s['root']}")
    print("\nAdd these to a config's train_datasets by their folder names, e.g.:")
    print("  train_datasets: [" + ", ".join(f'"{FAMILIES[f]["out_name"]}"' for f in families) + "]")

    warned = [s for s in summaries if s["warning"]]
    if warned:
        print("\nSource-format notes:")
        for s in warned:
            print(f"  {s['family']:<12} {s['warning']}")

    incomplete = [s for s in summaries if s["real"] < args.per_class or s["fake"] < args.per_class]
    if incomplete:
        print(f"\nNote: {len(incomplete)} family/families came up short of "
              f"--per-class {args.per_class}; see the per-source lines above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
