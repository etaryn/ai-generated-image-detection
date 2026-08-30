"""Patch scorers: the one learned component Layer 1 stands on.

model_03 trains no detector of its own. It takes an existing image-level
detector, asks it the same question many times -- once per patch -- and stitches
the answers into a map. Everything downstream (regions, routing, fusion) is
built on whatever this returns, so the backend is deliberately the first thing
the config names.

The contract is one method:

    score_patches(images: list[PIL.Image]) -> list[float]     # P(AI) per patch

Three implementations:

* `HFImageClassifierBackend` -- **the default.** Any public AI-image detector on
  the Hugging Face Hub; see `PUBLIC_MODELS` for the surveyed shortlist. These
  are trained at 224px on modern generator output, which is what a patch scorer
  needs to be.
* `Model01Backend` / `Model02Backend` -- this repo's own detectors, kept so the
  three models stay comparable on one harness. model_01's shipped weights are
  CIFAKE at **32x32**, so every patch is downsampled to 32px before scoring,
  destroying the fine blending seams this pipeline exists to find. Useful as a
  baseline; a poor instrument for localisation.

Scoring is batched because a 1024px image at three scales with 50% overlap is
about a thousand patches, and one-at-a-time scoring would make the demo
unusable. `Model01Backend` therefore reimplements model_01's preprocessing
rather than calling its single-image helper in a loop -- the same transform,
read from the same checkpoint.

A note on what a patch score means, which applies to *every* backend here. All
of them were trained on whole images, so scoring a 128px crop asks a slightly
off-distribution question: "does this fragment look generated?" rather than
"does this image look generated?". That is a real limitation of the MVP, not a
detail -- it is why the map is calibrated (see calibration.py) rather than
trusted raw, and why fusion refuses to turn a patch-level signal into a
whole-image verdict on its own. The honest fix is a patch-level training run;
the mapper is written so that dropping one in means adding a backend here and
changing nothing else.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Protocol, Sequence

SERVER_DIR = Path(__file__).resolve().parents[2]


class PatchScorer(Protocol):
    """Anything that turns a list of PIL patches into P(AI-generated)."""

    name: str

    def score_patches(self, images: Sequence) -> list[float]:
        ...


def _load_sibling_infer(model_key: str):
    """Import `server/<model_key>/infer.py` under its own module name.

    Both siblings ship a top-level `infer.py`. A plain `import infer` would bind
    whichever was imported first and then hand that same module back for the
    other one -- scoring model_02 patches with model_01's weights, silently.
    Loading each from its own file path under its own module name keeps them
    separate in sys.modules. (client/app.py does the same thing for the same
    reason; the duplication is deliberate -- neither project should have to
    import the other to be runnable on its own.)
    """
    model_dir = SERVER_DIR / model_key
    infer_path = model_dir / "infer.py"
    if not infer_path.exists():
        raise FileNotFoundError(
            f"No sibling detector at {infer_path}. model_03 scores patches with "
            f"model_01 or model_02; one of them has to be present."
        )

    # The sibling's own modules (`model.detector`, `features.pipeline`, ...) are
    # imported relative to its directory, so it has to be on sys.path.
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    mod_name = f"_model03_infer_{model_key}"
    module = sys.modules.get(mod_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(mod_name, str(infer_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            # Don't leave a half-initialised module behind for the next attempt.
            sys.modules.pop(mod_name, None)
            raise
    return module


class Model01Backend:
    """model_01 (CNN + Transformer) as a patch scorer.

    Preprocessing is read from the checkpoint, never assumed: model_01's
    positional embedding is a fixed-size parameter, so the input size is a hard
    shape constraint rather than a tunable. The shipped CIFAKE weights are 32px,
    which means every patch -- 128px or 224px -- is downsampled to 32px before
    scoring. That is lossy for exactly the fine blending seams this pipeline
    cares about; see the README's "Known weaknesses".
    """

    name = "model_01"

    def __init__(self, checkpoint: str | Path | None = None, batch_size: int = 64):
        import torch  # lazy: keeps this module importable in the torch-free tests

        self._torch = torch
        self._infer = _load_sibling_infer("model_01")
        self.batch_size = int(batch_size)

        self._model, self._cfg, self._device = self._infer.load_model(checkpoint)
        self.image_size = self._infer.checkpoint_image_size(self._cfg)
        self._transform = self._infer.build_inference_transform(self.image_size)
        self._use_freq = bool(self._cfg["model"]["use_freq_branch"])

    def score_patches(self, images: Sequence) -> list[float]:
        torch = self._torch
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                chunk = images[start : start + self.batch_size]
                batch = torch.stack([self._transform(im.convert("RGB")) for im in chunk])
                batch = batch.to(self._device)
                raw = batch if self._use_freq else None
                probs = self._model.predict_proba(batch, raw).detach().cpu().tolist()
                out.extend(float(p) for p in probs)
        return out


class Model02Backend:
    """model_02 (frozen DINOv2 + CLIP + FFT -> small classifier) as a patch scorer.

    Slower per patch than model_01 (two ViT forward passes plus the spectral
    block), but the backend to prefer when the map matters more than the frame
    rate: its FFT block reads the noise floor directly, which is the evidence a
    *local* edit actually leaves behind.
    """

    name = "model_02"

    def __init__(self, checkpoint: str | Path | None = None, batch_size: int = 32):
        self._infer = _load_sibling_infer("model_02")
        self.batch_size = int(batch_size)
        self._checkpoint = checkpoint
        bundle, _, _, _ = self._infer.load_model(checkpoint)
        self.image_size = int(bundle["canonical_size"])

    def score_patches(self, images: Sequence) -> list[float]:
        out: list[float] = []
        for start in range(0, len(images), self.batch_size):
            chunk = list(images[start : start + self.batch_size])
            out.extend(self._infer.predict_images(chunk, self._checkpoint))
        return out


class CallableBackend:
    """Wrap any `fn(list[PIL]) -> list[float]` as a scorer.

    Exists so the geometry, region and fusion stages can be tested against a
    scorer with known behaviour, with no checkpoint and no torch. The tests use
    it; so can a notebook holding a third-party detector.
    """

    def __init__(self, fn: Callable[[Sequence], Sequence[float]], name: str = "callable"):
        self._fn = fn
        self.name = name

    def score_patches(self, images: Sequence) -> list[float]:
        scores = list(self._fn(images))
        if len(scores) != len(images):
            raise ValueError(
                f"scorer returned {len(scores)} scores for {len(images)} patches"
            )
        return [float(s) for s in scores]


#: Public AI-image detectors on the Hugging Face Hub that work as patch scorers.
#: `labels` records what each model's config said when it was surveyed -- it is
#: documentation, not a contract: resolution always reads the config that was
#: actually downloaded (see mapper/labels.py). Note dima806's flipped ordering.
PUBLIC_MODELS = {
    "Organika/sdxl-detector": {
        "arch": "swin (87M)",
        "labels": {0: "artificial", 1: "human"},
        "note": "The default. Fine-tuned from umm-maybe on SDXL output, so it "
                "knows a more current generator family than its parent.",
    },
    "umm-maybe/AI-image-detector": {
        "arch": "swin (87M)",
        "labels": {0: "artificial", 1: "human"},
        "note": "The most-downloaded of the family and the oldest; trained on "
                "2022-era generators, so expect it to miss modern diffusion output.",
    },
    "haywoodsloan/ai-image-detector-deploy": {
        "arch": "swinv2 (87M)",
        "labels": {0: "artificial", 1: "real"},
        "note": "SwinV2; a reasonable second opinion to disagree with the default.",
    },
    "Ateeqq/ai-vs-human-image-detector": {
        "arch": "siglip (93M)",
        "labels": {0: "ai", 1: "hum"},
        "note": "SigLIP backbone -- a different feature family from the Swin models, "
                "so its errors are less correlated with theirs. Saturates on very "
                "small images (AUC 0.454 on 32px CIFAKE, both medians ~0.997); check "
                "it on your own data with scripts/check_backend.py first.",
    },
    "prithivMLmods/Deep-Fake-Detector-Model": {
        "arch": "siglip (93M)",
        "labels": {0: "Fake", 1: "Real"},
        "note": "Face/deepfake-oriented. Pair it with the face-edit route rather "
                "than using it as a general detector.",
    },
    "dima806/ai_vs_real_image_detection": {
        "arch": "vit (86M)",
        "labels": {0: "REAL", 1: "FAKE"},
        "note": "Labels are REVERSED relative to the others. Handled automatically "
                "by name resolution; a hard-coded index would invert it silently. "
                "Scores a perfect 1.000 AUC on CIFAKE, which most likely means "
                "CIFAKE was in its training set -- treat that number as leakage, "
                "not as evidence it generalises.",
    },
}

DEFAULT_HF_MODEL = "Organika/sdxl-detector"


def _pil_bicubic_weights(in_size: int, out_size: int):
    """PIL's exact bicubic resampling weights for one axis.

    `torch.nn.functional.interpolate(mode="bicubic")` is *not* the same filter
    as PIL's. Torch uses the Keys cubic convolution with a = -0.75; PIL uses
    a = -0.5, and derives its kernel support from the scale factor. On
    downsampling the difference is small, but upsampling a 64px crop to a 224px
    model input it is not: measured on structured content, patch scores diverged
    by up to 0.27, which is more than the width of the "likely AI" decision
    band. Approximating here would have shifted the finest scale -- the one
    carrying most of the localisation signal -- and nothing would have failed.

    So this reproduces PIL's `precompute_coeffs` directly. Because every window
    at a given scale is the same size, the weights depend only on
    (in_size, out_size) and can be built once and reused as a matrix, making the
    resize two batched matmuls.

    Returns a dense (out_size, in_size) float64 array.
    """
    import numpy as np

    scale = in_size / out_size
    filterscale = max(1.0, scale)
    support = 2.0 * filterscale  # bicubic support is 2.0

    def cubic(x: float) -> float:
        # PIL's bicubic filter, a = -0.5.
        a = -0.5
        x = abs(x)
        if x < 1.0:
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        if x < 2.0:
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a
        return 0.0

    weights = np.zeros((out_size, in_size), dtype=np.float64)
    inv = 1.0 / filterscale
    for out_index in range(out_size):
        center = (out_index + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        total = 0.0
        for x in range(xmin, xmax):
            w = cubic((x - center + 0.5) * inv)
            weights[out_index, x] = w
            total += w
        if total != 0.0:
            weights[out_index, xmin:xmax] /= total
    return weights


class HFImageClassifierBackend:
    """Any Hugging Face image-classification model as a patch scorer.

    This is model_03's default backend, and the reason is scope: model_01 and
    model_02 are this repo's own detectors, trained on CIFAKE at 32x32, so every
    patch handed to them is downsampled to 32px before scoring -- destroying the
    fine blending seams the whole pipeline is built to find. A public detector
    trained at 224px on modern generator output is a far better instrument for
    the same job, and swapping it in costs nothing but a config line, because
    the mapper only ever needed `score_patches`.

    What it does *not* fix: these models were also trained on whole images, so
    scoring a crop is still slightly off-distribution (see this module's header),
    and each carries its own generator-family bias. That is what the calibration
    step and the `PUBLIC_MODELS` registry's second opinions are for.

    Weights come from the Hub on first use and are cached by `huggingface_hub`
    thereafter; set $HF_HOME to move that cache, or pre-download on a machine
    with network access and run offline with $HF_HUB_OFFLINE=1.
    """

    def __init__(
        self,
        model_id: str | None = None,
        positive_label: str | None = None,
        positive_index: int | None = None,
        batch_size: int = 64,
        device: str | None = None,
        fp16: bool | None = None,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        from mapper.labels import resolve_positive_indices

        self._torch = torch
        self.model_id = model_id or DEFAULT_HF_MODEL
        self.name = f"hf:{self.model_id}"
        self.batch_size = int(batch_size)

        try:
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_id, trust_remote_code=trust_remote_code
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                self.model_id, trust_remote_code=trust_remote_code
            )
        except OSError as exc:
            raise OSError(
                f"Could not load {self.model_id!r} from the Hugging Face Hub or the local "
                f"cache ({exc}). With network access it downloads on first use; without, "
                f"pre-download it elsewhere and point $HF_HOME at the cache, or set "
                f"backend.name to model_01 to use this repo's own weights."
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 roughly doubles throughput on a GPU and is meaningless on CPU.
        self.fp16 = (self.device.startswith("cuda")) if fp16 is None else bool(fp16)
        self._model = self._model.to(self.device)
        if self.fp16:
            self._model = self._model.half()
        self._model.eval()

        # Which output means "AI-generated" -- by name, never by index. See
        # mapper/labels.py for why this is the module's most dangerous question.
        self.positive_indices, self.label_reason = resolve_positive_indices(
            self._model.config.id2label, positive_label, positive_index
        )
        self.id2label = dict(self._model.config.id2label)
        # Resampling matrices, keyed by (source size, target size). One entry
        # per window scale, built on first use.
        self._resize_cache: dict[tuple[int, int], object] = {}

    def describe(self) -> dict:
        """What this backend is, for the report's provenance line."""
        return {
            "backend": self.name,
            "device": self.device,
            "fp16": self.fp16,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "positive_indices": list(self.positive_indices),
            "positive_resolved_by": self.label_reason,
        }

    def score_patches(self, images: Sequence) -> list[float]:
        """Score a list of PIL patches through the model's own processor.

        The reference path: whatever the processor does is by definition
        correct. `score_crops` is the fast equivalent, and is checked against
        this one in tests/test_fast_preprocess.py.
        """
        torch = self._torch
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                chunk = [im.convert("RGB") for im in images[start : start + self.batch_size]]
                inputs = self._processor(images=chunk, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self.fp16:
                    inputs = {
                        k: (v.half() if v.dtype == torch.float32 else v)
                        for k, v in inputs.items()
                    }
                out.extend(self._forward(inputs["pixel_values"]))
        return out

    def _forward(self, pixel_values) -> list[float]:
        logits = self._model(pixel_values=pixel_values).logits.float()
        probs = logits.softmax(dim=-1)
        # Sum over every AI-side class, so a multi-class model ("real", "gan",
        # "diffusion") still yields one P(AI).
        ai = probs[:, self.positive_indices].sum(dim=-1)
        return [float(v) for v in ai.detach().cpu().tolist()]

    def _preprocess_config(self) -> dict | None:
        """Read the processor's settings, or None if it does something unusual.

        Only the plain resize -> rescale -> normalize pipeline is reproduced on
        the GPU. Anything else (centre crops, padding, per-image rescaling)
        falls back to the processor rather than being approximated, because a
        preprocessing mismatch does not fail loudly -- it just quietly shifts
        every score.
        """
        p = self._processor
        size = getattr(p, "size", None)
        height = getattr(size, "height", None) or (size or {}).get("height")
        width = getattr(size, "width", None) or (size or {}).get("width")
        if not height or not width:
            return None
        if getattr(p, "do_center_crop", None) or getattr(p, "crop_size", None):
            return None
        if not getattr(p, "do_resize", True) or not getattr(p, "do_normalize", True):
            return None
        if int(getattr(p, "resample", 3)) != 3:  # 3 == PIL.Image.BICUBIC
            return None
        return {
            "height": int(height),
            "width": int(width),
            "rescale": float(getattr(p, "rescale_factor", 1 / 255)),
            "mean": tuple(getattr(p, "image_mean", (0.485, 0.456, 0.406))),
            "std": tuple(getattr(p, "image_std", (0.229, 0.224, 0.225))),
        }

    def _resize_weights(self, in_size: int, out_size: int):
        """Cached PIL-equivalent resampling matrix, on device."""
        key = (int(in_size), int(out_size))
        cached = self._resize_cache.get(key)
        if cached is None:
            import numpy as np

            weights = _pil_bicubic_weights(int(in_size), int(out_size))
            cached = self._torch.from_numpy(np.ascontiguousarray(weights)).to(
                self.device, dtype=self._torch.float32
            )
            self._resize_cache[key] = cached
        return cached

    def score_crops(self, image, boxes: Sequence) -> list[float]:
        """Score many crops of one image, cropping and resizing on the GPU.

        Profiled on a 1024x768 image at three scales (932 windows), the
        processor path spent 4.72s resizing patches one at a time in Python
        against 4.58s of actual model forward -- half the wall clock doing work
        a batched `interpolate` does in milliseconds. PIL cropping was 0.06s, so
        the crops were never the problem; the resize was.

        This uploads the image once, slices every window out of it on device,
        and resizes each same-sized group as one batch. Windows of a given scale
        are all the same size, so the grouping is exactly the scale structure
        the mapper already has.

        `antialias=True` matters: PIL's bicubic downsampling filters first, and
        without it a 224px window downsampled to 224 (a no-op) and a 64px window
        upsampled would diverge from the reference in different directions.

        Falls back to `score_patches` whenever the processor does anything this
        cannot reproduce faithfully.
        """
        torch = self._torch
        cfg = self._preprocess_config()
        if cfg is None:
            return self.score_patches([image.crop(tuple(b)) for b in boxes])

        import numpy as np
        from collections import defaultdict

        rgb = np.asarray(image.convert("RGB"))
        with torch.no_grad():
            # np.array (not asarray): torch refuses to share a read-only buffer.
            frame = torch.from_numpy(np.array(rgb)).to(self.device).permute(2, 0, 1).float()

            mean = torch.tensor(cfg["mean"], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(cfg["std"], device=self.device).view(1, 3, 1, 1)

            groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            for i, box in enumerate(boxes):
                x0, y0, x1, y1 = box
                groups[(y1 - y0, x1 - x0)].append(i)

            out = [0.0] * len(boxes)
            for _, indices in groups.items():
                for start in range(0, len(indices), self.batch_size):
                    chunk = indices[start : start + self.batch_size]
                    stack = torch.stack(
                        [frame[:, boxes[i][1] : boxes[i][3], boxes[i][0] : boxes[i][2]] for i in chunk]
                    )
                    # Separable resize with PIL's own weights, as two batched
                    # matmuls. Horizontal first, then vertical, with a clamp and
                    # round in between -- that is not incidental: PIL writes an
                    # 8-bit image between its two passes, and bicubic overshoots
                    # on sharp transitions, so skipping the intermediate
                    # quantisation diverges by up to 24 levels on high-frequency
                    # content. With it, the two agree to within 2 levels of 255
                    # on pure noise and 1 on real images.
                    wx = self._resize_weights(stack.shape[3], cfg["width"])
                    wy = self._resize_weights(stack.shape[2], cfg["height"])
                    cols = torch.einsum("pj,ncij->ncip", wx, stack)
                    cols = cols.clamp(0, 255).round()
                    resized = torch.einsum("oi,ncip->ncop", wy, cols)

                    pixels = resized.clamp(0, 255).round() * cfg["rescale"]
                    pixels = (pixels - mean) / std
                    if self.fp16:
                        pixels = pixels.half()
                    for index, score in zip(chunk, self._forward(pixels)):
                        out[index] = score
        return out


class CallableBackend:
    """Wrap any `fn(list[PIL]) -> list[float]` as a scorer.

    Exists so the geometry, region and fusion stages can be tested against a
    scorer with known behaviour, with no checkpoint and no torch. The tests use
    it; so can a notebook holding a third-party detector.
    """

    def __init__(self, fn: Callable[[Sequence], Sequence[float]], name: str = "callable"):
        self._fn = fn
        self.name = name

    def score_patches(self, images: Sequence) -> list[float]:
        scores = list(self._fn(images))
        if len(scores) != len(images):
            raise ValueError(
                f"scorer returned {len(scores)} scores for {len(images)} patches"
            )
        return [float(s) for s in scores]


BACKENDS = {
    "hf": HFImageClassifierBackend,
    "model_01": Model01Backend,
    "model_02": Model02Backend,
}


def build_backend(spec: str | None = None, **kwargs) -> PatchScorer:
    """Build a patch scorer.

    `spec` accepts:
        'hf'                        the default public detector
        'hf:<hub model id>'         a specific public detector
        '<owner>/<model>'           the same, spelled as a bare Hub id
        'model_01' | 'model_02'     this repo's own detectors, kept for comparison

    Falls back to $AIGC_MODEL03_BACKEND, then to 'hf'. Only keyword arguments the
    chosen backend actually accepts are forwarded; a stray one raises rather than
    being silently dropped, because a silently-ignored `positive_label` is
    exactly the kind of mistake that inverts a map.
    """
    import inspect

    spec = spec or os.environ.get("AIGC_MODEL03_BACKEND") or "hf"

    if spec.startswith("hf:"):
        kwargs.setdefault("model_id", spec[len("hf:") :])
        spec = "hf"
    elif spec not in BACKENDS and Path(spec).is_dir():
        # A locally trained checkpoint directory (scripts/train_patch_scorer.py).
        # Checked before the "/" heuristic because a Windows path has no forward
        # slashes and would otherwise be rejected as an unknown backend.
        kwargs.setdefault("model_id", str(Path(spec)))
        spec = "hf"
    elif spec not in BACKENDS and "/" in spec:
        kwargs.setdefault("model_id", spec)
        spec = "hf"

    if spec not in BACKENDS:
        raise ValueError(
            f"unknown patch-scorer backend {spec!r}. Expected one of {sorted(BACKENDS)}, "
            f"'hf:<hub model id>', or a bare Hub id like 'Organika/sdxl-detector'."
        )

    cls = BACKENDS[spec]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    supplied = {k: v for k, v in kwargs.items() if v is not None}
    unknown = sorted(set(supplied) - accepted)
    if unknown:
        raise TypeError(
            f"backend {spec!r} does not accept {unknown}; it takes {sorted(accepted)}. "
            f"(Leaving a setting silently unapplied is how a detector ends up scoring "
            f"with the wrong class or the wrong checkpoint.)"
        )
    return cls(**supplied)
