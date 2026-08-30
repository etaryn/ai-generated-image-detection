"""Patch scorers: the one learned component Layer 1 stands on.

model_03 does not train a detector of its own. It borrows one of the sibling
projects' image-level detectors and asks it the same question many times, once
per patch, then stitches the answers back into a map. Everything downstream --
regions, routing, fusion -- is built on whatever this returns, so the backend is
deliberately the first thing the config names.

The contract is one method:

    score_patches(images: list[PIL.Image]) -> list[float]     # P(AI) per patch

Batched, because a single 1024x1024 image at two scales with 50% overlap is a
few hundred patches; scoring them one at a time through the siblings'
`predict_image` would make the demo unusable. `Model01Backend` therefore
reimplements model_01's preprocessing rather than calling its single-image
helper in a loop -- it is the same transform, read from the same checkpoint.

A note on what a patch score means. Both siblings were trained on whole images,
so scoring a 128px crop asks them something slightly off-distribution: "does
this fragment look generated?" rather than "does this image look generated?".
That is a real limitation of the MVP, not a detail -- it is why the map is
calibrated (see calibration.py) against patch scores rather than trusted raw,
and why fusion refuses to turn a patch-level signal into a whole-image verdict
on its own. The honest fix is a patch-level training run; the mapper is written
so that dropping one in means adding a backend here and nothing else.
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


BACKENDS = {"model_01": Model01Backend, "model_02": Model02Backend}


def build_backend(spec: str | None = None, **kwargs) -> PatchScorer:
    """Build the scorer named by `spec` ('model_01' | 'model_02').

    Falls back to $AIGC_MODEL03_BACKEND, then to model_01 -- the cheap one, and
    the one whose weights are in the repo.
    """
    spec = spec or os.environ.get("AIGC_MODEL03_BACKEND") or "model_01"
    if spec not in BACKENDS:
        raise ValueError(
            f"unknown patch-scorer backend {spec!r}; expected one of {sorted(BACKENDS)}"
        )
    return BACKENDS[spec](**kwargs)
