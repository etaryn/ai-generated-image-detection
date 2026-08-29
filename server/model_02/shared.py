"""Bridge to the pieces of `model_01` that both models must share.

model_02 is a different *architecture*, not a different *problem*: it has to be
trained on the same folder layout, augmented with the same transform family, and
scored with the same metric definitions as model_01 -- otherwise the two models'
numbers aren't comparable and the robustness table means nothing.

Rather than copy those files (which would inevitably drift), this module loads
them straight out of `../model_01/` by file path. File-path loading is used
instead of `sys.path.insert(model_01)` on purpose: both models have a top-level
`eval/` package, and putting model_01 on sys.path would let model_01's `eval`
shadow model_02's (or vice versa) depending on import order. Loading by path
gives the modules private names (`model_01.data.transforms`) that can't collide.

Re-exported here:
    RealFakeImageDataset, ImageFolderInference   (data/datasets.py)
    RobustnessAugment, SEVERITY_LEVELS,
    apply_named_transform                        (data/transforms.py)
    compute_all_metrics, fpr_at_threshold,
    threshold_for_target_fpr                     (eval/metrics.py)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

MODEL_01_ROOT = Path(__file__).resolve().parent.parent / "model_01"


def load_model_01_module(relpath: str, alias: str) -> ModuleType:
    """Import `model_01/<relpath>` under the private name `alias`.

    Raises a pointed error if model_01 isn't next to model_02 -- that's a layout
    problem the user needs to fix, not something to paper over with a fallback.
    """
    path = MODEL_01_ROOT / relpath
    if not path.exists():
        raise ImportError(
            f"model_02 shares data/augmentation/metric code with model_01, but "
            f"{path} does not exist. Expected layout: server/model_01/ and "
            f"server/model_02/ side by side."
        )
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_datasets = load_model_01_module("data/datasets.py", "model_01.data.datasets")
_transforms = load_model_01_module("data/transforms.py", "model_01.data.transforms")
_metrics = load_model_01_module("eval/metrics.py", "model_01.eval.metrics")

RealFakeImageDataset = _datasets.RealFakeImageDataset
ImageFolderInference = _datasets.ImageFolderInference
IMAGE_EXTENSIONS = _datasets.IMAGE_EXTENSIONS

RobustnessAugment = _transforms.RobustnessAugment
SEVERITY_LEVELS = _transforms.SEVERITY_LEVELS
apply_named_transform = _transforms.apply_named_transform

compute_all_metrics = _metrics.compute_all_metrics
fpr_at_threshold = _metrics.fpr_at_threshold
threshold_for_target_fpr = _metrics.threshold_for_target_fpr

__all__ = [
    "MODEL_01_ROOT",
    "load_model_01_module",
    "RealFakeImageDataset",
    "ImageFolderInference",
    "IMAGE_EXTENSIONS",
    "RobustnessAugment",
    "SEVERITY_LEVELS",
    "apply_named_transform",
    "compute_all_metrics",
    "fpr_at_threshold",
    "threshold_for_target_fpr",
]
