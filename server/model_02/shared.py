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
    RealDegSampler, RealDegStep, apply_realdeg_chain,
    REALDEG_OPERATORS, REALDEG_STRENGTHS,
    balanced_accuracy                            (eval/realdeg.py)
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

MODEL_01_ROOT = Path(__file__).resolve().parent.parent / "model_01"

# Alias -> file, relative to MODEL_01_ROOT.
_ALIAS_FILES = {
    "model_01.data.datasets": "data/datasets.py",
    "model_01.data.transforms": "data/transforms.py",
    "model_01.eval.metrics": "eval/metrics.py",
    "model_01.eval.realdeg": "eval/realdeg.py",
}
# Intermediate package names the aliases imply. They hold no code; they exist so
# `import model_01.data.transforms` can walk the chain.
_ALIAS_PACKAGES = ("model_01", "model_01.data", "model_01.eval")


class _AliasPackageLoader(importlib.abc.Loader):
    """Loader for the empty intermediate packages."""

    def create_module(self, spec):  # noqa: D102 - default semantics
        return None

    def exec_module(self, module):  # noqa: D102 - nothing to execute
        pass


class ModelO1AliasFinder(importlib.abc.MetaPathFinder):
    """Makes the private `model_01.*` aliases importable by name.

    Stuffing the modules directly into sys.modules (what this file used to do) is
    enough for the parent process, but pickle stores a class as module-name +
    qualname and re-imports that module on the other side. Under Python 3.13 and
    earlier that was invisible: DataLoader workers were forked, so they inherited
    the parent's sys.modules. Python 3.14 switched the POSIX default start method
    to forkserver, whose workers start clean and re-import -- at which point
    `model_01.data.transforms` does not exist and unpickling RobustnessAugment
    fails with "No module named 'model_01'".

    Registering a finder fixes it at the source, so the aliases resolve in any
    fresh interpreter. The alternative, forcing multiprocessing back to 'fork', is
    worse here: forking a process that has already initialised CUDA is a known
    deadlock, and this code is meant to run on GPU nodes.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _ALIAS_PACKAGES:
            spec = importlib.machinery.ModuleSpec(
                fullname, _AliasPackageLoader(), is_package=True
            )
            spec.submodule_search_locations = []
            return spec
        relpath = _ALIAS_FILES.get(fullname)
        if relpath is None:
            return None
        target_path = MODEL_01_ROOT / relpath
        if not target_path.exists():
            raise ImportError(
                f"model_02 shares data/augmentation/metric code with model_01, but "
                f"{target_path} does not exist. Expected layout: server/model_01/ "
                f"and server/model_02/ side by side."
            )
        return importlib.util.spec_from_file_location(fullname, target_path)


def _install_finder() -> None:
    if not any(isinstance(f, ModelO1AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, ModelO1AliasFinder())


_install_finder()


def load_model_01_module(relpath: str, alias: str) -> ModuleType:
    """Import `model_01/<relpath>` under the private name `alias`."""
    if alias in _ALIAS_FILES and _ALIAS_FILES[alias] != relpath:
        raise ValueError(f"alias {alias!r} is already mapped to {_ALIAS_FILES[alias]!r}")
    _ALIAS_FILES.setdefault(alias, relpath)
    _install_finder()
    return importlib.import_module(alias)


_datasets = load_model_01_module("data/datasets.py", "model_01.data.datasets")
_transforms = load_model_01_module("data/transforms.py", "model_01.data.transforms")
_metrics = load_model_01_module("eval/metrics.py", "model_01.eval.metrics")
_realdeg = load_model_01_module("eval/realdeg.py", "model_01.eval.realdeg")

RealFakeImageDataset = _datasets.RealFakeImageDataset
ImageFolderInference = _datasets.ImageFolderInference
IMAGE_EXTENSIONS = _datasets.IMAGE_EXTENSIONS

RobustnessAugment = _transforms.RobustnessAugment
SEVERITY_LEVELS = _transforms.SEVERITY_LEVELS
apply_named_transform = _transforms.apply_named_transform

RealDegSampler = _realdeg.RealDegSampler
RealDegStep = _realdeg.Step
REALDEG_OPERATORS = _realdeg.OPERATORS
REALDEG_STRENGTHS = _realdeg.STRENGTHS
apply_realdeg_chain = _realdeg.apply_chain
balanced_accuracy = _realdeg.balanced_accuracy

compute_all_metrics = _metrics.compute_all_metrics
fpr_at_threshold = _metrics.fpr_at_threshold
threshold_for_target_fpr = _metrics.threshold_for_target_fpr

__all__ = [
    "MODEL_01_ROOT",
    "ModelO1AliasFinder",
    "load_model_01_module",
    "RealFakeImageDataset",
    "ImageFolderInference",
    "IMAGE_EXTENSIONS",
    "RobustnessAugment",
    "SEVERITY_LEVELS",
    "apply_named_transform",
    "RealDegSampler",
    "RealDegStep",
    "REALDEG_OPERATORS",
    "REALDEG_STRENGTHS",
    "apply_realdeg_chain",
    "balanced_accuracy",
    "compute_all_metrics",
    "fpr_at_threshold",
    "threshold_for_target_fpr",
]
