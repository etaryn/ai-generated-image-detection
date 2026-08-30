"""Backend selection: spec parsing, and refusing to drop settings on the floor.

`build_backend` is where a config turns into the one learned component the whole
pipeline stands on, so its failure modes matter more than its happy path. These
tests substitute a dummy class into the registry, so nothing here downloads
weights or needs a GPU.

The kwarg test is the important one. Silently ignoring an unrecognised keyword
is how a run ends up scoring with the wrong class or the wrong checkpoint while
looking entirely healthy -- a mistyped `positive_label` that gets dropped leaves
the map inverted, with no error anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapper import backends  # noqa: E402
from mapper.backends import DEFAULT_HF_MODEL, PUBLIC_MODELS, build_backend  # noqa: E402
from mapper.labels import resolve_positive_indices  # noqa: E402


class _DummyBackend:
    """Stands in for a real scorer: records what it was constructed with."""

    def __init__(self, model_id: str | None = None, batch_size: int = 32, device: str | None = None):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self.name = f"dummy:{model_id}"

    def score_patches(self, images):
        return [0.5] * len(images)


def _with_dummy(fn):
    """Run `fn` with the 'hf' registry entry replaced by the dummy."""
    original = backends.BACKENDS["hf"]
    backends.BACKENDS["hf"] = _DummyBackend
    try:
        return fn()
    finally:
        backends.BACKENDS["hf"] = original


def test_bare_spec_uses_the_default_model():
    scorer = _with_dummy(lambda: build_backend("hf"))
    assert scorer.model_id is None, "the backend picks its own default when none is named"


def test_prefixed_spec_selects_a_model():
    scorer = _with_dummy(lambda: build_backend("hf:some/model"))
    assert scorer.model_id == "some/model"


def test_a_bare_hub_id_is_understood_as_a_model():
    scorer = _with_dummy(lambda: build_backend("owner/detector-v2"))
    assert scorer.model_id == "owner/detector-v2"


def test_explicit_model_id_is_not_overwritten_by_the_spec():
    scorer = _with_dummy(lambda: build_backend("hf:from/spec", model_id="from/kwarg"))
    assert scorer.model_id == "from/kwarg", "an explicit kwarg must win over the spec"


def test_unknown_backend_names_raise():
    try:
        build_backend("model_99")
    except ValueError as exc:
        assert "hf" in str(exc), "the error should name the valid options"
        return
    raise AssertionError("an unknown backend should raise")


def test_none_valued_settings_are_ignored_not_forwarded():
    # analyze.py's DEFAULTS carry a null for every optional setting, including
    # ones this backend does not take; nulls must simply not be passed on.
    scorer = _with_dummy(lambda: build_backend("hf", model_id="a/b", checkpoint=None, fp16=None))
    assert scorer.model_id == "a/b"


def test_a_setting_the_backend_cannot_accept_raises():
    try:
        _with_dummy(lambda: build_backend("hf", positive_labl="fake"))  # typo, on purpose
    except TypeError as exc:
        assert "positive_labl" in str(exc)
        return
    raise AssertionError("an unusable setting must raise rather than be dropped")


def test_the_registry_is_self_consistent():
    assert DEFAULT_HF_MODEL in PUBLIC_MODELS, "the default must be a documented model"
    for model_id, entry in PUBLIC_MODELS.items():
        assert entry["note"] and entry["arch"], model_id
        # The recorded label maps are documentation, but they must at least be
        # ones the resolver handles -- otherwise the shortlist recommends models
        # that would fail on first use.
        indices, _ = resolve_positive_indices(entry["labels"])
        assert len(indices) == 1, f"{model_id}: resolved {indices} from {entry['labels']}"


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} backend-selection tests passed")


if __name__ == "__main__":
    run()
