"""Which output means "AI-generated" -- the question that must never be guessed.

Five of the six surveyed public detectors put the AI class at index 0 and one
puts it at index 1. Getting that wrong does not crash anything: it inverts every
score, highlights the untampered regions, and produces a confident, plausible,
completely wrong report. So these tests cover the real label maps verbatim, the
overrides, and -- most importantly -- that unrecognised labels *raise* instead of
falling back to a default index.

Torch-free and network-free: `resolve_positive_indices` takes a plain dict, so
none of this needs a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapper.labels import (  # noqa: E402
    LabelResolutionError,
    normalise,
    resolve_positive_indices,
)

# Exactly what each model's config reported when surveyed.
SURVEYED = {
    "Organika/sdxl-detector": ({0: "artificial", 1: "human"}, [0]),
    "umm-maybe/AI-image-detector": ({0: "artificial", 1: "human"}, [0]),
    "haywoodsloan/ai-image-detector-deploy": ({0: "artificial", 1: "real"}, [0]),
    "Ateeqq/ai-vs-human-image-detector": ({0: "ai", 1: "hum"}, [0]),
    "prithivMLmods/Deep-Fake-Detector-Model": ({0: "Fake", 1: "Real"}, [0]),
    # The one that would be inverted by any index-based assumption.
    "dima806/ai_vs_real_image_detection": ({0: "REAL", 1: "FAKE"}, [1]),
}


def test_every_surveyed_public_model_resolves_correctly():
    for model_id, (id2label, expected) in SURVEYED.items():
        indices, reason = resolve_positive_indices(id2label)
        assert indices == expected, f"{model_id}: got {indices}, expected {expected} ({reason})"


def test_the_reversed_model_is_not_confused_with_the_others():
    """The specific inversion this module exists to prevent."""
    standard, _ = resolve_positive_indices({0: "artificial", 1: "human"})
    reversed_, _ = resolve_positive_indices({0: "REAL", 1: "FAKE"})
    assert standard == [0] and reversed_ == [1], "index-based resolution has crept back in"


def test_case_and_punctuation_are_ignored():
    for label in ("AI_Generated", "ai-generated", "  AIGenerated ", "Ai Generated"):
        indices, _ = resolve_positive_indices({0: label, 1: "real photo"})
        assert indices == [0], label


def test_negated_labels_are_not_matched_as_ai():
    """'not_ai' and 'non-ai' contain 'ai'; substring matching would invert them."""
    for negated in ("not_ai", "non-ai", "notAI"):
        indices, _ = resolve_positive_indices({0: "fake", 1: negated})
        assert indices == [0], f"{negated} was matched as the AI class"


def test_unknown_labels_raise_rather_than_guessing():
    for id2label in ({0: "LABEL_0", 1: "LABEL_1"}, {0: "class_a", 1: "class_b"}, {0: "cat", 1: "dog"}):
        try:
            resolve_positive_indices(id2label)
        except LabelResolutionError as exc:
            assert "positive_label" in str(exc), "the error must say how to fix it"
            continue
        raise AssertionError(f"{id2label} should not have resolved")


def test_a_model_with_no_real_class_raises():
    try:
        resolve_positive_indices({0: "fake", 1: "synthetic"})
    except LabelResolutionError as exc:
        assert "measured against" in str(exc)
        return
    raise AssertionError("a model with no authentic class should raise")


def test_a_model_with_no_ai_class_raises():
    try:
        resolve_positive_indices({0: "real", 1: "human"})
    except LabelResolutionError:
        return
    raise AssertionError("a model with no AI class should raise")


def test_empty_label_map_raises():
    try:
        resolve_positive_indices({})
    except LabelResolutionError as exc:
        assert "positive_index" in str(exc)
        return
    raise AssertionError("an empty id2label should raise")


def test_explicit_label_override_wins():
    indices, reason = resolve_positive_indices({0: "cat", 1: "dog"}, positive_label="dog")
    assert indices == [1] and "positive_label" in reason


def test_explicit_index_override_wins_over_label():
    indices, _ = resolve_positive_indices(
        {0: "artificial", 1: "human"}, positive_label="human", positive_index=0
    )
    assert indices == [0]


def test_overrides_are_validated_not_trusted():
    try:
        resolve_positive_indices({0: "artificial", 1: "human"}, positive_index=7)
    except LabelResolutionError:
        pass
    else:
        raise AssertionError("an out-of-range positive_index should raise")

    try:
        resolve_positive_indices({0: "artificial", 1: "human"}, positive_label="banana")
    except LabelResolutionError:
        return
    raise AssertionError("a positive_label matching nothing should raise")


def test_multiclass_sums_every_ai_side_class():
    indices, _ = resolve_positive_indices({0: "real", 1: "gan", 2: "diffusion"})
    assert sorted(indices) == [1, 2]


def test_normalise():
    assert normalise("AI_Generated") == "aigenerated"
    assert normalise("  Real-Photo ") == "realphoto"


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} label-resolution tests passed")


if __name__ == "__main__":
    run()
