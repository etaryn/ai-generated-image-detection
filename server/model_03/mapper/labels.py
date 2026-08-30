"""Working out which of a public classifier's outputs means "AI-generated".

This is the highest-risk twenty lines in model_03, and it is separated out and
tested on its own for that reason. Every public AI-image detector on the Hub
emits two logits; none of them agree on what order they are in or what to call
them. Surveyed at the time of writing:

    Organika/sdxl-detector                 {0: 'artificial', 1: 'human'}
    umm-maybe/AI-image-detector            {0: 'artificial', 1: 'human'}
    haywoodsloan/ai-image-detector-deploy  {0: 'artificial', 1: 'real'}
    Ateeqq/ai-vs-human-image-detector      {0: 'ai',         1: 'hum'}
    prithivMLmods/Deep-Fake-Detector-Model {0: 'Fake',       1: 'Real'}
    dima806/ai_vs_real_image_detection     {0: 'REAL',       1: 'FAKE'}   <-- flipped

Five put the AI class at index 0 and one puts it at index 1. Hard-coding an
index would silently invert that last model: every generated image would read as
authentic and every photograph as generated, the heatmap would highlight exactly
the untampered regions, and nothing anywhere in the pipeline would look broken.
A confidently-wrong forensics tool is worse than no tool, so this module's
governing rule is:

    **resolve by name, and refuse to guess.**

An unrecognised label is an error naming the labels it saw and telling the
caller to set `positive_label` explicitly -- never a fallback to index 0. That
means a new model with idiosyncratic labels fails loudly on the first call
rather than producing a plausible-looking inverted map.

Matching is on *exact normalised tokens*, not substrings. Substring matching
looks more forgiving and is a trap: "not_ai" and "nonai" both contain "ai", so a
substring rule maps the authentic class onto the AI class -- the exact inversion
this module exists to prevent.
"""
from __future__ import annotations

import re

# Normalised label tokens meaning "this class is AI-generated / manipulated".
AI_TOKENS = {
    "ai",
    "aigenerated",
    "aiart",
    "aiimage",
    "artificial",
    "computergenerated",
    "deepfake",
    "diffusion",
    "fake",
    "gan",
    "generated",
    "machine",
    "midjourney",
    "sdxl",
    "stablediffusion",
    "synthetic",
    "spoof",
}

# Normalised label tokens meaning "this class is a real photograph".
REAL_TOKENS = {
    "authentic",
    "camera",
    "genuine",
    "hum",          # Ateeqq/ai-vs-human-image-detector truncates "human"
    "human",
    "humanmade",
    "live",
    "natural",
    "nonai",
    "notai",
    "photo",
    "photograph",
    "photographic",
    "pristine",
    "real",
    "realimage",
    "realphoto",
}


def normalise(label: str) -> str:
    """Lowercase and strip everything that is not a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


class LabelResolutionError(ValueError):
    """Raised when it is not certain which output means "AI-generated"."""


def resolve_positive_indices(
    id2label: dict,
    positive_label: str | None = None,
    positive_index: int | None = None,
) -> tuple[list[int], str]:
    """Return (indices whose probability sums to P(AI), how it was decided).

    `positive_index` wins over `positive_label`, which wins over automatic
    resolution. Both overrides are validated rather than trusted.
    """
    if not id2label:
        raise LabelResolutionError(
            "the model config carries no id2label, so there is no way to tell which "
            "output means AI-generated. Set backend.positive_index explicitly."
        )

    labels = {int(k): str(v) for k, v in id2label.items()}

    if positive_index is not None:
        index = int(positive_index)
        if index not in labels:
            raise LabelResolutionError(
                f"positive_index={index} is not one of this model's outputs "
                f"{sorted(labels)} ({labels})"
            )
        return [index], f"positive_index={index} was set explicitly ('{labels[index]}')"

    if positive_label is not None:
        wanted = normalise(positive_label)
        hits = [i for i, name in labels.items() if normalise(name) == wanted]
        if not hits:
            raise LabelResolutionError(
                f"positive_label={positive_label!r} matches none of this model's "
                f"labels {labels}"
            )
        return hits, f"positive_label={positive_label!r} matched {hits}"

    ai_hits, real_hits, unknown = [], [], []
    for index, name in sorted(labels.items()):
        token = normalise(name)
        if token in AI_TOKENS:
            ai_hits.append(index)
        elif token in REAL_TOKENS:
            real_hits.append(index)
        else:
            unknown.append((index, name))

    if unknown:
        raise LabelResolutionError(
            f"cannot tell which output means AI-generated: label(s) "
            f"{[name for _, name in unknown]} are not recognised (full map: {labels}). "
            f"Set backend.positive_label to the name of the AI class, or "
            f"backend.positive_index to its index. Refusing to guess -- an inverted "
            f"map would look plausible and be exactly wrong."
        )
    if not ai_hits:
        raise LabelResolutionError(
            f"none of this model's labels {labels} name an AI-generated class, so it "
            f"is not an AI-image detector (or it uses vocabulary this module does not "
            f"know). Set backend.positive_label explicitly."
        )
    if not real_hits:
        raise LabelResolutionError(
            f"every label in {labels} reads as an AI class, leaving nothing for "
            f"P(AI) to be measured against. Set backend.positive_label explicitly."
        )

    described = ", ".join(f"{i}:{labels[i]}" for i in ai_hits)
    return ai_hits, f"resolved by label name ({described} vs {len(real_hits)} real class(es))"
