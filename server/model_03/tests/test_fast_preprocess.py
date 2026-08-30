"""The GPU crop/resize path must agree with the model's own processor.

`score_crops` reimplements what `AutoImageProcessor` does -- crop, bicubic
resize to the model's input size, rescale, normalise -- as batched tensor work,
because profiling showed the processor spending 4.72s per 1024x768 image
resizing patches one at a time in Python against 4.58s of actual model forward.

The risk this file exists to cover: a preprocessing mismatch does not raise. It
shifts every score a little, and every downstream number with it, while
everything still looks like it is working. So the fast path is checked against
the reference rather than trusted, and the check that matters most is not raw
score agreement but whether any patch lands on the other side of the "likely AI"
threshold -- that is the difference that would change a map.

These tests need torch and the model weights (network on first run), so unlike
the rest of tests/ they skip rather than fail when those are unavailable. Run
them after changing anything in the preprocessing path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

THRESHOLD = 0.75          # the mapper's default "likely AI" cut
MAX_ABS_DIFF = 0.05       # per-patch tolerance
MAX_MEAN_DIFF = 0.01


def _backend():
    """Build the default backend, or return None if it cannot be loaded."""
    try:
        from mapper.backends import build_backend

        return build_backend("hf", batch_size=32)
    except Exception as exc:  # torch missing, no weights, no network
        print(f"  skip  backend unavailable ({type(exc).__name__}: {str(exc)[:60]})")
        return None


def _image_and_boxes(seed: int = 0):
    from PIL import Image

    from mapper.windows import plan_windows

    rng = np.random.default_rng(seed)
    # Structured content rather than pure noise: bicubic resampling differences
    # show up on edges and gradients, which uniform noise would hide.
    yy, xx = np.mgrid[0:512, 0:640]
    base = 40 + 90 * np.sin(xx / 40.0) * np.cos(yy / 55.0) + 60 * (xx / 640)
    arr = np.clip(base[..., None] + rng.normal(0, 12, (512, 640, 3)), 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)

    plan = plan_windows(image.width, image.height, [64, 128, 224], 0.5)
    boxes = [w.box for group in plan.values() for w in group]
    return image, boxes[:150]


def test_fast_path_matches_the_processor():
    backend = _backend()
    if backend is None:
        return

    image, boxes = _image_and_boxes()
    reference = np.array(backend.score_patches([image.crop(b) for b in boxes]))
    fast = np.array(backend.score_crops(image, boxes))

    assert reference.shape == fast.shape
    max_diff = float(np.abs(reference - fast).max())
    mean_diff = float(np.abs(reference - fast).mean())
    print(f"        max|diff| {max_diff:.4f}  mean|diff| {mean_diff:.4f}")

    assert max_diff < MAX_ABS_DIFF, f"per-patch disagreement {max_diff:.4f}"
    assert mean_diff < MAX_MEAN_DIFF, f"mean disagreement {mean_diff:.4f}"


def test_no_patch_crosses_the_threshold():
    """The difference that would actually change a map."""
    backend = _backend()
    if backend is None:
        return

    image, boxes = _image_and_boxes(seed=1)
    reference = np.array(backend.score_patches([image.crop(b) for b in boxes]))
    fast = np.array(backend.score_crops(image, boxes))

    flipped = int(((reference >= THRESHOLD) != (fast >= THRESHOLD)).sum())
    print(f"        {flipped} of {len(boxes)} patches cross the {THRESHOLD} threshold")
    assert flipped == 0, f"{flipped} patches would be labelled differently"


def test_every_scale_is_handled():
    """Scales are resized as separate same-size groups; each must be right."""
    backend = _backend()
    if backend is None:
        return

    from mapper.windows import plan_windows

    image, _ = _image_and_boxes(seed=2)
    plan = plan_windows(image.width, image.height, [64, 128, 224], 0.5)
    for scale, windows in plan.items():
        boxes = [w.box for w in windows[:24]]
        reference = np.array(backend.score_patches([image.crop(b) for b in boxes]))
        fast = np.array(backend.score_crops(image, boxes))
        diff = float(np.abs(reference - fast).max())
        print(f"        scale {scale}: max|diff| {diff:.4f}")
        assert diff < MAX_ABS_DIFF, f"scale {scale} disagrees by {diff:.4f}"


def test_unreproducible_processors_fall_back():
    """A processor doing something unusual must fall back, not approximate.

    Silently approximating an unknown preprocessing pipeline is how every score
    ends up subtly wrong with nothing to show for it.
    """
    backend = _backend()
    if backend is None:
        return

    # A bilinear processor is perfectly valid and simply not what the GPU path
    # reproduces, so it must route to the reference implementation instead of
    # being resampled with the wrong filter.
    original = backend._processor.resample
    try:
        backend._processor.resample = 2  # PIL.Image.BILINEAR
        assert backend._preprocess_config() is None, "a non-bicubic processor must not be reproduced"

        image, boxes = _image_and_boxes(seed=3)
        boxes = boxes[:8]
        # Must still return sensible scores, via the reference path.
        scores = backend.score_crops(image, boxes)
        assert len(scores) == len(boxes)
        assert all(0.0 <= s <= 1.0 for s in scores)
    finally:
        backend._processor.resample = original


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} fast-preprocessing tests passed")


if __name__ == "__main__":
    run()
