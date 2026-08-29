"""Forward-pass tests for the two pretrained-backbone extractors.

These run against *randomly initialized* models of the real architectures, not
downloaded checkpoints: the point is to check the plumbing (preprocessing,
pooling, output width, L2 normalization, batch handling) without making the test
suite depend on a ~1GB weight download. Behaviour of the actual pretrained
weights is what the ablation table (eval/ablation.py) measures, and that needs
real data anyway.

Run tests/test_fft_features.py + this file before kicking off a long extraction:
between them they cover all three Step-1 branches.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CANONICAL = 256


def _canonical_batch(n: int = 2) -> "torch.Tensor":
    torch.manual_seed(0)
    return torch.rand(n, 3, CANONICAL, CANONICAL)


def test_dinov2_pooling_and_shape():
    """CLS / mean / CLS+mean pooling widths, on a tiny DINOv2 built from config."""
    try:
        import transformers
        from transformers import Dinov2Config, Dinov2Model
    except ImportError:
        print("  (transformers not installed -- skipping)")
        return

    from features.dino import DinoV2Features

    hidden = 32
    tiny_cfg = Dinov2Config(
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=224,
        patch_size=14,
    )
    original = transformers.AutoModel.from_pretrained
    transformers.AutoModel.from_pretrained = staticmethod(lambda *a, **k: Dinov2Model(tiny_cfg))
    try:
        for pooling, expected in (("cls", hidden), ("mean", hidden), ("cls_mean", hidden * 2)):
            ex = DinoV2Features(model_name="tiny", pooling=pooling, l2_normalize=True)
            assert ex.dim == expected, f"{pooling}: expected dim {expected}, got {ex.dim}"
            feats = ex(_canonical_batch())
            assert feats.shape == (2, expected)
            assert np.isfinite(feats).all()
            assert len(ex.feature_names()) == ex.dim
            norms = np.linalg.norm(feats, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-4), f"{pooling}: L2 normalization did not apply"
    finally:
        transformers.AutoModel.from_pretrained = original


def test_dinov2_rejects_image_size_that_is_not_a_multiple_of_the_patch_size():
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("  (transformers not installed -- skipping)")
        return

    from features.dino import DinoV2Features

    try:
        DinoV2Features(model_name="tiny", image_size=225)
    except ValueError:
        return
    raise AssertionError("image_size=225 is not a multiple of 14 and should be rejected up front")


def test_clip_encode_image_shape_and_norm():
    """Real open_clip vision tower, random weights (pretrained=None -> no download)."""
    try:
        import open_clip
    except ImportError:
        print("  (open_clip not installed -- skipping)")
        return

    from features.clip import ClipFeatures

    original = open_clip.create_model_and_transforms
    open_clip.create_model_and_transforms = lambda name, pretrained=None, **k: original(
        name, pretrained=None, **k
    )
    try:
        ex = ClipFeatures(backbone_name="ViT-B-32", pretrained="openai", l2_normalize=True)
        assert ex.impl == "open_clip"
        feats = ex(_canonical_batch())
        assert feats.shape == (2, ex.dim)
        assert ex.dim == 512, f"ViT-B-32's projection width should be 512, got {ex.dim}"
        assert np.isfinite(feats).all()
        assert np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-4)
    finally:
        open_clip.create_model_and_transforms = original


def test_resize_and_normalize_is_a_no_op_at_matching_size():
    from features.base import resize_and_normalize

    x = torch.rand(2, 3, 224, 224)
    out = resize_and_normalize(x, 224, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert torch.allclose(out, x), "identity mean/std at the target size should return the input"

    out = resize_and_normalize(x, 112, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert out.shape == (2, 3, 112, 112)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    import traceback

    if not HAS_TORCH:
        print("torch is not installed in this environment -- skipping backbone extractor tests.")
        sys.exit(0)

    tests = {n: o for n, o in list(globals().items()) if n.startswith("test_") and callable(o)}
    passed, failed = 0, 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
