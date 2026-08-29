"""End-to-end test of Step 2: cached features -> train.py -> checkpoint -> predictor.

Runs the real `train.py` main() against a synthetic feature cache, so it covers
the parts that only break in integration: the cache format written by
extract_features.py, the group split, the scaler, checkpoint contents, and
`classifiers.load_predictor` rebuilding a working model from what was saved.

The synthetic cache is linearly separable on purpose -- this asserts the pipeline
is wired up correctly, not that the model is good. XGBoost is exercised too when
the package is installed, and skipped (loudly) when it isn't.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import torch
    import yaml

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    import train as train_module
    from classifiers import load_predictor

N_IMAGES, COPIES, DINO_DIM, FFT_DIM = 200, 2, 8, 4


def _write_cache(path: Path, seed: int = 0):
    """A cache with the same schema extract_features.py writes.

    Two blocks: a "dino" block carrying the class signal and an "fft" block of
    pure noise -- so an ablation on this cache has a knowably right answer.
    """
    rng = np.random.default_rng(seed)
    y_img = rng.integers(0, 2, size=N_IMAGES)

    rows, labels, groups, paths, aug = [], [], [], [], []
    for gid in range(N_IMAGES):
        signal = rng.normal(loc=2.0 * y_img[gid] - 1.0, scale=0.5, size=DINO_DIM)
        for copy_idx in range(1 + COPIES):
            jitter = rng.normal(scale=0.1, size=DINO_DIM) if copy_idx else 0.0
            rows.append(np.concatenate([signal + jitter, rng.normal(size=FFT_DIM)]))
            labels.append(y_img[gid])
            groups.append(gid)
            paths.append(f"/fake/path/img{gid:04d}.jpg")
            aug.append(copy_idx > 0)

    meta = {
        "features": {
            "dim": DINO_DIM + FFT_DIM,
            "blocks": [
                {"name": "dino", "dim": DINO_DIM, "start": 0, "stop": DINO_DIM},
                {"name": "fft", "dim": FFT_DIM, "start": DINO_DIM, "stop": DINO_DIM + FFT_DIM},
            ],
            "extractors": [],
        },
        "feature_names": [f"dino_{i}" for i in range(DINO_DIM)] + [f"fft_{i}" for i in range(FFT_DIM)],
        "config": {
            "data": {"canonical_size": 256},
            "features": {"dinov2": {"enabled": True}, "clip": {"enabled": False}, "fft": {"enabled": True}},
        },
        "aug_copies": COPIES,
        "severity": None,
        "n_source_images": N_IMAGES,
        "demo_eval_set": False,
    }
    np.savez_compressed(
        path,
        X=np.stack(rows).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int64),
        paths=np.asarray(paths),
        aug_flags=np.asarray(aug, dtype=bool),
        meta=json.dumps(meta),
    )


def _write_config(path: Path, tmp: Path, classifier_type: str):
    cfg = {
        "data": {"val_split": 0.2, "canonical_size": 256},
        "features": {"cache_dir": str(tmp), "cache_name": "cache.npz"},
        "classifier": {
            "type": classifier_type,
            "mlp": {
                "hidden_dims": [16],
                "dropout": 0.1,
                "epochs": 12,
                "batch_size": 32,
                "lr": 5.0e-3,
                "weight_decay": 1.0e-4,
                "early_stopping_patience": 0,
            },
            "xgboost": {
                "n_estimators": 40,
                "max_depth": 3,
                "learning_rate": 0.2,
                "early_stopping_rounds": 10,
                "verbose_eval": 0,
            },
        },
        "train": {"checkpoint_dir": str(tmp / "checkpoints"), "seed": 42},
        "eval": {"threshold": 0.5, "target_fpr": 0.05, "output_dir": str(tmp / "eval")},
    }
    path.write_text(yaml.safe_dump(cfg))


def _run_train(tmp: Path, classifier_type: str, extra_argv: list[str] | None = None) -> dict:
    cache_path = tmp / "cache.npz"
    config_path = tmp / "config.yaml"
    _write_cache(cache_path)
    _write_config(config_path, tmp, classifier_type)

    argv = sys.argv
    sys.argv = ["train.py", "--config", str(config_path)] + (extra_argv or [])
    try:
        train_module.main()
    finally:
        sys.argv = argv

    return torch.load(tmp / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)


def test_mlp_train_saves_a_usable_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = _run_train(Path(tmpdir), "mlp")

    assert bundle["classifier_type"] == "mlp"
    for key in ("scaler", "block_spec", "feature_names", "features_config", "canonical_size",
                "feature_columns", "val_metrics", "calibrated_threshold"):
        assert key in bundle, f"checkpoint is missing {key!r}, which infer.py needs"

    assert bundle["val_metrics"]["accuracy"] > 0.9, (
        f"pipeline should solve a linearly separable problem; got "
        f"{bundle['val_metrics']['accuracy']:.3f}"
    )
    assert bundle["scaler"]["mean"].shape == (DINO_DIM + FFT_DIM,)
    assert (bundle["scaler"]["std"] > 0).all()


def test_predictor_reloads_and_scores_in_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = _run_train(Path(tmpdir), "mlp")

    predict = load_predictor(bundle)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, DINO_DIM + FFT_DIM)).astype(np.float32)
    probs = predict(X)
    assert probs.shape == (16,)
    assert np.all((probs >= 0) & (probs <= 1)), "predictions must be probabilities"

    # Standardized-space vectors from each class should score on opposite sides.
    scaler = bundle["scaler"]
    real = (np.concatenate([np.full(DINO_DIM, -1.0), np.zeros(FFT_DIM)]) - scaler["mean"]) / scaler["std"]
    fake = (np.concatenate([np.full(DINO_DIM, 1.0), np.zeros(FFT_DIM)]) - scaler["mean"]) / scaler["std"]
    scores = predict(np.stack([real, fake]).astype(np.float32))
    assert scores[1] > scores[0], "the fake-signal vector should score higher than the real one"


def test_block_selection_records_the_columns_it_trained_on():
    """--blocks fft must both subset the data and record the columns, so infer.py
    can extract the full vector and slice it the same way."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = _run_train(Path(tmpdir), "mlp", ["--blocks", "fft"])

    assert bundle["blocks_used"] == ["fft"]
    assert bundle["feature_columns"] == list(range(DINO_DIM, DINO_DIM + FFT_DIM))
    assert bundle["scaler"]["mean"].shape == (FFT_DIM,)
    assert all(n.startswith("fft_") for n in bundle["feature_names"])
    # The fft block here is pure noise, so this is also a check that the ablation
    # is genuinely training on the columns it says it is.
    assert bundle["val_metrics"]["accuracy"] < 0.75, (
        "a noise-only block should not separate the classes -- the column selection is wrong"
    )


def test_xgboost_train_and_reload():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("  (xgboost not installed -- skipping)")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = _run_train(Path(tmpdir), "xgboost")

    assert bundle["classifier_type"] == "xgboost"
    assert isinstance(bundle["booster_raw"], (bytes, bytearray))
    assert bundle["val_metrics"]["accuracy"] > 0.9

    probs = load_predictor(bundle)(np.zeros((4, DINO_DIM + FFT_DIM), dtype=np.float32))
    assert probs.shape == (4,)
    assert np.all((probs >= 0) & (probs <= 1))


if __name__ == "__main__":
    import traceback

    if not HAS_TORCH:
        print("torch is not installed in this environment -- skipping train round-trip tests.")
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
