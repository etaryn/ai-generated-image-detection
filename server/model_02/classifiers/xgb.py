"""XGBoost classifier over the cached feature vector.

Trees are the natural counterpart to the MLP here. The feature vector is a
concatenation of two dense unit-norm embeddings and ~130 hand-built spectral
statistics on wildly different scales; boosted trees don't care about scale,
handle that heterogeneity without tuning, and hand back per-feature importances
that read directly against `features/fft.py`'s named columns -- which is how you
find out *which* artifact the detector is actually keying on.

The trained booster is stored in the checkpoint as raw bytes (`save_raw()`), so a
checkpoint stays a single self-contained .pt with no sidecar model file, and
inference reconstructs a bare Booster without needing the sklearn wrapper.
"""
from __future__ import annotations

import numpy as np

from shared import compute_all_metrics


def _import_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "classifier.type is 'xgboost' but the xgboost package isn't installed "
            "(pip install xgboost>=2.0), or switch classifier.type to 'mlp'."
        ) from exc
    return xgb


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    threshold: float = 0.5,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    """Returns (payload for the checkpoint bundle, per-round history)."""
    xgb = _import_xgboost()

    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": cfg.get("max_depth", 6),
        "eta": cfg.get("learning_rate", 0.05),
        "subsample": cfg.get("subsample", 0.8),
        "colsample_bytree": cfg.get("colsample_bytree", 0.8),
        "min_child_weight": cfg.get("min_child_weight", 1.0),
        "reg_lambda": cfg.get("reg_lambda", 1.0),
        "scale_pos_weight": n_neg / max(n_pos, 1.0),
        "seed": seed,
        "tree_method": cfg.get("tree_method", "hist"),
        "nthread": cfg.get("nthread", 0),
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    evals_result: dict = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.get("n_estimators", 800),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
        evals_result=evals_result,
        verbose_eval=cfg.get("verbose_eval", 50),
    )

    history = [
        {
            "epoch": i,
            "train_loss": evals_result["train"]["logloss"][i],
            "val_auc": evals_result["val"]["auc"][i],
            "val_logloss": evals_result["val"]["logloss"][i],
        }
        for i in range(len(evals_result["val"]["logloss"]))
    ]

    # iteration_range=(0, 0) means "all rounds"; when early stopping fired, cut
    # the prediction off at the best round instead of the last one.
    best_iteration = getattr(booster, "best_iteration", None)
    iteration_range = (0, best_iteration + 1) if best_iteration is not None else (0, 0)
    val_probs = booster.predict(dval, iteration_range=iteration_range)
    metrics = compute_all_metrics(y_val, val_probs, threshold)
    print(
        f"xgboost best_iteration={best_iteration} val_acc={metrics['accuracy']:.4f} "
        f"val_auc={metrics['auc']:.4f} val_fpr={metrics['fpr_at_threshold']:.4f}"
    )

    # gain-based importance, ordered, so the top rows can be read straight off
    # against features/fft.py's column names.
    importance = booster.get_score(importance_type="gain")
    top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:20]

    payload = {
        "classifier_type": "xgboost",
        "booster_raw": bytes(booster.save_raw(raw_format="ubj")),
        "best_iteration": best_iteration,
        "feature_importance_top20": top,
        # Stored so inference can build a DMatrix with the same column names --
        # xgboost refuses to predict when the names it was trained with don't
        # match the ones it's handed.
        "feature_names": feature_names,
        "val_probs": val_probs,
    }
    return payload, history


def load_xgb_predictor(bundle: dict):
    """Rebuild the trained booster as a `predict(X) -> P(fake)` callable."""
    xgb = _import_xgboost()
    booster = xgb.Booster()
    booster.load_model(bytearray(bundle["booster_raw"]))
    best_iteration = bundle.get("best_iteration")
    feature_names = bundle.get("feature_names")
    iteration_range = (0, best_iteration + 1) if best_iteration is not None else (0, 0)

    def predict(X: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(np.asarray(X, dtype=np.float32), feature_names=feature_names)
        return booster.predict(dmat, iteration_range=iteration_range)

    return predict
