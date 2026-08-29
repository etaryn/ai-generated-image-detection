"""MLP classifier over the cached feature vector.

Deliberately small (two hidden layers, ~1.2M params on the default 2178-dim
input): the representation work was already done by the frozen extractors, so
this only has to carve a decision boundary in that space. A bigger head mostly
buys a faster route to memorizing the training set.

Trains on CPU in minutes even for a six-figure row count -- the features are
already in memory as a dense matrix, so there is no image decoding in the loop.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared import compute_all_metrics


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int] = [512, 128], dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers += [
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = hidden
        layers.append(nn.Linear(prev, 1))  # single logit: P(fake)
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,) logits


@torch.no_grad()
def _predict_probs(model: nn.Module, X: np.ndarray, batch_size: int, device) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        chunk = torch.from_numpy(X[start : start + batch_size]).to(device)
        out.append(torch.sigmoid(model(chunk)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> tuple[dict, list[dict]]:
    """Returns (payload for the checkpoint bundle, per-epoch history).

    Model selection is on validation AUC rather than accuracy: AUC is threshold-free,
    so the chosen epoch isn't an artifact of a 0.5 cutoff that eval/error_analysis
    may well move afterwards to hit a false-positive budget.
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    model = FeatureMLP(
        in_dim=X_train.shape[1],
        hidden_dims=cfg.get("hidden_dims", [512, 128]),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)

    epochs = cfg.get("epochs", 40)
    batch_size = cfg.get("batch_size", 256)
    patience = cfg.get("early_stopping_patience", 8)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Guard against an unbalanced cache (e.g. a dataset with far more fakes than
    # reals) skewing the model toward the majority class.
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(X_train) > batch_size,  # BatchNorm needs >1 sample in the last batch
    )

    history: list[dict] = []
    best_auc, best_state, best_epoch, epochs_since_best = -1.0, None, -1, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()

        train_loss = total_loss / max(len(loader.dataset), 1)
        val_probs = _predict_probs(model, X_val, batch_size, device)
        metrics = compute_all_metrics(y_val, val_probs, threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[0],
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_acc={metrics['accuracy']:.4f} "
            f"val_auc={metrics['auc']:.4f} val_fpr={metrics['fpr_at_threshold']:.4f}"
        )

        score = metrics["auc"] if np.isfinite(metrics["auc"]) else metrics["accuracy"]
        if score > best_auc:
            best_auc, best_epoch, epochs_since_best = score, epoch, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            print(f"  -> new best (val_auc={best_auc:.4f})")
        else:
            epochs_since_best += 1
            if patience and epochs_since_best >= patience:
                print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    payload = {
        "classifier_type": "mlp",
        "state_dict": best_state,
        "arch": {
            "in_dim": model.in_dim,
            "hidden_dims": model.hidden_dims,
            "dropout": model.dropout,
        },
        "best_epoch": best_epoch,
        "val_probs": _predict_probs(model, X_val, batch_size, device),
    }
    return payload, history


def load_mlp_predictor(bundle: dict):
    """Rebuild the trained MLP as a `predict(X) -> P(fake)` callable."""
    arch = bundle["arch"]
    model = FeatureMLP(arch["in_dim"], arch["hidden_dims"], arch["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def predict(X: np.ndarray) -> np.ndarray:
        return _predict_probs(model, np.asarray(X, dtype=np.float32), 1024, device)

    return predict
