"""Tests for meso_uq.surrogate.training — train_model early stopping."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from meso_uq.surrogate.model import MLP, init_weights
from meso_uq.surrogate.training import train_model


def test_train_model_reduces_loss() -> None:
    """A tiny model on a simple target should reduce validation loss."""
    torch.manual_seed(0)
    X = torch.randn(40, 2)
    y = (X[:, 0:1] + X[:, 1:2]) * 0.5  # simple linear target

    n_val = 10
    Xv, yv = X[:n_val], y[:n_val]
    Xt, yt = X[n_val:], y[n_val:]

    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = MLP(input_dims=2, output_dims=1, hl_dims=[8])
    model.apply(init_weights)

    model, tr_hist, va_hist = train_model(model, loader, Xv, yv, lr=1e-3, max_epoch=50, info_every=100)

    assert len(tr_hist) > 0
    assert len(va_hist) > 0
    # loss should decrease over training
    assert va_hist[-1] < va_hist[0]


def test_train_model_early_stops() -> None:
    """With a very small learning rate and max_number_of_rounds=5, training should stop early."""
    torch.manual_seed(1)
    X = torch.randn(20, 2)
    y = X[:, 0:1]

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=20, shuffle=False)

    model = MLP(input_dims=2, output_dims=1, hl_dims=[4])

    # Very low lr → patience will be exceeded, triggering early stop rounds
    model, tr_hist, va_hist = train_model(model, loader, X, y, lr=1e-8, max_epoch=5000, info_every=10000)

    # Should have stopped well before 5000 epochs
    assert len(tr_hist) < 5000
