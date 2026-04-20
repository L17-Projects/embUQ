from __future__ import annotations

import numpy as np
import pytest
import torch

from meso_uq.surrogate import bnn_training


def test_split_like_dnn_uses_expected_sizes() -> None:
    Xz = torch.arange(100, dtype=torch.float32).reshape(20, 5)
    yz = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    X_phys = np.arange(100, dtype=float).reshape(20, 5)
    y_phys = np.arange(20, dtype=float)

    split = bnn_training._split_like_dnn(Xz, yz, X_phys, y_phys, seed=1234)

    assert split["n_val"] == 2
    assert split["n_train"] == 18
    assert split["X_train"].shape == (18, 5)
    assert split["X_val"].shape == (2, 5)
    assert split["X_val_phys"].shape == (2, 5)
    assert split["y_val_phys"].shape == (2,)


def test_train_tabular_bnn_rejects_invalid_basic_parameters() -> None:
    df = None
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        bnn_training.train_tabular_bnn_surrogate(
            df,
            input_cols=["x"],
            target_col="y",
            out_path="out.pt",
            dnn_reference_path="dnn.pkl",
            max_steps=0,
        )
    with pytest.raises(ValueError, match="parity_tol must be > 0"):
        bnn_training.train_tabular_bnn_surrogate(
            df,
            input_cols=["x"],
            target_col="y",
            out_path="out.pt",
            dnn_reference_path="dnn.pkl",
            max_steps=1,
            parity_tol=0.0,
        )
