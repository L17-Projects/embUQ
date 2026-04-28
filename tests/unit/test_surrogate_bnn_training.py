from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("pyro")

from meso_uq.surrogate import bnn_training
from meso_uq.surrogate.model import save_model_states


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
    # Backward compatibility: legacy callers may still pass obs_noise.
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        bnn_training.train_tabular_bnn_surrogate(
            df,
            input_cols=["x"],
            target_col="y",
            out_path="out.pt",
            dnn_reference_path="dnn.pkl",
            max_steps=0,
            obs_noise=0.2,
        )


def test_resolve_obs_noise_prior_scale_accepts_legacy_alias() -> None:
    resolved = bnn_training._resolve_obs_noise_prior_scale(
        obs_noise_prior_scale=1.0,
        obs_noise=0.25,
    )
    assert resolved == pytest.approx(0.25)


def test_resolve_obs_noise_prior_scale_rejects_conflicting_values() -> None:
    with pytest.raises(ValueError, match="conflicting values"):
        bnn_training._resolve_obs_noise_prior_scale(
            obs_noise_prior_scale=0.5,
            obs_noise=0.2,
        )


def test_train_tabular_bnn_surrogate_writes_report_and_reload_summary(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Yt": np.linspace(1.0, 2.0, 12),
            "kb": np.linspace(2.0, 3.0, 12),
            "b1": np.full(12, 0.1),
            "b2": np.full(12, 0.2),
            "a3": np.full(12, 0.3),
            "a4": np.full(12, 0.4),
            "disp": np.linspace(0.1, 1.2, 12),
            "F": np.linspace(5.0, 10.5, 12),
        }
    )

    dnn_ref = tmp_path / "reference.pkl"
    dnn_model = torch.nn.Linear(7, 1, bias=False)
    with torch.no_grad():
        dnn_model.weight.zero_()
    save_model_states(
        dnn_model,
        xshift=[0.0] * 7,
        xscale=[1.0] * 7,
        yshift=[0.0],
        yscale=[1.0],
        path=str(dnn_ref),
    )

    report_path = tmp_path / "bnn_report.json"
    result = bnn_training.train_tabular_bnn_surrogate(
        df,
        input_cols=["Yt", "kb", "b1", "b2", "a3", "a4", "disp"],
        target_col="F",
        out_path=str(tmp_path / "candidate.pt"),
        dnn_reference_path=str(dnn_ref),
        report_path=str(report_path),
        width=4,
        depth=2,
        batch_size=4,
        lr=1e-3,
        max_steps=2,
        eval_every=1,
        predictive_mc_samples=2,
        max_walltime_seconds=30,
        seed=11,
        require_parity=False,
        device="cpu",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert Path(result["out"]).exists()
    assert payload["training"]["step_count"] >= 1
    assert payload["reload"]["passed_rel_tol_0p05"] in {True, False}
    assert "degradation_rel" in payload["reload"]
