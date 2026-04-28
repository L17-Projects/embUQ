from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from meso_uq.surrogate import bnn_training
from meso_uq.surrogate.model import save_model_states


class _RecordingSVI:
    def __init__(self, batch_records: list[tuple[int, ...]]) -> None:
        self._batch_records = batch_records

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        _ = y
        self._batch_records.append(tuple(int(v) for v in x[:, 0].detach().cpu().tolist()))
        return float(len(x))


def _fake_make_tensors(df, input_cols, target_col):
    _ = input_cols, target_col
    n = len(df)
    X = torch.arange(n, dtype=torch.float32).reshape(n, 1)
    y = torch.arange(n, dtype=torch.float32).reshape(n, 1)
    return X, y, [0.0], [1.0], [0.0], [1.0]


def _install_recording_bnn_runtime(monkeypatch: pytest.MonkeyPatch, batch_records: list[tuple[int, ...]]) -> None:
    pyro = SimpleNamespace(
        infer=SimpleNamespace(
            SVI=lambda model, guide, optim, loss: _RecordingSVI(batch_records),
            Trace_ELBO=lambda: object(),
        ),
        optim=SimpleNamespace(Adam=lambda cfg: cfg),
        set_rng_seed=lambda seed: None,
        clear_param_store=lambda: None,
    )
    base_model = torch.nn.Linear(1, 1)
    guide = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        bnn_training,
        "build_variational_components",
        lambda **kwargs: (pyro, base_model, object(), guide),
    )
    monkeypatch.setattr(bnn_training, "make_tensors", _fake_make_tensors)
    monkeypatch.setattr(
        bnn_training,
        "_predictive_mean_std",
        lambda pyro, model, guide, x, num_samples: (
            np.zeros(len(x), dtype=np.float64),
            np.zeros(len(x), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(bnn_training, "_snapshot_pyro_params", lambda pyro: {})
    monkeypatch.setattr(bnn_training, "_restore_pyro_params", lambda pyro, snapshot, device: None)
    monkeypatch.setattr(bnn_training, "_evaluate_dnn_rmse", lambda **kwargs: 1.0)


def _run_bnn_and_capture_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_size: int,
    seed: int,
    max_steps: int = 5,
    max_epochs: int | None = None,
) -> tuple[list[tuple[int, ...]], dict[str, object]]:
    batch_records: list[tuple[int, ...]] = []
    _install_recording_bnn_runtime(monkeypatch, batch_records)

    df = pd.DataFrame(
        {
            "x": np.arange(10, dtype=float),
            "y": np.arange(10, dtype=float),
        }
    )
    result = bnn_training.train_tabular_bnn_surrogate(
        df,
        input_cols=["x"],
        target_col="y",
        out_path=str(tmp_path / f"candidate_bs{batch_size}_seed{seed}.pt"),
        dnn_reference_path=str(tmp_path / "reference.pkl"),
        width=1,
        depth=1,
        batch_size=batch_size,
        lr=1e-3,
        max_steps=max_steps,
        max_epochs=max_epochs,
        eval_every=1,
        predictive_mc_samples=1,
        max_walltime_seconds=30,
        seed=seed,
        require_parity=False,
        device="cpu",
    )
    return batch_records, result


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
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        bnn_training.train_tabular_bnn_surrogate(
            df,
            input_cols=["x"],
            target_col="y",
            out_path="out.pt",
            dnn_reference_path="dnn.pkl",
            max_steps=1,
            batch_size=0,
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
    with pytest.raises(ValueError, match="max_epochs must be >= 1"):
        bnn_training.train_tabular_bnn_surrogate(
            df,
            input_cols=["x"],
            target_col="y",
            out_path="out.pt",
            dnn_reference_path="dnn.pkl",
            max_steps=1,
            max_epochs=0,
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
    pytest.importorskip("pyro")

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


def test_train_tabular_bnn_surrogate_uses_batch_size_for_minibatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_batches, small_result = _run_bnn_and_capture_batches(
        tmp_path / "small",
        monkeypatch,
        batch_size=4,
        seed=7,
    )

    assert [len(batch) for batch in small_batches] == [4, 4, 1, 4, 4]
    assert small_result["training"]["batch_size"] == 4

    large_batches, large_result = _run_bnn_and_capture_batches(
        tmp_path / "large",
        monkeypatch,
        batch_size=20,
        seed=7,
    )

    assert [len(batch) for batch in large_batches] == [9, 9, 9, 9, 9]
    assert large_result["training"]["batch_size"] == 9
    assert small_batches != large_batches


def test_train_tabular_bnn_surrogate_seeded_minibatches_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_batches, _ = _run_bnn_and_capture_batches(
        tmp_path / "run_a",
        monkeypatch,
        batch_size=4,
        seed=11,
    )
    second_batches, _ = _run_bnn_and_capture_batches(
        tmp_path / "run_b",
        monkeypatch,
        batch_size=4,
        seed=11,
    )

    assert first_batches == second_batches


def test_resolve_training_budget_uses_epoch_budget_when_requested() -> None:
    resolved_steps, steps_per_epoch = bnn_training._resolve_training_budget(
        n_train=9,
        batch_size=4,
        max_steps=2500,
        max_epochs=3,
    )
    assert steps_per_epoch == 3
    assert resolved_steps == 9


def test_train_tabular_bnn_surrogate_max_epochs_resolves_dataset_sized_step_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_records, result = _run_bnn_and_capture_batches(
        tmp_path / "epochs",
        monkeypatch,
        batch_size=4,
        seed=7,
        max_steps=99,
        max_epochs=2,
    )
    # n_train is 9 after the DNN-like 90/10 split, so batch_size=4 yields 3 steps per epoch.
    assert len(batch_records) == 6
    assert result["training"]["requested_max_epochs"] == 2
    assert result["training"]["resolved_max_steps"] == 6
    assert result["training"]["steps_per_epoch"] == 3
