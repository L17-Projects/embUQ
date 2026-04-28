from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _run_script_with_args(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *extra_args: str,
):
    data = tmp_path / "train.dat"
    dnn = tmp_path / "dnn.pkl"
    out = tmp_path / "out.pt"
    data.write_text("placeholder", encoding="utf-8")
    dnn.write_text("placeholder", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_read(path: str, *, curve_axis_name: str, value_name: str):
        captured["read_path"] = path
        captured["curve_axis_name"] = curve_axis_name
        captured["value_name"] = value_name
        return {"dummy": True}

    def fake_train(df, **kwargs):
        captured["df"] = df
        captured["train_kwargs"] = kwargs
        return {
            "out": kwargs["out_path"],
            "training": {
                "final_val_rmse": 1.0,
                "parity_dnn_rmse": 1.0,
                "parity_ratio": 1.0,
                "step_count": 1,
                "stop_reason": "max_steps",
            },
        }

    monkeypatch.setattr(module, "read_compression_training_table", fake_read)
    monkeypatch.setattr(module, "train_tabular_bnn_surrogate", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "emb_train_bnn.py",
            str(data),
            "--dnn-reference",
            str(dnn),
            "--out",
            str(out),
            *extra_args,
        ],
    )

    module.main()
    return captured


def test_compression_emb_train_bnn_accepts_legacy_obs_noise_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "compression_emb_train_bnn_legacy_flag_test",
    )

    captured = _run_script_with_args(
        module,
        monkeypatch,
        tmp_path,
        "--obs-noise",
        "0.42",
    )
    kwargs = captured["train_kwargs"]
    assert captured["curve_axis_name"] == "disp"
    assert captured["value_name"] == "F"
    assert kwargs["obs_noise_prior_scale"] == pytest.approx(0.42)


def test_compression_emb_train_bnn_accepts_new_prior_scale_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "compression_emb_train_bnn_new_flag_test",
    )

    captured = _run_script_with_args(
        module,
        monkeypatch,
        tmp_path,
        "--obs-noise-prior-scale",
        "0.77",
    )
    kwargs = captured["train_kwargs"]
    assert kwargs["obs_noise_prior_scale"] == pytest.approx(0.77)


def test_compression_emb_train_bnn_passes_max_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "compression_emb_train_bnn_max_epochs_test",
    )

    captured = _run_script_with_args(
        module,
        monkeypatch,
        tmp_path,
        "--max-epochs",
        "25",
    )
    kwargs = captured["train_kwargs"]
    assert kwargs["max_epochs"] == 25
