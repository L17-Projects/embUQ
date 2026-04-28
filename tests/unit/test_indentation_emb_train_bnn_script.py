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
    data = tmp_path / "samples_all.dat"
    dnn = tmp_path / "dnn.pkl"
    out = tmp_path / "out.pt"
    data.write_text("placeholder", encoding="utf-8")
    dnn.write_text("placeholder", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_read_indentation(path: str, *, disp_source: str, rupture_ratio_threshold):
        captured["read_path"] = path
        captured["disp_source"] = disp_source
        captured["rupture_ratio_threshold"] = rupture_ratio_threshold
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

    monkeypatch.setattr(module, "read_indentation_table", fake_read_indentation)
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


def test_indentation_emb_train_bnn_uses_read_indentation_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: must use read_indentation_table, not read_wide_curve_table.

    read_wide_curve_table swaps axes (treats diameter columns as F and force
    columns as disp), causing the BNN to learn the wrong function and predict
    force-scale values (~2500) for displacement queries (~0-9).
    """
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "indentation_emb_train_bnn_reader_test",
    )

    # read_wide_curve_table must NOT be importable from the module namespace.
    assert not hasattr(module, "read_wide_curve_table"), (
        "emb_train_bnn.py must not import read_wide_curve_table; "
        "it swaps force/displacement axes for indentation data."
    )

    captured = _run_script_with_args(module, monkeypatch, tmp_path)

    assert "read_path" in captured, "read_indentation_table was not called"
    assert captured["disp_source"] == "auto"
    assert captured["rupture_ratio_threshold"] == pytest.approx(2.0)
    kwargs = captured["train_kwargs"]
    assert kwargs["input_cols"] == ["Yt", "kb", "b1", "b2", "a3", "a4", "F"]
    assert kwargs["target_col"] == "disp"


def test_indentation_emb_train_bnn_accepts_obs_noise_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "indentation_emb_train_bnn_obs_noise_test",
    )
    captured = _run_script_with_args(module, monkeypatch, tmp_path, "--obs-noise", "0.42")
    assert captured["train_kwargs"]["obs_noise_prior_scale"] == pytest.approx(0.42)


def test_indentation_emb_train_bnn_accepts_loader_knobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "indentation_emb_train_bnn_loader_knobs_test",
    )
    captured = _run_script_with_args(
        module,
        monkeypatch,
        tmp_path,
        "--disp-source",
        "displacement",
        "--rupture-ratio",
        "0",
    )
    assert captured["disp_source"] == "displacement"
    assert captured["rupture_ratio_threshold"] is None


def test_indentation_emb_train_bnn_passes_max_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module = _load_module(
        repo_root / "indentation" / "surrogate" / "scripts" / "emb_train_bnn.py",
        "indentation_emb_train_bnn_max_epochs_test",
    )
    captured = _run_script_with_args(
        module,
        monkeypatch,
        tmp_path,
        "--max-epochs",
        "25",
    )
    assert captured["train_kwargs"]["max_epochs"] == 25
