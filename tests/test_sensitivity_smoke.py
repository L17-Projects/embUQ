import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from meso_uq.sensitivity import VarConfig
from meso_uq.sensitivity import UniformPrior
from meso_uq.sensitivity import build_problem
from meso_uq.sensitivity import run_sobol_over_axis


class _LinearModel(torch.nn.Module):
    def forward(self, values):
        return values[:, :1] + 0.5 * values[:, 1:2]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_model_bundle(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "model": _LinearModel(),
                "xshift": np.zeros(7),
                "xscale": np.ones(7),
                "yshift": np.array([0.0]),
                "yscale": np.array([1.0]),
            },
            handle,
        )


def test_run_sobol_over_axis_returns_expected_shape():
    model = _LinearModel()
    problem = build_problem(
        [
            VarConfig("Yt", UniformPrior(0.0, 1.0)),
            VarConfig("kb", UniformPrior(0.0, 1.0)),
        ]
    )

    result = run_sobol_over_axis(
        model=model,
        xshift=np.array([0.0, 0.0, 0.0]),
        xscale=np.array([1.0, 1.0, 1.0]),
        yshift=np.array([0.0]),
        yscale=np.array([1.0]),
        problem=problem,
        fixed_axis_name="disp",
        fixed_axis_values=[0.2, 0.8],
        evaluate_columns=["Yt", "kb", "disp"],
        n_samples=64,
        calc_second_order=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result["parameter"]) == {"Yt", "kb"}
    assert set(result["index_type"]) == {"S1", "ST"}
    assert set(result["axis"]) == {0.2, 0.8}
    assert len(result) == 8


def test_compression_sobol_cli_writes_csv(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "compression" / "surrogate" / "sensitivity" / "scripts" / "run_sobol_vs_disp.py",
        "compression_sobol_cli_test",
    )
    model_path = tmp_path / "compression_model.pkl"
    output_path = tmp_path / "compression_sobol.csv"
    _write_model_bundle(model_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sobol_vs_disp.py",
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--n-samples",
            "64",
            "--n-displacements",
            "3",
        ],
    )

    module.main()

    result = pd.read_csv(output_path)
    assert set(result["parameter"]) == {"Yt", "kb", "b1", "b2", "a3", "a4"}
    assert set(result["index_type"]) == {"S1", "ST"}


def test_indentation_sobol_cli_writes_csv(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "indentation" / "surrogate" / "sensitivity" / "scripts" / "run_sobol_vs_force.py",
        "indentation_sobol_cli_test",
    )
    model_path = tmp_path / "indentation_model.pkl"
    output_path = tmp_path / "indentation_sobol.csv"
    _write_model_bundle(model_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sobol_vs_force.py",
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--n-samples",
            "64",
            "--n-forces",
            "3",
        ],
    )

    module.main()

    result = pd.read_csv(output_path)
    assert set(result["parameter"]) == {"Yt", "kb", "b1", "b2", "a3", "a4"}
    assert set(result["index_type"]) == {"S1", "ST"}
