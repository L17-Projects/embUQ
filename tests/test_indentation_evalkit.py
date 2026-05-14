from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from emb.indentation.evalkit import convert_reference_data
from emb.indentation.evalkit import prepare_env
from emb.indentation.evalkit import tools


def _write_params(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(
            {
                "rho_water": 1.0,
                "rhow": 1.0,
                "energyFactor": 1.0,
                "kbol": 1.0,
                "t0": 1.0,
                "ul": 1.0,
                "fscale": 1.0,
            }
        ),
        encoding="utf-8",
    )


def test_convert_reference_csv_force_displacement(tmp_path: Path):
    params_file = tmp_path / "params.yaml"
    _write_params(params_file)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("# Force, Displacement\n1.0,2.0\n3.0,4.0\n", encoding="utf-8")

    output_path = tmp_path / "converted.dat"
    result = convert_reference_data.convert_reference_csv(
        input_path=csv_path,
        diameter_um=3.4,
        output_path=output_path,
        init_path=str(params_file),
    )

    assert result == output_path
    data = np.loadtxt(output_path, skiprows=1, ndmin=2)
    assert np.allclose(data[:, 0], [1e-09, 3e-09])
    assert np.allclose(data[:, 1], [2e-09, 4e-09])


def test_convert_reference_csv_displacement_force(tmp_path: Path):
    params_file = tmp_path / "params.yaml"
    _write_params(params_file)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("# Displacement, Force\n2.0,1.0\n4.0,3.0\n", encoding="utf-8")

    output_path = tmp_path / "converted.dat"
    convert_reference_data.convert_reference_csv(
        input_path=csv_path,
        diameter_um=3.4,
        output_path=output_path,
        input_order="displacement-force",
        init_path=str(params_file),
    )

    data = np.loadtxt(output_path, skiprows=1, ndmin=2)
    assert np.allclose(data[:, 0], [1e-09, 3e-09])
    assert np.allclose(data[:, 1], [2e-09, 4e-09])


def test_prepare_indentation_converts_csv_to_dpd_file(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_file = data_dir / "indentation_data_2.9um.csv"
    csv_file.write_text("# Force, Displacement\n1.0,2.0\n", encoding="utf-8")
    params_file = tmp_path / "params.yaml"
    _write_params(params_file)

    tools.prepareIndentation(
        2.9,
        data_dir=str(data_dir),
        data_prefix="indentation_data_",
        data_file=str(csv_file),
        init_path=str(params_file),
    )

    dpd_file = data_dir / "indentation_data_2.9um.dat"
    assert dpd_file.exists()
    data = np.loadtxt(dpd_file, skiprows=1, ndmin=2)
    assert np.allclose(data[0], [1e-09, 2e-09])


def test_prepare_env_filter_reference_data_reduces_rows(tmp_path: Path):
    data_path = tmp_path / "indentation_data_3.2um.dat"
    data_path.write_text(
        "Force Displacement\n0.0 0.0\n1.0 0.1\n2.0 0.2\n3.0 0.3\n4.0 0.4\n5.0 0.5\n",
        encoding="utf-8",
    )

    prepare_env._filter_reference_data(3.2, data_path)

    filtered = np.loadtxt(data_path, skiprows=1, ndmin=2)
    assert filtered.shape[0] == 3
    assert np.allclose(filtered[:, 0], [3.0, 4.0, 5.0])


def test_prepare_env_wrappers_require_diameter():
    with pytest.raises(ValueError):
        prepare_env.getReferencePoints()
    with pytest.raises(ValueError):
        prepare_env.getReferenceData()


def test_prepare_env_delegates_and_filters(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_prepare(diameter_um, data_dir=None, data_prefix=None, data_file=None):
        calls["prepare"] = (diameter_um, data_dir, data_prefix, data_file)

    def fake_filter(diameter_um, data_path):
        calls["filter"] = (diameter_um, data_path)

    monkeypatch.setattr(prepare_env, "_prepareIndentation", fake_prepare)
    monkeypatch.setattr(prepare_env, "_filter_reference_data", fake_filter)

    prepare_env.prepareIndentation(
        3.2,
        data_dir="/tmp/indentation",
        data_prefix="prefix_",
        data_file="/tmp/emb/indentation/custom.dat",
    )

    assert calls["prepare"] == (3.2, "/tmp/indentation", "prefix_", "/tmp/emb/indentation/custom.dat")
    assert calls["filter"] == (3.2, Path("/tmp/indentation/prefix_3.2um.dat"))


def test_indentation_reference_columns_from_explicit_dir(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "indentation_data_3.2um.dat"
    data_file.write_text("Force Displacement\n1.0 0.1\n2.0 0.2\n", encoding="utf-8")

    forces = tools.getReferencePoints(3.2, data_dir=str(data_dir))
    displacements = tools.getReferenceData(3.2, data_dir=str(data_dir))

    assert forces == [1.0, 2.0]
    assert displacements == [0.1, 0.2]


def test_convert_reference_main_writes_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    params_file = tmp_path / "params.yaml"
    _write_params(params_file)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("1.0,2.0\n", encoding="utf-8")
    output_path = tmp_path / "converted.dat"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_reference_data.py",
            "--input",
            str(csv_path),
            "--diameter",
            "3.4",
            "--output",
            str(output_path),
            "--init-path",
            str(params_file),
        ],
    )

    convert_reference_data.main()
    assert output_path.exists()
