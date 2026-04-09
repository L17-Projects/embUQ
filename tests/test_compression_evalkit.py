from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from compression.evalkit import tools


def _write_compression_params(path: Path) -> None:
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


def test_public_compression_reference_series_are_well_formed():
    for diameter_um in (2.1, 2.9, 3.0):
        displacements = np.asarray(tools.getReferencePoints(diameter_um), dtype=float)
        forces = np.asarray(tools.getReferenceData(diameter_um), dtype=float)

        assert len(displacements) == len(forces)
        assert len(displacements) > 0
        assert np.all(np.isfinite(displacements))
        assert np.all(np.isfinite(forces))
        assert np.all(displacements >= 0.0)
        assert np.all(np.diff(displacements) >= 0.0)


def test_generate_compression_data_skips_initial_rows(tmp_path: Path):
    csv_path = tmp_path / "compression.csv"
    csv_path.write_text(
        "# header 1\n# header 2\n# header 3\n"
        "0.1,1.0\n0.2,2.0\n0.3,3.0\n0.4,4.0\n0.5,5.0\n0.6,6.0\n0.7,7.0\n",
        encoding="utf-8",
    )

    tools.generateCompressionData(str(csv_path), str(tmp_path))

    interpolated = np.loadtxt(csv_path.with_name("compression_interp.dat"), skiprows=1, ndmin=2)
    assert interpolated.shape == (4, 2)
    assert np.allclose(interpolated[:, 0], [0.4, 0.5, 0.6, 0.7])
    assert np.allclose(interpolated[:, 1], [4.0, 5.0, 6.0, 7.0])


def test_convert_to_dpd_units_writes_expected_reference_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    init_root = tmp_path / "_init_compression_2.1um"
    param_file = init_root / "parameter" / "parameters-default00001.yaml"
    _write_compression_params(param_file)

    evalkit_root = tmp_path / "compression" / "evalkit"
    monkeypatch.setattr(tools, "__file__", str(evalkit_root / "tools.py"))

    input_file = tmp_path / "compression_interp.dat"
    input_file.write_text("Force [nN]  Displacement [nm]\n2.0 3.0\n", encoding="utf-8")

    tools.convertToDPDUnits(str(input_file), str(init_root) + "/", 2.1)

    output_file = evalkit_root / "data" / "compression_data_2.1um.dat"
    assert output_file.exists()
    converted = np.loadtxt(output_file, skiprows=1, ndmin=2)
    assert np.allclose(converted[0], [3e-09, 2e-09])


def test_convert_to_force_from_dpd_units_uses_explicit_template(tmp_path: Path):
    param_file = tmp_path / "_init_compression_2.1um" / "parameter" / "parameters-default00001.yaml"
    _write_compression_params(param_file)

    converted = tools.convertToForceFromDPDUnits(2.0, 2.1, init_dir=str(tmp_path))
    assert converted == pytest.approx(2.0e9)


def test_prepare_compression_rejects_unknown_diameter():
    with pytest.raises(ValueError, match="No data file mapped"):
        tools.prepareCompression(9.9)
