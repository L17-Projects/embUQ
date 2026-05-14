from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from emb.compression.evalkit import tools


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
    csv_path = tmp_path / "emb.compression.csv"
    csv_path.write_text(
        "# header 1\n# header 2\n# header 3\n"
        "0.1,1.0\n0.2,2.0\n0.3,3.0\n0.4,4.0\n0.5,5.0\n0.6,6.0\n0.7,7.0\n",
        encoding="utf-8",
    )

    tools.generateCompressionData(str(csv_path), str(tmp_path))

    interpolated = np.loadtxt(csv_path.with_name("emb.compression_interp.dat"), skiprows=1, ndmin=2)
    assert interpolated.shape == (4, 2)
    assert np.allclose(interpolated[:, 0], [0.4, 0.5, 0.6, 0.7])
    assert np.allclose(interpolated[:, 1], [4.0, 5.0, 6.0, 7.0])


def test_convert_to_dpd_units_writes_expected_reference_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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


def test_convert_to_dpd_units_accepts_explicit_output_file(tmp_path: Path):
    init_root = tmp_path / "_init_compression_2.1um"
    param_file = init_root / "parameter" / "parameters-default00001.yaml"
    _write_compression_params(param_file)

    input_file = tmp_path / "compression_interp.dat"
    output_file = tmp_path / "lane" / "compression_data_2.1um.dat"
    input_file.write_text("Force [nN]  Displacement [nm]\n2.0 3.0\n", encoding="utf-8")

    tools.convertToDPDUnits(str(input_file), str(init_root) + "/", 2.1, output_file=str(output_file))

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


def test_prepare_compression_rejects_missing_explicit_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_root = tmp_path / "project"
    source_dir = project_root / "emb" / "compression" / "src"
    data_dir = project_root / "emb" / "compression" / "evalkit" / "data"
    source_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "data_1.csv").write_text(
        "# h1\n# h2\n# h3\n0.1,1.0\n0.2,2.0\n0.3,3.0\n0.4,4.0\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(project_root)

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        tools.prepareCompression(
            2.1,
            data_dir=str(data_dir),
            data_prefix="compression_data_",
            data_file=str(data_dir / "missing.csv"),
            init_path=str(tmp_path / "lane" / "_init_compression_2.1um"),
        )


def test_find_project_root_falls_back_to_module_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_root = tmp_path / "project"
    source_dir = project_root / "emb" / "compression" / "src"
    evalkit_dir = project_root / "emb" / "compression" / "evalkit"
    source_dir.mkdir(parents=True)
    evalkit_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.chdir(outside)
    monkeypatch.setattr(tools, "__file__", str(evalkit_dir / "tools.py"))

    assert tools._find_project_root() == str(project_root)


def test_resolve_compression_paths_handles_relative_defaults_and_init(tmp_path: Path):
    project_root = tmp_path / "project"
    data_root = project_root / "lane-data"
    data_root.mkdir(parents=True)
    raw_csv = data_root / "data_1.csv"
    raw_csv.write_text("# h1\n# h2\n# h3\n0.1,1.0\n", encoding="utf-8")

    raw, target, init_root = tools._resolve_compression_paths(
        2.1,
        str(project_root),
        data_dir="lane-data",
        data_prefix="compression_data_",
        data_file=None,
        init_path="runtime/_init_compression_2.1um",
    )

    assert raw == str(raw_csv)
    assert target == str(data_root / "compression_data_2.1um.dat")
    assert init_root == str((project_root / "runtime" / "_init_compression_2.1um").resolve()) + "/"


def test_resolve_compression_paths_uses_fallback_raw_data_only_for_generated_dpd(
    tmp_path: Path,
):
    project_root = tmp_path / "project"
    fallback_root = project_root / "emb" / "compression" / "evalkit" / "data"
    fallback_root.mkdir(parents=True)
    fallback_csv = fallback_root / "data_1.csv"
    fallback_csv.write_text("# h1\n# h2\n# h3\n0.1,1.0\n", encoding="utf-8")

    external_root = tmp_path / "lane-data"
    external_root.mkdir()
    output_file = external_root / "compression_data_2.1um.dat"

    raw, target, init_root = tools._resolve_compression_paths(
        2.1,
        str(project_root),
        data_dir=str(external_root),
        data_prefix="compression_data_",
        data_file=str(output_file),
        init_path=None,
    )

    assert raw == str(fallback_csv)
    assert target == str(output_file)
    assert init_root == str((project_root / "_init_compression_2.1um").resolve()) + "/"


def test_resolve_compression_paths_raises_when_raw_data_is_missing(tmp_path: Path):
    project_root = tmp_path / "project"
    (project_root / "emb" / "compression" / "evalkit" / "data").mkdir(parents=True)
    external_root = tmp_path / "lane-data"
    external_root.mkdir()

    with pytest.raises(FileNotFoundError, match="data_1.csv"):
        tools._resolve_compression_paths(
            2.1,
            str(project_root),
            data_dir=str(external_root),
            data_prefix="compression_data_",
            data_file=str(external_root / "compression_data_2.1um.dat"),
            init_path=None,
        )


def test_resolve_compression_paths_handles_relative_explicit_data_file(tmp_path: Path):
    project_root = tmp_path / "project"
    data_root = project_root / "data"
    data_root.mkdir(parents=True)
    raw_csv = data_root / "raw.csv"
    raw_csv.write_text("# h1\n# h2\n# h3\n0.1,1.0\n", encoding="utf-8")

    raw, target, _init_root = tools._resolve_compression_paths(
        2.1,
        str(project_root),
        data_dir="data",
        data_prefix="compression_data_",
        data_file="data/raw.csv",
        init_path=None,
    )

    assert raw == str(raw_csv)
    assert target == str(data_root / "compression_data_2.1um.dat")


def test_load_source_module_raises_when_spec_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(tools.importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="Could not load compression source module"):
        tools._load_source_module("generate", str(tmp_path))


def test_prepare_compression_rejects_empty_parameter_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_root = tmp_path / "project"
    source_dir = project_root / "emb" / "compression" / "src"
    data_dir = project_root / "emb" / "compression" / "evalkit" / "data"
    source_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "data_1.csv").write_text(
        "# h1\n# h2\n# h3\n0.1,1.0\n0.2,2.0\n0.3,3.0\n0.4,4.0\n",
        encoding="utf-8",
    )
    (source_dir / "generate.py").write_text("def generate_sim(**kwargs):\n    return None\n", encoding="utf-8")
    (source_dir / "parameters.py").write_text(
        "from pathlib import Path\n"
        "def write_parameters(source_path, simu_path, simnum):\n"
        "    path = Path(simu_path) / 'parameter' / f'parameters-default{simnum}.yaml'\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text('', encoding='utf-8')\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(tools.os, "system", lambda cmd: 0)

    with pytest.raises(ValueError, match="parameter template is empty"):
        tools.prepareCompression(
            2.1,
            data_dir=str(data_dir),
            data_prefix="compression_data_",
            data_file=str(data_dir / "compression_data_2.1um.dat"),
            init_path=str(tmp_path / "lane" / "_init_compression_2.1um"),
        )


def test_prepare_compression_with_mocked_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project_root = tmp_path / "project"
    source_dir = project_root / "emb" / "compression" / "src"
    data_dir = project_root / "emb" / "compression" / "evalkit" / "data"
    source_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "data_1.csv").write_text(
        "# h1\n# h2\n# h3\n0.1,1.0\n0.2,2.0\n0.3,3.0\n0.4,4.0\n",
        encoding="utf-8",
    )
    target_data = data_dir / "compression_data_2.1um.dat"
    target_data.write_text("Displacement [DPD units]  Force [DPD units]\n1.0 2.0\n", encoding="utf-8")

    (source_dir / "generate.py").write_text(
        "def generate_sim(**kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (source_dir / "parameters.py").write_text(
        "from pathlib import Path\n"
        "import yaml\n"
        "def write_parameters(source_path, simu_path, simnum):\n"
        "    path = Path(simu_path) / 'parameter' / f'parameters-default{simnum}.yaml'\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(yaml.dump({'ul': 1.0e-7}), encoding='utf-8')\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(tools.os, "system", lambda cmd: 0)

    lane_init = tmp_path / "lane" / "_init_compression_2.1um"
    tools.prepareCompression(
        2.1,
        data_dir=str(data_dir),
        data_prefix="compression_data_",
        data_file=str(target_data),
        init_path=str(lane_init),
    )

    params = yaml.safe_load(
        (
            lane_init / "parameter" / "parameters-default00001.yaml"
        ).read_text(encoding="utf-8")
    )
    assert params["radp"] == pytest.approx(10.5)
    assert params["Lx"] == pytest.approx(26.0)
    assert not (project_root / "_init_compression_2.1um").exists()


def test_prepare_compression_generates_missing_reference_data_in_target_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project_root = tmp_path / "project"
    source_dir = project_root / "emb" / "compression" / "src"
    data_dir = project_root / "emb" / "compression" / "evalkit" / "data"
    source_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    raw_csv = data_dir / "data_1.csv"
    raw_csv.write_text(
        "# h1\n# h2\n# h3\n0.1,1.0\n0.2,2.0\n0.3,3.0\n0.4,4.0\n0.5,5.0\n",
        encoding="utf-8",
    )

    (source_dir / "generate.py").write_text(
        "def generate_sim(**kwargs):\n"
        "    from pathlib import Path\n"
        "    Path(kwargs['simu_path'], 'parameter').mkdir(parents=True, exist_ok=True)\n",
        encoding="utf-8",
    )
    (source_dir / "parameters.py").write_text(
        "from pathlib import Path\n"
        "import yaml\n"
        "def write_parameters(source_path, simu_path, simnum):\n"
        "    path = Path(simu_path) / 'parameter' / f'parameters-default{simnum}.yaml'\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(yaml.dump({"
        "'ul': 1.0e-7, 'rho_water': 1.0, 'rhow': 1.0, 'energyFactor': 1.0, "
        "'kbol': 1.0, 't0': 1.0, 'fscale': 1.0"
        "}), encoding='utf-8')\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(tools.os, "system", lambda cmd: 0)

    lane_init = tmp_path / "lane" / "_init_compression_2.1um"
    output_data = tmp_path / "lane-data" / "compression_data_2.1um.dat"
    tools.prepareCompression(
        2.1,
        data_dir=str(data_dir),
        data_prefix="compression_data_",
        data_file=str(output_data),
        init_path=str(lane_init),
    )

    assert output_data.exists()
    assert (lane_init / "reference_data" / "data_1_interp.dat").exists()
    assert not (project_root / "_init_compression_2.1um").exists()


def test_prepare_compression_raises_when_project_root_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tools, "__file__", str(tmp_path / "compression" / "evalkit" / "tools.py"))
    with pytest.raises(RuntimeError, match="Could not find project root"):
        tools.prepareCompression(2.1)


def test_convert_to_dpd_units_rejects_empty_parameter_template(tmp_path: Path):
    init_root = tmp_path / "_init_compression_2.1um"
    param_file = init_root / "parameter" / "parameters-default00001.yaml"
    param_file.parent.mkdir(parents=True)
    param_file.write_text("", encoding="utf-8")
    input_file = tmp_path / "compression_interp.dat"
    input_file.write_text("Force [nN]  Displacement [nm]\n2.0 3.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="parameter template is empty"):
        tools.convertToDPDUnits(str(input_file), str(init_root) + "/", 2.1)
