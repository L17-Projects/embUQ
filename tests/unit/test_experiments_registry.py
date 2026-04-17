from pathlib import Path

import pytest

from meso_uq.experiments import load_experiments


def test_load_experiments_default(tmp_path: Path):
    config = {
        "emb_diameters": [3.0, 2.1, 2.9],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.0, 1.0],
        "experiment": "compression",
    }
    experiments = load_experiments(config, tmp_path)
    assert len(experiments) == 1
    exp = experiments[0]
    assert exp.name == "compression"
    assert exp.diameters == [2.1, 2.9, 3.0]


def test_experiment_reference_loading(tmp_path: Path):
    data_dir = tmp_path / "indentation" / "evalkit" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "indentation_data_2.1um.dat"
    data_file.write_text("Displacement Force\n0.0 1.0\n0.5 2.0\n")
    config = {
        "emb_diameters": [2.1],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.0, 1.0],
        "experiments": [
            {
                "name": "indentation",
                "enabled": True,
                "diameters": [2.1],
                "data_dir": str(data_dir),
                "data_prefix": "indentation_data_",
                "surrogate_dir": "indentation/surrogate/diameters",
            }
        ],
    }
    exp = load_experiments(config, Path(tmp_path))[0]
    assert exp.get_reference_points(2.1) == [0.0, 0.5]
    assert exp.get_reference_data(2.1) == [1.0, 2.0]


def test_experiment_data_file_uses_explicit_data_files_with_string_key(tmp_path: Path):
    data_dir = tmp_path / "compression" / "evalkit" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    explicit = data_dir / "mapped_file.dat"
    explicit.write_text("x y\n0.0 0.1\n", encoding="utf-8")
    config = {
        "emb_diameters": [2.1],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.0, 1.0],
        "experiments": [
            {
                "name": "compression",
                "enabled": True,
                "diameters": [2.1],
                "data_dir": str(data_dir),
                "data_prefix": "compression_data_",
                "data_files": {"2.1": "mapped_file.dat"},
            }
        ],
    }
    exp = load_experiments(config, tmp_path)[0]
    assert exp.data_file(2.1) == explicit


def test_csv_reference_requires_dpd_companion_file(tmp_path: Path):
    data_dir = tmp_path / "compression" / "evalkit" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_file = data_dir / "compression_data_2.1um.csv"
    csv_file.write_text("x,y\n0.0,1.0\n", encoding="utf-8")
    config = {
        "emb_diameters": [2.1],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.0, 1.0],
        "experiments": [
            {
                "name": "compression",
                "enabled": True,
                "diameters": [2.1],
                "data_dir": str(data_dir),
                "data_prefix": "compression_data_",
                "data_files": {2.1: "compression_data_2.1um.csv"},
            }
        ],
    }
    exp = load_experiments(config, tmp_path)[0]
    with pytest.raises(FileNotFoundError, match="Compression DPD data missing"):
        exp.get_reference_points(2.1)


def test_load_experiments_rejects_missing_diameters(tmp_path: Path):
    config = {
        "emb_diameters": [2.1],
        "experiments": [
            {
                "name": "compression",
                "enabled": True,
                "diameters": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="must define diameters"):
        load_experiments(config, tmp_path)
