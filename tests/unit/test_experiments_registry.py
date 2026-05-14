from pathlib import Path

import pytest

from meso_uq.config.models import emb_geometry_id
from meso_uq.experiments import (
    canonical_dataset_id,
    canonical_experiment_id,
    diameter_from_geometry_id,
    load_experiments,
)


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
    assert exp.structure == "emb"
    assert exp.name == "compression"
    assert exp.experiment_id == canonical_experiment_id("emb", "compression")
    assert exp.geometries == [emb_geometry_id(2.1), emb_geometry_id(2.9), emb_geometry_id(3.0)]
    assert exp.diameters == [2.1, 2.9, 3.0]


def test_diameter_from_geometry_id_rejects_non_emb_geometry():
    with pytest.raises(ValueError, match="not a normalized EMB diameter geometry"):
        diameter_from_geometry_id("gv_rad2_height14_28")


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
                "surrogate_dir": "emb/indentation/surrogate/diameters",
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


def test_load_experiments_supports_canonical_gv_dataset_identity(tmp_path: Path):
    config = {
        "experiments": [
            {
                "structure": "gv",
                "name": "stretching",
                "geometries": ["gv_rad2_height14_28"],
                "controls": ["tot_force_500_50000__bpress_-91"],
                "data_dir": "gv/stretching/data",
                "surrogate_dir": "gv/stretching/surrogate",
            }
        ]
    }
    exp = load_experiments(config, tmp_path)[0]
    assert exp.experiment_id == canonical_experiment_id("gv", "stretching")
    assert exp.diameters == []
    assert exp.dataset_name("gv_rad2_height14_28") == canonical_dataset_id(
        "gv",
        "stretching",
        "gv_rad2_height14_28",
        "tot_force_500_50000__bpress_-91",
    )
    assert exp.dataset_name("gv_rad2_height14_28", control="tot_force_500_50000__bpress_-91") == canonical_dataset_id(
        "gv",
        "stretching",
        "gv_rad2_height14_28",
        "tot_force_500_50000__bpress_-91",
    )


def test_gv_dataset_name_rejects_unknown_control_and_geometry(tmp_path: Path):
    config = {
        "experiments": [
            {
                "structure": "gv",
                "name": "stretching",
                "geometries": ["gv_rad2_height14_28"],
                "controls": ["tot_force_500_50000__bpress_-91"],
            }
        ]
    }
    exp = load_experiments(config, tmp_path)[0]

    with pytest.raises(ValueError, match="is not configured"):
        exp.dataset_name("gv_rad2_height14_28", control="unknown")
    with pytest.raises(ValueError, match="is not configured"):
        exp.dataset_name("gv_rad3_height14_28", control="tot_force_500_50000__bpress_-91")


def test_emb_dataset_name_rejects_unconfigured_geometry(tmp_path: Path):
    exp = load_experiments({"experiment": "compression", "emb_diameters": [2.1]}, tmp_path)[0]

    with pytest.raises(ValueError, match="is not configured"):
        exp.dataset_name(2.9)


def test_gv_dataset_name_rejects_ambiguous_default_control(tmp_path: Path):
    config = {
        "experiments": [
            {
                "structure": "gv",
                "name": "stretching",
                "geometries": ["gv_rad2_height14_28"],
                "controls": [
                    "tot_force_500_50000__bpress_-91",
                    "tot_force_1000_50000__bpress_-91",
                ],
            }
        ]
    }
    exp = load_experiments(config, tmp_path)[0]
    with pytest.raises(ValueError, match="requires an explicit control selection"):
        exp.dataset_name("gv_rad2_height14_28")


def test_load_experiments_rejects_ambiguous_structureless_non_emb_experiment(tmp_path: Path):
    config = {
        "experiments": [
            {
                "name": "stretching",
                "geometries": ["gv_rad2_height14_28"],
            }
        ]
    }
    with pytest.raises(ValueError, match="requires an explicit structure"):
        load_experiments(config, tmp_path)


def test_load_experiments_rejects_default_structureless_non_emb_experiment(tmp_path: Path):
    with pytest.raises(ValueError, match="requires an explicit structure"):
        load_experiments({"experiment": "stretching"}, tmp_path)


def test_load_experiments_rejects_gv_without_geometries(tmp_path: Path):
    config = {"experiments": [{"structure": "gv", "name": "stretching"}]}

    with pytest.raises(ValueError, match="must define geometries"):
        load_experiments(config, tmp_path)


def test_load_experiments_rejects_duplicate_structure_scoped_experiment(tmp_path: Path):
    config = {
        "experiments": [
            {"structure": "gv", "name": "stretching", "geometries": ["gv_rad2_height14_28"]},
            {"structure": "gv", "name": "stretching", "geometries": ["gv_rad3_height14_28"]},
        ]
    }

    with pytest.raises(ValueError, match="Duplicate experiment reference"):
        load_experiments(config, tmp_path)
