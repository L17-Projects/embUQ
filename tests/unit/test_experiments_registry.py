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


def test_experiment_phase1_prior_overrides_use_diameter_keys(tmp_path: Path):
    config = {
        "emb_diameters": [3.2, 3.4],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.0, 1.0],
        "experiments": [
            {
                "name": "indentation",
                "enabled": True,
                "diameters": [3.2, 3.4],
                "prior_ka_by_diameter_um": {
                    "3.2": [17951.817681, 19959.913358],
                    "3.4": [17945.070213, 18942.065540],
                },
                "prior_kb_by_diameter_um": {
                    "3.2": [8369.922251, 9702.253910],
                    "3.4": [1191.930971, 1261.869441],
                },
            }
        ],
    }

    exp = load_experiments(config, tmp_path)[0]

    assert exp.phase1_prior_overrides(3.2) == {
        "ka": [17951.817681, 19959.913358],
        "kb": [8369.922251, 9702.253910],
    }
    assert exp.phase1_prior_overrides(3.4) == {
        "ka": [17945.070213, 18942.065540],
        "kb": [1191.930971, 1261.869441],
    }
    assert exp.phase1_prior_overrides(5.8) == {}


def test_experiment_phase1_prior_overrides_use_experiment_defaults_before_diameter_keys(
    tmp_path: Path,
) -> None:
    config = {
        "emb_diameters": [2.1, 2.9],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.0, 1.0],
        "experiments": [
            {
                "name": "compression",
                "enabled": True,
                "diameters": [2.1, 2.9],
                "prior_ka": [5600.0, 55800.0],
                "prior_kb": [101.0, 999.0],
                "prior_ka_by_diameter_um": {
                    "2.9": [6000.0, 50000.0],
                },
            }
        ],
    }

    exp = load_experiments(config, tmp_path)[0]

    assert exp.phase1_prior_overrides(2.1) == {
        "ka": [5600.0, 55800.0],
        "kb": [101.0, 999.0],
    }
    assert exp.phase1_prior_override_sources(2.1) == {
        "ka": "experiment_override",
        "kb": "experiment_override",
    }
    assert exp.phase1_prior_overrides(2.9) == {
        "ka": [6000.0, 50000.0],
        "kb": [101.0, 999.0],
    }
    assert exp.phase1_prior_override_sources(2.9) == {
        "ka": "diameter_override",
        "kb": "experiment_override",
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([2.0], "exactly two values"),
        ([2.0, 1.0], "min < max"),
        (4.0, "two-value sequence"),
    ],
)
def test_experiment_phase1_prior_overrides_reject_malformed_bounds(
    tmp_path: Path,
    value,
    message: str,
) -> None:
    config = {
        "emb_diameters": [3.2],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.0, 1.0],
        "experiments": [
            {
                "name": "indentation",
                "enabled": True,
                "diameters": [3.2],
                "prior_ka_by_diameter_um": {"3.2": value},
            }
        ],
    }

    exp = load_experiments(config, tmp_path)[0]

    with pytest.raises(ValueError, match=message):
        exp.phase1_prior_overrides(3.2)


def test_emb_lanes_have_distinct_dataset_identity_and_labels(tmp_path: Path):
    config = {
        "structure": "emb",
        "experiments": [
            {
                "name": "compression",
                "lane": "soft",
                "enabled": True,
                "diameters": [4.1],
                "diameter_labels": {"4.1": "4.10"},
                "data_dir": "soft/data",
                "data_prefix": "compression_soft_data_",
                "surrogate_parameterization": "direct_ka_kb",
            },
            {
                "name": "compression",
                "lane": "hard",
                "enabled": True,
                "diameters": [4.1],
                "diameter_labels": {"4.1": "4.10"},
                "data_dir": "hard/data",
                "data_prefix": "compression_hard_data_",
            },
        ],
    }

    soft, hard = load_experiments(config, tmp_path)

    assert soft.name == "compression"
    assert soft.routing_name == "compression_soft"
    assert soft.experiment_id == canonical_experiment_id("emb", "compression_soft")
    assert soft.dataset_name(4.1) == "compression_soft_4.10um"
    assert soft.data_file(4.1) == tmp_path / "soft" / "data" / "compression_soft_data_4.10um.dat"
    assert soft.surrogate_parameterization == "direct_ka_kb"
    assert hard.experiment_id == canonical_experiment_id("emb", "compression_hard")
    assert hard.dataset_name(4.1) == "compression_hard_4.10um"


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


def test_emb_dataset_name_rejects_unconfigured_geometry(tmp_path: Path):
    exp = load_experiments({"experiment": "compression", "emb_diameters": [2.1]}, tmp_path)[0]

    with pytest.raises(ValueError, match="is not configured"):
        exp.dataset_name(2.9)


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


def test_load_experiments_rejects_duplicate_lanes(tmp_path: Path):
    config = {
        "experiments": [
            {"structure": "emb", "name": "compression", "lane": "soft", "diameters": [4.1]},
            {"structure": "emb", "name": "compression", "lane": "soft", "diameters": [4.71]},
        ]
    }

    with pytest.raises(ValueError, match="Duplicate experiment reference 'emb:compression_soft'"):
        load_experiments(config, tmp_path)
