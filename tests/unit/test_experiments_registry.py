from pathlib import Path

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
