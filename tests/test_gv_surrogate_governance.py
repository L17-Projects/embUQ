from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.config.models import InferenceConfig
from meso_uq.experiments import canonical_experiment_id, load_experiments
from meso_uq.structures.gv import (
    ALL_GV_CONTROLS,
    DEFAULT_GV_GEOMETRY,
    GV_EXPERIMENTS,
    GV_PARAMETER_CONTRACT,
    GV_STRUCTURE,
    build_geometry,
)
from meso_uq.structures.gv.runtime import RUNTIME_EXPERIMENTS, plan_runtime
from meso_uq.surrogate.gv_catalog import resolve_gv_surrogate_catalog_entries


def _prior_kwargs() -> dict[str, list[float]]:
    return {
        "prior_ka": [0.1, 1.1],
        "prior_kb": [0.2, 1.2],
        "prior_mu": [0.3, 1.3],
        "prior_b1": [0.0, 1.0],
        "prior_b2": [0.0, 1.0],
        "prior_a3": [-1.0, 1.0],
        "prior_a4": [0.0, 1.0],
        "prior_mu_l": [0.8, 1.8],
        "prior_c": [0.9, 1.9],
        "prior_sigma": [0.0, 1.0],
    }


def test_gv_controls_are_not_calibrated_parameters() -> None:
    control_names = {control.name for control in ALL_GV_CONTROLS}
    calibrated_names = set(GV_PARAMETER_CONTRACT.calibrated_names)
    nuisance_names = set(GV_PARAMETER_CONTRACT.nuisance_names)

    assert control_names.isdisjoint(calibrated_names)
    assert control_names.isdisjoint(nuisance_names)
    assert "excluded from calibrated" in GV_STRUCTURE.metadata["control_policy"]


def test_gv_config_supports_multiple_geometries_for_every_nonempty_experiment_subset() -> None:
    alternate_geometry = build_geometry(
        radius=2.5,
        height=16.0,
        source="tests/test_gv_surrogate_governance.py",
    )
    geometries = [DEFAULT_GV_GEOMETRY.id, alternate_geometry.id]
    experiment_names = [experiment.name for experiment in GV_EXPERIMENTS]

    for subset_size in range(1, len(experiment_names) + 1):
        for subset in combinations(experiment_names, subset_size):
            config = InferenceConfig(
                structure="gv",
                structures=["gv"],
                experiments=[
                    {
                        "structure": "gv",
                        "name": experiment_name,
                        "geometries": geometries,
                    }
                    for experiment_name in subset
                ],
                **_prior_kwargs(),
            )

            assert [selection.name for selection in config.experiments] == list(subset)
            for selection in config.experiments:
                assert selection.structure == "gv"
                assert selection.geometries == sorted(geometries)


def test_gv_dataset_keys_remain_control_scoped_and_link_runtime_to_surrogate_config(
    tmp_path: Path,
) -> None:
    for experiment_name in RUNTIME_EXPERIMENTS:
        dry_run = plan_runtime(
            experiment_name,
            output_root=tmp_path / "_runs" / experiment_name,
            include_experimental=experiment_name == "shear_flow",
        )
        config = {
            "experiments": [
                {
                    "structure": "gv",
                    "name": experiment_name,
                    "geometries": [dry_run.geometry],
                    "controls": [dry_run.control_id],
                    "data_dir": f"gv/{experiment_name}/references",
                    "surrogate_dir": f"gv/{experiment_name}/surrogate/dnn",
                }
            ]
        }

        experiment = load_experiments(config, tmp_path)[0]
        dataset_id = experiment.dataset_name(dry_run.geometry, control=dry_run.control_id)

        assert dataset_id == dry_run.dataset_id
        assert dataset_id != canonical_experiment_id("gv", experiment_name)
        assert dataset_id.count(":") == 3
        assert experiment.surrogate_dir.parts[-2:] == ("surrogate", "dnn")
        assert "gv_simulation_files" not in experiment.surrogate_dir.parts


def test_gv_surrogate_catalog_tracks_legacy_import_roots_separately() -> None:
    entries = resolve_gv_surrogate_catalog_entries("/repo")

    for entry in entries:
        paths = entry["metadata"]["paths"]
        experiment = str(entry["experiment"])

        assert paths["provenance_root"] == f"gv/{experiment}/src"
        assert paths["source_root"] == f"gv/{experiment}/src"
        assert paths["legacy_import_root"].startswith("gv_simulation_files/")
        assert "gv_simulation_files" not in paths["provenance_root"]
        assert "gv_simulation_files" not in paths["source_root"]
