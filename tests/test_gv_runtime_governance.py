from __future__ import annotations

import subprocess
from pathlib import Path

from meso_uq.experiments import canonical_experiment_id
from meso_uq.structures.gv import EXPERIMENT_CONTROLS, GV_PARAMETER_CONTRACT
from meso_uq.structures.gv.runtime import (
    RUNTIME_EXPERIMENTS,
    load_runtime_descriptor,
    plan_runtime,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GV_GENERATED_SAMPLES = (
    "gv_simulation_files/stretching/gv/mesh/generated.off",
    "gv_simulation_files/stretching/gv/cgal_scripts/scale_space",
    "gv_simulation_files/stretching/gv/commands.txt",
    "gv_simulation_files/stretching/gv/solver.out",
)


def test_generated_gv_samples_are_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-v", *GV_GENERATED_SAMPLES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    for sample in GV_GENERATED_SAMPLES:
        assert sample in stdout


def test_gv_source_of_truth_excludes_generated_samples() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "gv_simulation_files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    tracked_set = set(tracked)
    for sample in GV_GENERATED_SAMPLES:
        assert sample not in tracked_set


def test_wrapper_tree_exists_without_staging_outputs() -> None:
    wrapper_root = REPO_ROOT / "scripts" / "workflows" / "gv"
    assert (wrapper_root / "run_gv_dry_run.py").is_file()
    assert not any(
        path.is_file() and "gv_simulation_files" in path.as_posix()
        for path in wrapper_root.rglob("*")
    )


def test_runtime_catalog_keeps_structure_qualified_control_scoped_identity(
    tmp_path: Path,
) -> None:
    calibrated_names = set(GV_PARAMETER_CONTRACT.calibrated_names)
    nuisance_names = set(GV_PARAMETER_CONTRACT.nuisance_names)

    for experiment_name in RUNTIME_EXPERIMENTS:
        descriptor = load_runtime_descriptor(experiment_name)
        dry_run = plan_runtime(
            experiment_name,
            output_root=tmp_path / experiment_name,
            include_experimental=experiment_name == "shear_flow",
        )
        manifest = dry_run.to_manifest()
        expected_control_names = {
            control.name for control in EXPERIMENT_CONTROLS[experiment_name]
        }

        assert dry_run.structure == "gv"
        assert descriptor.experiment == experiment_name
        assert set(descriptor.control_names) == expected_control_names
        assert expected_control_names.isdisjoint(calibrated_names)
        assert expected_control_names.isdisjoint(nuisance_names)
        assert manifest["control_id"] == dry_run.control_id
        assert manifest["dataset_id"] == dry_run.dataset_id
        assert manifest["dataset_id"].startswith(f"gv:{experiment_name}:{dry_run.geometry}:")
        assert manifest["dataset_id"] != canonical_experiment_id("gv", experiment_name)


def test_runtime_dry_run_canaries_stay_offline_and_provenance_backed(
    tmp_path: Path,
) -> None:
    for experiment_name in RUNTIME_EXPERIMENTS:
        dry_run = plan_runtime(
            experiment_name,
            output_root=tmp_path / "_runs" / experiment_name,
            include_experimental=experiment_name == "shear_flow",
        )
        manifest = dry_run.to_manifest()
        provenance_root = Path(dry_run.provenance_root)
        work_dir = Path(dry_run.work_dir)

        assert provenance_root == REPO_ROOT / "gv" / experiment_name / "src"
        assert dry_run.legacy_import_root.startswith(str(REPO_ROOT / "gv_simulation_files"))
        assert work_dir.is_dir()
        assert Path(dry_run.source_manifest).is_file()

        for source_file in manifest["source_files"]:
            source_path = Path(source_file)
            assert "gv_simulation_files" not in source_path.parts
            assert not source_path.is_absolute()
            assert (provenance_root / source_path).is_file()
            assert (work_dir / source_path).is_file()

        for command in manifest["commands"]:
            argv = command["argv"]
            assert argv
            assert command["cwd"] == dry_run.work_dir
