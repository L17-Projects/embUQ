from __future__ import annotations

import json
from pathlib import Path

import pytest

from meso_uq.structures.gv.runtime import (
    ControlSweep,
    RuntimeDescriptor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_plan_stages_sources_and_writes_source_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "provenance_root"
    source_root.mkdir()
    (source_root / "run_all.sh").write_text("echo run_all", encoding="utf-8")
    nested = source_root / "analysis"
    nested.mkdir()
    (nested / "run.sh").write_text("echo analysis", encoding="utf-8")

    descriptor = RuntimeDescriptor(
        experiment="custom_staging",
        provenance_root=str(source_root),
        legacy_import_root="gv_simulation_files/custom",
        source_files=("run_all.sh", "analysis/run.sh"),
        control_sweeps=(ControlSweep("seed", 1.0, 1.0, 1),),
        sweep_mode="forward",
        first_restart=False,
    )

    dry_run = descriptor.plan(output_root=tmp_path / "runtime-output", geometry="gv_rad2_height14_28")
    manifest = dry_run.to_manifest()
    work_dir = Path(manifest["work_dir"])

    assert work_dir.is_dir()
    assert (work_dir / "run_all.sh").read_text(encoding="utf-8") == "echo run_all"
    assert (work_dir / "analysis" / "run.sh").read_text(encoding="utf-8") == "echo analysis"
    assert Path(manifest["source_manifest"]).is_file()

    source_manifest = json.loads(Path(manifest["source_manifest"]).read_text(encoding="utf-8"))
    assert source_manifest["structure"] == "gv"
    assert source_manifest["experiment"] == "custom_staging"
    assert source_manifest["geometry"] == "gv_rad2_height14_28"
    assert source_manifest["controls"] == {"seed": 1.0}
    assert source_manifest["control_id"] == "seed_1"
    assert source_manifest["dataset_id"] == manifest["dataset_id"]
    assert source_manifest["source_root"] == str(source_root.resolve())
    assert source_manifest["provenance_root"] == str(source_root.resolve())
    assert source_manifest["legacy_import_root"] == "gv_simulation_files/custom"
    assert source_manifest["staged_work_dir"] == str(work_dir)
    assert source_manifest["source_files"] == ["run_all.sh", "analysis/run.sh"]
    assert source_manifest["source_file_count"] == 2
    assert len(source_manifest["source_sha256"]) == 64
    assert source_manifest["runtime_package"] == "mirheo"
    assert source_manifest["experimental"] is False

    assert manifest["commands"][0]["cwd"] == str(work_dir)


def test_runtime_plan_rejects_gv_simulation_files_output_root(tmp_path: Path) -> None:
    source_root = tmp_path / "provenance_root"
    source_root.mkdir()
    (source_root / "run_all.sh").write_text("echo run_all", encoding="utf-8")

    descriptor = RuntimeDescriptor(
        experiment="custom_staging",
        provenance_root=str(source_root),
        source_files=("run_all.sh",),
        control_sweeps=(ControlSweep("seed", 1.0, 1.0, 1),),
        sweep_mode="forward",
        first_restart=False,
    )

    with pytest.raises(ValueError, match=r"must not be inside '.*gv_simulation_files"):
        descriptor.plan(output_root=REPO_ROOT / "gv_simulation_files" / "blocked")
