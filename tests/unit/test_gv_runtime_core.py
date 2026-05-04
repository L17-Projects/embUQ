from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.structures.gv.geometries import DEFAULT_GV_GEOMETRY
from meso_uq.structures.gv.runtime import ControlSweep, KnownIssue, RuntimeDescriptor
from meso_uq.structures.gv.runtime.catalog import (
    load_runtime_descriptor,
    plan_runtime,
    runtime_module_name,
)


def test_control_sweep_values_and_identity_are_deterministic() -> None:
    sweep = ControlSweep("tot_force", 500.0, 50000.0, 80)

    values = sweep.values()

    assert values[0] == 500.0
    assert values[-1] == 50000.0
    assert len(values) == 80
    assert sweep.identifier_part() == "tot_force_500_50000"


def test_control_sweep_rejects_empty_sweep() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        ControlSweep("bpress", -91.0, -91.0, 0)


def test_runtime_descriptor_builds_manifest_without_writing_outputs(tmp_path: Path) -> None:
    descriptor = RuntimeDescriptor(
        experiment="stretching",
        provenance_root="gv/stretching/src",
        source_files=("run_all.sh", "generate.py", "parameters.py", "equil.py"),
        legacy_import_root="gv_simulation_files/stretching/gv",
        control_sweeps=(
            ControlSweep("tot_force", 500.0, 50000.0, 80),
            ControlSweep("bpress", -91.0, -100.0, 1),
        ),
        sweep_mode="forward",
        first_restart=True,
    )

    dry_run = descriptor.plan(output_root=tmp_path / "_runs" / "gv-runtime")
    manifest = dry_run.to_manifest()

    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "stretching"
    assert manifest["geometry"] == DEFAULT_GV_GEOMETRY.id
    assert manifest["control_id"] == "tot_force_500_50000__bpress_-91"
    assert manifest["dataset_id"].endswith(":tot_force_500_50000__bpress_-91")
    assert manifest["provenance_root"].endswith("gv/stretching/src")
    assert Path(manifest["source_root"]).resolve() == Path(manifest["provenance_root"]).resolve()
    assert manifest["legacy_import_root"].endswith("gv_simulation_files/stretching/gv")
    assert manifest["sweep_mode"] == "forward"
    assert manifest["first_restart"] is True
    assert manifest["runtime_package"] == "mirheo"
    assert manifest["commands"][0]["argv"] == [
        "python3",
        "generate.py",
        "-p",
        "tot_force",
        "500",
        "50000",
        "80",
        "-p",
        "bpress",
        "-91",
        "-100",
        "1",
        "--object",
        "gv",
        "--forward",
        "--first",
    ]
    assert manifest["commands"][1]["argv"] == ["bash", "commands.txt"]
    assert Path(dry_run.source_manifest).is_file()
    manifest_data = Path(dry_run.source_manifest).read_text(encoding="utf-8")
    assert "source_files" in manifest_data


def test_runtime_descriptor_rejects_unknown_control_override(tmp_path: Path) -> None:
    descriptor = RuntimeDescriptor(
        experiment="torsion",
        provenance_root="gv_simulation_files/torsion/gv",
        source_files=("run_all.sh", "generate.py"),
        control_sweeps=(ControlSweep("theta", 0.01, 0.1, 10),),
        sweep_mode="forward",
        first_restart=False,
    )

    with pytest.raises(ValueError, match="Unknown controls"):
        descriptor.plan(output_root=tmp_path, controls={"buck": 0.25})


def test_runtime_descriptor_uses_control_overrides_in_generated_command(tmp_path: Path) -> None:
    descriptor = RuntimeDescriptor(
        experiment="stretching",
        provenance_root="gv/stretching/src",
        source_files=("run_all.sh", "generate.py", "parameters.py", "equil.py"),
        control_sweeps=(
            ControlSweep("tot_force", 500.0, 50000.0, 80),
            ControlSweep("bpress", -91.0, -100.0, 1),
        ),
        sweep_mode="forward",
        first_restart=True,
    )

    dry_run = descriptor.plan(output_root=tmp_path, controls={"tot_force": 750.0})
    manifest = dry_run.to_manifest()

    assert manifest["control_id"] == "bpress_-91__tot_force_750"
    assert manifest["control_sweeps"][0] == {"name": "tot_force", "start": 750.0, "stop": 750.0, "steps": 1}
    assert manifest["commands"][0]["argv"] == [
        "python3",
        "generate.py",
        "-p",
        "tot_force",
        "750",
        "750",
        "1",
        "-p",
        "bpress",
        "-91",
        "-100",
        "1",
        "--object",
        "gv",
        "--forward",
        "--first",
    ]


def test_runtime_plan_records_material_parameter_overrides(tmp_path: Path) -> None:
    descriptor = RuntimeDescriptor(
        experiment="stretching",
        provenance_root="gv/stretching/src",
        source_files=("run_all.sh", "generate.py", "parameters.py", "equil.py"),
        legacy_import_root="gv_simulation_files/stretching/gv",
        control_sweeps=(
            ControlSweep("tot_force", 500.0, 50000.0, 80),
            ControlSweep("bpress", -91.0, -100.0, 1),
        ),
        sweep_mode="forward",
        first_restart=True,
    )
    material_overrides = {
        "ka": 1.1,
        "kb": 0.9,
        "mu": 0.7,
        "b1": 0.2,
        "b2": 0.3,
        "a3": 0.4,
        "a4": 0.5,
        "mu_l": 0.6,
        "c": 0.8,
    }
    dry_run = descriptor.plan(output_root=tmp_path, material_parameter_overrides=material_overrides)
    manifest = dry_run.to_manifest()
    assert manifest["material_parameter_overrides"] == material_overrides

    source_manifest = json.loads(Path(manifest["source_manifest"]).read_text(encoding="utf-8"))
    assert source_manifest["material_parameter_overrides"] == material_overrides


def test_runtime_descriptor_rejects_source_output_root(tmp_path: Path) -> None:
    descriptor = RuntimeDescriptor(
        experiment="eigenmodes",
        provenance_root="gv/eigenmodes/src",
        source_files=("run_all.sh", "generate.py"),
        control_sweeps=(ControlSweep("bpress", -91.0, -91.0, 1),),
        sweep_mode="forward",
        first_restart=True,
    )

    with pytest.raises(ValueError, match=r"must not be inside '/.*(gv_simulation_files|src)'"):
        descriptor.plan(output_root=Path("gv_simulation_files") / "scratch")
    with pytest.raises(ValueError, match=r"must not be (inside '/.*/src'|a source directory)"):
        descriptor.plan(output_root=Path("src"))
    with pytest.raises(ValueError, match="repository root 'gv'"):
        descriptor.plan(output_root=tmp_path / "gv")


def test_runtime_descriptor_rejects_missing_or_outside_source_files(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    nested = source_root / "nested"
    nested.mkdir()
    (nested / "run.sh").write_text("echo nested", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')", encoding="utf-8")

    missing_descriptor = RuntimeDescriptor(
        experiment="custom",
        provenance_root=str(source_root),
        source_files=("missing.py",),
        control_sweeps=(ControlSweep("seed", 1.0, 1.0, 1),),
        sweep_mode="forward",
        first_restart=False,
    )
    with pytest.raises(ValueError, match="source file not found"):
        missing_descriptor.plan(output_root=tmp_path / "runtime-missing")

    outside_descriptor = RuntimeDescriptor(
        experiment="custom",
        provenance_root=str(source_root),
        source_files=(str(outside),),
        control_sweeps=(ControlSweep("seed", 1.0, 1.0, 1),),
        sweep_mode="forward",
        first_restart=False,
    )
    with pytest.raises(ValueError, match="must be under source root"):
        outside_descriptor.plan(output_root=tmp_path / "runtime-outside")

    no_root_descriptor = RuntimeDescriptor(
        experiment="custom",
        provenance_root=str(tmp_path / "does-not-exist"),
        source_files=("run.sh",),
        control_sweeps=(ControlSweep("seed", 1.0, 1.0, 1),),
        sweep_mode="forward",
        first_restart=False,
    )
    with pytest.raises(FileNotFoundError, match="source root does not exist"):
        no_root_descriptor.plan(output_root=tmp_path / "runtime-no-root")


def test_experimental_runtime_requires_opt_in(tmp_path: Path) -> None:
    descriptor = RuntimeDescriptor(
        experiment="shear_flow",
        provenance_root="gv/shear_flow/src",
        legacy_import_root="gv_simulation_files/shear_flow",
        source_files=("run_all_HPC.sh", "generate.py", "equil.py"),
        control_sweeps=(ControlSweep("ptan", 0.4, 0.4, 1),),
        sweep_mode="parallel",
        first_restart=True,
        runtime_package="mirheoOBMD",
        experimental=True,
        known_issues=(
            KnownIssue(
                id="shear-flow-bouncer-candidates",
                severity="blocking",
                summary="Imported shear-flow run exceeds bouncer collision candidates.",
                evidence="output.out",
            ),
        ),
    )

    with pytest.raises(ValueError, match="include_experimental=True"):
        descriptor.plan(output_root=tmp_path)

    manifest = descriptor.plan(output_root=tmp_path, include_experimental=True).to_manifest()
    assert manifest["experimental"] is True
    assert manifest["legacy_import_root"].endswith("gv_simulation_files/shear_flow")
    assert manifest["runtime_package"] == "mirheoOBMD"
    assert manifest["known_issues"][0]["id"] == "shear-flow-bouncer-candidates"


def test_runtime_catalog_rejects_unknown_experiment() -> None:
    with pytest.raises(ValueError, match="Unsupported GV runtime experiment"):
        runtime_module_name("compression")

    assert runtime_module_name("stretching") == "meso_uq.structures.gv.runtime.stretching"


def test_runtime_catalog_loads_real_descriptor_and_plans_dry_run(tmp_path: Path) -> None:
    descriptor = load_runtime_descriptor("stretching")
    dry_run = plan_runtime(
        "stretching",
        output_root=tmp_path,
        geometry="gv_rad2_height14_28",
        controls={"tot_force": 750.0},
    )
    manifest = dry_run.to_manifest()

    assert descriptor.experiment == "stretching"
    assert manifest["experiment"] == "stretching"
    assert manifest["geometry"] == "gv_rad2_height14_28"
    assert manifest["controls"]["tot_force"] == 750.0
    assert manifest["dataset_id"].startswith("gv:stretching:")
    assert Path(manifest["source_manifest"]).is_file()
    assert Path(manifest["work_dir"]).is_dir()


def test_runtime_catalog_preserves_shear_flow_execution_contract(tmp_path: Path) -> None:
    descriptor = load_runtime_descriptor("shear_flow")
    dry_run = plan_runtime(
        "shear_flow",
        output_root=tmp_path,
        include_experimental=True,
    )
    manifest = dry_run.to_manifest()

    assert descriptor.provenance_root.endswith("gv/shear_flow/src")
    assert manifest["runtime_package"] == "mirheoOBMD"
    assert manifest["commands"][0]["argv"] == [
        "python3",
        "generate.py",
        "-p",
        "ptan",
        "0_4",
        "0_4",
        "1",
        "-p",
        "afsi",
        "0",
        "0",
        "1",
        "-p",
        "bpress",
        "-91",
        "-91",
        "1",
        "--object",
        "gv",
        "--parallel",
        "--first",
    ]
    assert manifest["commands"][1]["argv"] == ["sbatch", "run_HPC.sbatch"]
    assert "run_all_HPC.sh" in manifest["source_files"]


def test_runtime_catalog_rejects_module_without_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    from meso_uq.structures.gv.runtime import catalog

    class _ModuleWithoutDescriptor:
        __name__ = "missing_descriptor"

    monkeypatch.setattr(catalog, "import_module", lambda _name: _ModuleWithoutDescriptor())

    with pytest.raises(TypeError, match="must expose DESCRIPTOR"):
        catalog.load_runtime_descriptor("stretching")
