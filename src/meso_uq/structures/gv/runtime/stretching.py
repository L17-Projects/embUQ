from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..controls import EXPERIMENT_CONTROLS
from ..geometries import DEFAULT_GV_GEOMETRY
from . import ControlSweep, RuntimeDescriptor, RuntimeDryRun, _find_repo_root


STRUCTURE_NAME = "gv"
EXPERIMENT_NAME = "stretching"
REQUIRED_CONTROLS = tuple(control.name for control in EXPERIMENT_CONTROLS[EXPERIMENT_NAME])


REPO_ROOT = _find_repo_root()
SOURCE_ROOT = REPO_ROOT / "gv" / EXPERIMENT_NAME / "src"
PROVENANCE_ROOT = SOURCE_ROOT
LEGACY_IMPORT_ROOT = REPO_ROOT / "gv_simulation_files" / EXPERIMENT_NAME / STRUCTURE_NAME
SOURCE_FILES = tuple(
    str(filename)
    for filename in (
        "clean_all.sh",
        "run_all.sh",
        "generate.py",
        "parameters.py",
        "run.sh",
        "equil.py",
        "parameters-default.gv.yaml",
        "gas_vesicle/create_gv.py",
        "gas_vesicle/parameters.py",
        "gas_vesicle/parameters.yaml",
        "gas_vesicle/run.sh",
        "gas_vesicle/statistics.py",
        "gas_vesicle/add_to_off.py",
    )
)

RUNTIME_DESCRIPTOR = RuntimeDescriptor(
    experiment=EXPERIMENT_NAME,
    provenance_root=str(PROVENANCE_ROOT),
    legacy_import_root=str(LEGACY_IMPORT_ROOT),
    source_files=SOURCE_FILES,
    control_sweeps=(
        ControlSweep(name="tot_force", start=500.0, stop=50000.0, steps=80),
        ControlSweep(name="bpress", start=-91.0, stop=-100.0, steps=1),
    ),
    sweep_mode="forward",
    first_restart=True,
    generate_script="generate.py",
    run_script="run_all.sh",
    generated_subdirs=("logs", "mesh", "anchor", "parameter", "restart"),
    notes=(
        "Imported from gv/stretching/src/run_all.sh.",
        "The staging lane declares bpress as -91.0 -> -100.0 with one step, so the effective default remains -91.0.",
        "Runtime entry points are run_all.sh, generate.py, parameters.py, and equil.py.",
    ),
)
DESCRIPTOR = RUNTIME_DESCRIPTOR


def build_dry_run_descriptor(
    output_root: str | Path,
    *,
    geometry: str = DEFAULT_GV_GEOMETRY.id,
    controls: Mapping[str, float] | None = None,
    material_parameter_overrides: Mapping[str, float] | None = None,
) -> RuntimeDryRun:
    return RUNTIME_DESCRIPTOR.plan(
        output_root=output_root,
        geometry=geometry,
        controls=controls,
        material_parameter_overrides=material_parameter_overrides,
    )


def get_runtime_descriptor() -> RuntimeDescriptor:
    return RUNTIME_DESCRIPTOR


DRY_RUN_DESCRIPTOR = RUNTIME_DESCRIPTOR.plan()
