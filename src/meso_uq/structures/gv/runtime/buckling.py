from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..controls import EXPERIMENT_CONTROLS
from ..geometries import DEFAULT_GV_GEOMETRY
from . import ControlSweep, RuntimeDescriptor, RuntimeDryRun


STRUCTURE_NAME = "gv"
EXPERIMENT_NAME = "buckling"
REQUIRED_CONTROLS = tuple(control.name for control in EXPERIMENT_CONTROLS[EXPERIMENT_NAME])


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate the repository root from the GV buckling runtime descriptor.")


PROVENANCE_ROOT = _find_repo_root() / "gv_simulation_files" / EXPERIMENT_NAME / STRUCTURE_NAME
SOURCE_FILES = tuple(
    str(PROVENANCE_ROOT / filename)
    for filename in (
        "clean_all.sh",
        "run_all.sh",
        "generate.py",
        "parameters.py",
        "run.sh",
        "equil.py",
        "parameters-default.gv.yaml",
    )
)

RUNTIME_DESCRIPTOR = RuntimeDescriptor(
    experiment=EXPERIMENT_NAME,
    provenance_root=str(PROVENANCE_ROOT),
    source_files=SOURCE_FILES,
    control_sweeps=(
        ControlSweep(name="buck", start=0.0, stop=0.75, steps=50),
        ControlSweep(name="bpress", start=-91.0, stop=-91.0, steps=1),
    ),
    sweep_mode="forward",
    first_restart=False,
    generate_script=str(PROVENANCE_ROOT / "generate.py"),
    run_script=str(PROVENANCE_ROOT / "run_all.sh"),
    generated_subdirs=("logs", "mesh", "anchor", "parameter", "restart"),
    notes=(
        "Imported from gv_simulation_files/buckling/gv/run_all.sh.",
        "Runtime entry points are run_all.sh, generate.py, parameters.py, and equil.py.",
    ),
)
DESCRIPTOR = RUNTIME_DESCRIPTOR


def build_dry_run_descriptor(
    output_root: str | Path,
    *,
    geometry: str = DEFAULT_GV_GEOMETRY.id,
    controls: Mapping[str, float] | None = None,
) -> RuntimeDryRun:
    return RUNTIME_DESCRIPTOR.plan(output_root=output_root, geometry=geometry, controls=controls)


def get_runtime_descriptor() -> RuntimeDescriptor:
    return RUNTIME_DESCRIPTOR


DRY_RUN_DESCRIPTOR = RUNTIME_DESCRIPTOR.plan()
