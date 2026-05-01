from __future__ import annotations

from ..geometries import DEFAULT_GV_GEOMETRY
from . import ControlSweep, KnownIssue, RuntimeDescriptor, _find_repo_root


_REPO_ROOT = _find_repo_root()
_PROVENANCE_ROOT = (_REPO_ROOT / "gv" / "torsion" / "src").resolve()
_LEGACY_IMPORT_ROOT = (_REPO_ROOT / "gv_simulation_files" / "torsion" / "gv").resolve()

_SOURCE_FILES = tuple(
    str(filename)
    for filename in (
        "run_all.sh",
        "generate.py",
        "parameters.py",
        "run.sh",
        "equil.py",
        "parameters-default.gv.yaml",
        "clean_all.sh",
        "gas_vesicle/create_gv.py",
        "gas_vesicle/parameters.py",
        "gas_vesicle/parameters.yaml",
        "gas_vesicle/run.sh",
        "gas_vesicle/statistics.py",
        "gas_vesicle/add_to_off.py",
    )
)


TORSION_RUNTIME_DESCRIPTOR = RuntimeDescriptor(
    experiment="torsion",
    provenance_root=str(_PROVENANCE_ROOT),
    legacy_import_root=str(_LEGACY_IMPORT_ROOT),
    source_files=_SOURCE_FILES,
    control_sweeps=(ControlSweep(name="theta", start=0.01, stop=0.1, steps=10),),
    sweep_mode="forward",
    first_restart=False,
    generated_subdirs=(
        "logs",
        "mesh",
        "parameter",
        "restart",
        "stats",
        "trj_eq",
        "force",
        "anchor",
    ),
    known_issues=(
        KnownIssue(
            id="torsion-hardcoded-bpress",
            severity="medium",
            summary="The staged torsion runtime hard-codes a negative membrane pressure instead of reading a control.",
            evidence=str((_PROVENANCE_ROOT / "equil.py").resolve()),
        ),
    ),
    notes=(
        "Imported from `run_all.sh`: `python3 generate.py -p theta 0.01 0.1 10 --object gv --forward`.",
        "Theta remains a runtime control and is excluded from calibrated GV parameters.",
        f"Default geometry provenance remains `{DEFAULT_GV_GEOMETRY.source}`.",
        "The staged `run.sh` invokes `parameters.py` before `mpirun ... equil.py --vacuum`.",
    ),
)
DESCRIPTOR = TORSION_RUNTIME_DESCRIPTOR


def get_runtime_descriptor() -> RuntimeDescriptor:
    return TORSION_RUNTIME_DESCRIPTOR


def build_dry_run_manifest(**plan_kwargs: object) -> dict[str, object]:
    return TORSION_RUNTIME_DESCRIPTOR.plan(**plan_kwargs).to_manifest()


__all__ = [
    "TORSION_RUNTIME_DESCRIPTOR",
    "build_dry_run_manifest",
    "get_runtime_descriptor",
]
