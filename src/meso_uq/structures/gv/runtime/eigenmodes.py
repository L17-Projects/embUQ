from __future__ import annotations

from pathlib import Path

from ..geometries import DEFAULT_GV_GEOMETRY
from .base import ControlSweep, DryRunCommand, RuntimeDescriptor, _find_repo_root

_REPO_ROOT = _find_repo_root()
_PROVENANCE_ROOT = (_REPO_ROOT / "gv" / "eigenmodes" / "src").resolve()
_ANALYSIS_ROOT = (_PROVENANCE_ROOT / "analysis").resolve()
_LEGACY_IMPORT_ROOT = (_REPO_ROOT / "gv_simulation_files" / "eigenmodes" / "gv").resolve()

_SOURCE_FILES = tuple(
    str(path)
    for path in (
        _PROVENANCE_ROOT / "run_all.sh",
        _PROVENANCE_ROOT / "generate.py",
        _PROVENANCE_ROOT / "parameters.py",
        Path("run.sh"),
        _PROVENANCE_ROOT / "equil.py",
        Path("parameters-default.gv.yaml"),
        Path("clean_all.sh"),
        Path("gas_vesicle/create_gv.py"),
        Path("gas_vesicle/parameters.yaml"),
        Path("gas_vesicle/run.sh"),
        Path("gas_vesicle/statistics.py"),
        _ANALYSIS_ROOT / "all.sh",
        _ANALYSIS_ROOT / "all_analysis.py",
        _ANALYSIS_ROOT / "combine.py",
        _ANALYSIS_ROOT / "initial.py",
        _ANALYSIS_ROOT / "plot_freq.py",
        _ANALYSIS_ROOT / "plot_modes.py",
        _ANALYSIS_ROOT / "run.sh",
        _ANALYSIS_ROOT / "run_an.sh",
        _ANALYSIS_ROOT / "run_eig.sh",
        _ANALYSIS_ROOT / "trim.sh",
        _ANALYSIS_ROOT / "trim_svd.sh",
    )
)


EIGENMODES_RUNTIME_DESCRIPTOR = RuntimeDescriptor(
    experiment="eigenmodes",
    provenance_root=str(_PROVENANCE_ROOT),
    legacy_import_root=str(_LEGACY_IMPORT_ROOT),
    source_files=_SOURCE_FILES,
    control_sweeps=(ControlSweep(name="bpress", start=-91.0, stop=-91.0, steps=1),),
    sweep_mode="forward",
    first_restart=True,
    generated_subdirs=(
        "logs",
        "mesh",
        "parameter",
        "restart",
        "stats",
        "trj_eq",
        "analysis/output",
    ),
    analysis_commands=(
        DryRunCommand(
            argv=("python3", "combine.py"),
            cwd="{work_dir}/analysis",
            description="Combine staged XYZ frames into a single trajectory file for eigenmode analysis.",
        ),
        DryRunCommand(
            argv=("python3", "initial.py", "--simnum", "00001"),
            cwd="{work_dir}/analysis",
            description="Write the reference geometry consumed by the eigenmode analysis scripts.",
        ),
        DryRunCommand(
            argv=("python3", "all_analysis.py", "--simnum", "00001"),
            cwd="{work_dir}/analysis",
            description="Run the MDAnalysis-based covariance and SVD eigenmode pipeline without importing analysis dependencies here.",
        ),
        DryRunCommand(
            argv=("bash", "trim_svd.sh", "30"),
            cwd="{work_dir}/analysis",
            description="Trim the staged eigenpairs to the default reported mode count.",
        ),
    ),
    generate_script="generate.py",
    notes=(
        "Imported from `run_all.sh`: `python3 generate.py -p bpress -91.0 -91.0 1 --object gv --forward --first`.",
        "Background pressure remains a runtime control and is excluded from calibrated GV parameters.",
        f"Default geometry provenance remains `{DEFAULT_GV_GEOMETRY.source}`.",
        "Analysis provenance is recorded through `source_files` and `analysis_commands`; no MDAnalysis or plotting dependency is imported at module load.",
    ),
)
DESCRIPTOR = EIGENMODES_RUNTIME_DESCRIPTOR


def get_runtime_descriptor() -> RuntimeDescriptor:
    return EIGENMODES_RUNTIME_DESCRIPTOR


def build_dry_run_manifest(**plan_kwargs: object) -> dict[str, object]:
    return EIGENMODES_RUNTIME_DESCRIPTOR.plan(**plan_kwargs).to_manifest()


__all__ = [
    "EIGENMODES_RUNTIME_DESCRIPTOR",
    "build_dry_run_manifest",
    "get_runtime_descriptor",
]
