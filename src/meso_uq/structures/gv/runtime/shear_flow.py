from __future__ import annotations

from pathlib import Path
from typing import Any

from ..controls import EXPERIMENT_CONTROLS
from ..geometries import DEFAULT_GV_GEOMETRY
from ..parameters import GV_PARAMETER_CONTRACT
from . import ControlSweep, KnownIssue, RuntimeDescriptor


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ROOT = REPO_ROOT / "gv_simulation_files" / "shear_flow"
PROVENANCE_ROOT = SOURCE_ROOT / "a0"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_runs" / "gv" / "shear_flow"

STRUCTURE_NAME = "gv"
EXPERIMENT_NAME = "shear_flow"
RUNTIME_MODULE_NAME = "mirheoOBMD"
KNOWN_ISSUE_ID = "bouncer_collision_candidates_coarse"
KNOWN_ISSUE_SUMMARY = (
    "Found too many triangle collision candidates (coarse) (1481863, max 14020) "
    "in bouncer 'membrane_bounce'."
)
KNOWN_ISSUE_EVIDENCE = "gv_simulation_files/shear_flow/a0/output.out:841"


SHEAR_FLOW_RUNTIME = RuntimeDescriptor(
    experiment=EXPERIMENT_NAME,
    provenance_root=str(PROVENANCE_ROOT),
    source_files=(
        str(SOURCE_ROOT / "README.md"),
        str(SOURCE_ROOT / "run_all.py"),
        str(SOURCE_ROOT / "copy_and_modify.py"),
        str(PROVENANCE_ROOT / "commands.txt"),
        str(PROVENANCE_ROOT / "run_all_HPC.sh"),
        str(PROVENANCE_ROOT / "run_HPC.sbatch"),
        str(PROVENANCE_ROOT / "generate.py"),
        str(PROVENANCE_ROOT / "parameters.py"),
        str(PROVENANCE_ROOT / "run.sh"),
        str(PROVENANCE_ROOT / "equil.py"),
        str(PROVENANCE_ROOT / "parameters-default.gv.yaml"),
        str(PROVENANCE_ROOT / "output.out"),
    ),
    control_sweeps=(
        ControlSweep("ptan", start=0.4, stop=0.4, steps=1),
        ControlSweep("afsi", start=0.0, stop=0.0, steps=1),
        ControlSweep("bpress", start=-91.0, stop=-91.0, steps=1),
    ),
    sweep_mode="parallel",
    first_restart=True,
    runtime_package=RUNTIME_MODULE_NAME,
    generate_script=str(PROVENANCE_ROOT / "generate.py"),
    execute_argv=("sbatch", "run_HPC.sbatch"),
    execute_description="Submit the generated GV shear-flow job script to Slurm.",
    experimental=True,
    known_issues=(
        KnownIssue(
            id=KNOWN_ISSUE_ID,
            severity="error",
            summary=KNOWN_ISSUE_SUMMARY,
            evidence=KNOWN_ISSUE_EVIDENCE,
        ),
    ),
    notes=(
        "Imported generator defaults to --parallel --first with a single ptan lane.",
        "The imported run_all_HPC.sh passes -g 1 -N 1 to generate.py.",
        "afsi and bpress are controls for runtime planning and are excluded from calibrated parameters.",
    ),
)
DESCRIPTOR = SHEAR_FLOW_RUNTIME


def build_descriptor() -> RuntimeDescriptor:
    return SHEAR_FLOW_RUNTIME


def build_dry_run(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_root: str | Path | None = None,
    controls: dict[str, float] | None = None,
    include_experimental: bool = False,
) -> Any:
    resolved_output_root = run_root if run_root is not None else output_root
    return SHEAR_FLOW_RUNTIME.plan(
        output_root=resolved_output_root,
        geometry=DEFAULT_GV_GEOMETRY.id,
        controls=controls,
        include_experimental=include_experimental,
    )


def build_dry_run_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_root: str | Path | None = None,
    controls: dict[str, float] | None = None,
    include_experimental: bool = False,
) -> dict[str, Any]:
    dry_run = build_dry_run(
        output_root=output_root,
        run_root=run_root,
        controls=controls,
        include_experimental=include_experimental,
    )
    manifest = dry_run.to_manifest()
    manifest.update(
        {
            "descriptor_version": 1,
            "geometry_spec": {
                "id": DEFAULT_GV_GEOMETRY.id,
                "label": DEFAULT_GV_GEOMETRY.label,
                "shape": DEFAULT_GV_GEOMETRY.shape,
                "parameters": dict(DEFAULT_GV_GEOMETRY.parameters),
                "source": DEFAULT_GV_GEOMETRY.source,
            },
            "control_metadata": {
                control.name: {
                    "description": control.description,
                    "units": control.units,
                    "default": manifest["controls"][control.name],
                    "role": "control",
                    "is_calibrated_parameter": False,
                }
                for control in EXPERIMENT_CONTROLS[EXPERIMENT_NAME]
            },
            "parameter_contract": {
                "calibrated": list(GV_PARAMETER_CONTRACT.calibrated_names),
                "nuisance": list(GV_PARAMETER_CONTRACT.nuisance_names),
            },
            "identity_axes": {
                "structure": manifest["structure"],
                "experiment": manifest["experiment"],
                "geometry": manifest["geometry"],
                "controls": list(manifest["controls"]),
            },
            "default_sweep": {
                "mode": manifest["sweep_mode"],
                "first_restart": manifest["first_restart"],
                "resource_defaults": {"gpus": 1, "nodes": 1},
                "controls": {
                    "ptan": {
                        "kind": "sweep",
                        "start": 0.4,
                        "stop": 0.4,
                        "steps": 1,
                        "values": [0.4],
                    },
                    "afsi": {"kind": "fixed", "value": 0.0},
                    "bpress": {"kind": "fixed", "value": -91.0},
                },
            },
            "runtime_requirements": [
                {
                    "module": manifest["runtime_package"],
                    "purpose": "OBMD-enabled GV shear-flow execution runtime",
                    "required_at_execution": True,
                    "imported_on_module_load": False,
                }
            ],
            "paths": {
                "source_root": str(SOURCE_ROOT),
                "provenance_root": str(PROVENANCE_ROOT),
                "output_root": manifest["output_root"],
                "work_dir": manifest["work_dir"],
            },
            "artifacts_produced": False,
            "generated_artifacts": [],
        }
    )
    return manifest


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "EXPERIMENT_NAME",
    "KNOWN_ISSUE_EVIDENCE",
    "KNOWN_ISSUE_ID",
    "KNOWN_ISSUE_SUMMARY",
    "PROVENANCE_ROOT",
    "RUNTIME_MODULE_NAME",
    "SHEAR_FLOW_RUNTIME",
    "SOURCE_ROOT",
    "STRUCTURE_NAME",
    "DESCRIPTOR",
    "build_descriptor",
    "build_dry_run",
    "build_dry_run_manifest",
]
