from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from meso_uq.experiments import canonical_dataset_id
from meso_uq.structures.registry import ExperimentSpec, GeometrySpec, StructureSpec, get_structure


@dataclass(frozen=True)
class GVReferenceContext:
    structure: StructureSpec
    experiment: ExperimentSpec
    geometry: GeometrySpec
    controls: dict[str, float]
    control_id: str
    dataset_id: str
    runtime_manifest: dict[str, Any]


def resolve_gv_reference_context(
    *,
    runtime_manifest: Mapping[str, object] | None = None,
    experiment: str | None = None,
    geometry: str | None = None,
    controls: Mapping[str, float] | None = None,
) -> GVReferenceContext:
    manifest = dict(runtime_manifest or {})
    structure_name = str(manifest.get("structure", "gv"))
    structure = get_structure(structure_name)
    if structure.name != "gv":
        raise ValueError(
            f"GV reference helpers only support structure 'gv'; received '{structure_name}'."
        )

    experiment_name = str(experiment or manifest.get("experiment") or "")
    if not experiment_name:
        raise ValueError("GV reference helpers require an experiment name or runtime manifest.")
    experiment_spec = structure.get_experiment(experiment_name, include_experimental=True)

    geometry_id = str(geometry or manifest.get("geometry") or "")
    if not geometry_id:
        raise ValueError("GV reference helpers require a geometry id or runtime manifest.")
    geometry_spec = structure.get_geometry(geometry_id)

    raw_controls = dict(manifest.get("controls", {}))
    if controls is not None:
        raw_controls.update(controls)
    if not raw_controls:
        raise ValueError(
            f"GV reference helpers require controls for experiment '{experiment_name}'. "
            "Pass them explicitly or provide a runtime manifest with a controls section."
        )
    normalized_controls = _normalize_controls(
        experiment_name=experiment_name,
        allowed_controls=experiment_spec.control_names,
        controls=raw_controls,
    )

    control_id = str(manifest.get("control_id") or _control_identifier(normalized_controls))
    dataset_id = str(
        manifest.get("dataset_id")
        or canonical_dataset_id(structure.name, experiment_name, geometry_id, control_id)
    )
    return GVReferenceContext(
        structure=structure,
        experiment=experiment_spec,
        geometry=geometry_spec,
        controls=normalized_controls,
        control_id=control_id,
        dataset_id=dataset_id,
        runtime_manifest=manifest,
    )


def build_manifest_prefix(
    context: GVReferenceContext,
    *,
    reference_kind: str,
    reference_source: str,
    surrogate_backend: str = "dnn",
) -> dict[str, Any]:
    contract = context.structure.parameter_contract
    noise_model = contract.noise_model
    return {
        "structure": context.structure.name,
        "experiment": context.experiment.name,
        "geometry": context.geometry.id,
        "controls": dict(context.controls),
        "control_id": context.control_id,
        "dataset_id": context.dataset_id,
        "reference_kind": reference_kind,
        "reference_source": reference_source,
        "surrogate_backend": surrogate_backend,
        "observable_names": [observable.name for observable in context.experiment.observables],
        "geometry_parameters": dict(context.geometry.parameters),
        "calibrated_parameter_names": list(contract.calibrated_names),
        "noise_model": (
            {
                "kind": noise_model.kind,
                "parameter": noise_model.parameter,
                "description": noise_model.description,
            }
            if noise_model is not None
            else None
        ),
        "provenance": {
            "geometry_source": context.geometry.source,
            "runtime_provenance_root": context.runtime_manifest.get("provenance_root"),
            "source_files": list(context.runtime_manifest.get("source_files", [])),
            "commands": list(context.runtime_manifest.get("commands", [])),
            "analysis_commands": list(context.runtime_manifest.get("analysis_commands", [])),
        },
        "outputs": {
            "output_root": context.runtime_manifest.get("output_root"),
            "work_dir": context.runtime_manifest.get("work_dir"),
            "generated_subdirs": list(context.runtime_manifest.get("generated_subdirs", [])),
        },
    }


def ensure_existing_reference_path(path: str | Path, *, dataset_id: str, reference_kind: str) -> Path:
    reference_path = Path(path)
    if reference_path.exists():
        return reference_path
    raise FileNotFoundError(
        f"Missing {reference_kind} GV reference data for dataset '{dataset_id}': {reference_path}. "
        "Generate the DPD reference artifact first or pass a valid existing path."
    )


def _normalize_controls(
    *,
    experiment_name: str,
    allowed_controls: tuple[str, ...],
    controls: Mapping[str, object],
) -> dict[str, float]:
    unknown = sorted(set(controls) - set(allowed_controls))
    if unknown:
        raise ValueError(
            f"Unknown GV controls for experiment '{experiment_name}': {', '.join(unknown)}. "
            f"Expected only: {', '.join(allowed_controls)}."
        )
    missing = [name for name in allowed_controls if name not in controls]
    if missing:
        raise ValueError(
            f"Missing GV controls for experiment '{experiment_name}': {', '.join(missing)}. "
            "Pass the full control set or provide a runtime manifest produced by the GV dry-run planner."
        )
    return {name: float(controls[name]) for name in allowed_controls}


def _control_identifier(controls: Mapping[str, float]) -> str:
    parts = [f"{name}_{_format_float(value)}" for name, value in sorted(controls.items())]
    return "__".join(parts) or "default"


def _format_float(value: float) -> str:
    text = f"{float(value):.12g}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text
