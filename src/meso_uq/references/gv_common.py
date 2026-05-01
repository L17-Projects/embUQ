from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from meso_uq.experiments import canonical_dataset_id
from meso_uq.structures.registry import ExperimentSpec, GeometrySpec, StructureSpec, get_structure
from meso_uq.workflow_acceleration import GV_CALIBRATED_PARAMETER_ORDER

DEFAULT_SURROGATE_BACKEND = "dnn"
GV_REFERENCE_KINDS = ("synthetic", "dpd_generated")
_MANIFEST_SCHEMA_VERSION = 2
_GV_GEOMETRY_PARAMETER_NAMES = ("radius", "height")
_REQUIRED_DATASET_ID_COMPONENTS = 4
_GV_REFERENCE_MANIFEST_VERSION_V1 = 1
_GV_REFERENCE_MANIFEST_VERSION_V2 = 2
_SUPPORTED_GV_REFERENCE_MANIFEST_VERSIONS = (_GV_REFERENCE_MANIFEST_VERSION_V1, _GV_REFERENCE_MANIFEST_VERSION_V2)
_GV_REFERENCE_MANIFEST_REQUIRED_FIELDS: dict[int, tuple[str, ...]] = {
    _GV_REFERENCE_MANIFEST_VERSION_V1: (
        "manifest_schema_version",
        "structure",
        "experiment",
        "geometry",
        "controls",
        "control_id",
        "dataset_id",
        "reference_kind",
        "surrogate_backend",
        "observable_names",
        "calibrated_parameter_names",
        "nuisance_parameter_names",
        "noise_model",
        "provenance",
        "outputs",
    ),
    _GV_REFERENCE_MANIFEST_VERSION_V2: (
        "manifest_schema_version",
        "structure",
        "experiment",
        "geometry",
        "controls",
        "control_id",
        "dataset_id",
        "reference_kind",
        "surrogate_backend",
        "observable_names",
        "geometry_parameters",
        "calibrated_parameter_names",
        "nuisance_parameter_names",
        "noise_model",
        "provenance",
        "outputs",
        "geometry_spec",
        "observable_schema",
        "generation_seed",
    ),
}


@dataclass(frozen=True)
class GVReferenceContext:
    structure: StructureSpec
    experiment: ExperimentSpec
    geometry: GeometrySpec
    controls: dict[str, float]
    control_id: str
    dataset_id: str
    runtime_manifest: dict[str, Any]


@dataclass(frozen=True)
class GVReferenceRecord:
    context: GVReferenceContext
    reference_kind: str
    reference_source: str
    surrogate_backend: str
    observable: str
    points: tuple[float, ...]
    values: tuple[float, ...]
    nuisance_parameters: dict[str, float]
    metadata: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        manifest = build_manifest_prefix(
            self.context,
            reference_kind=self.reference_kind,
            reference_source=self.reference_source,
            surrogate_backend=self.surrogate_backend,
        )
        if self.metadata.get("synthetic_seed") is not None:
            manifest["generation_seed"] = int(self.metadata["synthetic_seed"])
        manifest.update(
            {
                "observable": self.observable,
                "points": list(self.points),
                "values": list(self.values),
                "nuisance_parameters": dict(self.nuisance_parameters),
                "metadata": _jsonable(self.metadata),
            }
        )
        return manifest


@dataclass(frozen=True)
class GVReferenceMaterialization:
    data_path: Path
    metadata_path: Path
    manifest: dict[str, Any]

    @property
    def dataset_path(self) -> Path:
        return self.data_path

    @property
    def manifest_path(self) -> Path:
        return self.metadata_path

    def to_manifest(self) -> dict[str, Any]:
        return dict(self.manifest)


def resolve_gv_reference_context(
    *,
    runtime_manifest: Mapping[str, object] | None = None,
    experiment: str | None = None,
    geometry: str | GeometrySpec | Mapping[str, object] | None = None,
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

    geometry_value = geometry or manifest.get("geometry")
    geometry_id = _geometry_id(geometry_value)
    if not geometry_id:
        raise ValueError("GV reference helpers require a geometry id or runtime manifest.")
    geometry_spec = _resolve_geometry_spec(
        structure,
        geometry_value,
        runtime_manifest=manifest,
    )

    raw_controls = dict(manifest.get("controls", {}))
    if controls is not None:
        raw_controls.update(controls)
    if not raw_controls:
        raise ValueError(
            f"GV reference helpers require controls for experiment '{experiment_name}'. "
            "Pass them explicitly or provide a runtime manifest with a controls section."
        )
    normalized_controls = normalize_controls(
        experiment_name=experiment_name,
        allowed_controls=experiment_spec.control_names,
        controls=raw_controls,
    )

    control_id = str(manifest.get("control_id") or control_identifier(normalized_controls))
    dataset_id = str(
        manifest.get("dataset_id")
        or canonical_dataset_id(structure.name, experiment_name, geometry_id, control_id)
    )
    if "dataset_id" in manifest:
        validate_gv_dataset_id(dataset_id, structure_name=structure.name)
    return GVReferenceContext(
        structure=structure,
        experiment=experiment_spec,
        geometry=geometry_spec,
        controls=normalized_controls,
        control_id=control_id,
        dataset_id=dataset_id,
        runtime_manifest=manifest,
    )


def _geometry_id(geometry: object) -> str:
    if isinstance(geometry, GeometrySpec):
        return geometry.id
    if isinstance(geometry, Mapping):
        return str(geometry.get("id", ""))
    return "" if geometry is None else str(geometry)


def _resolve_geometry_spec(
    structure: StructureSpec,
    geometry: object,
    *,
    runtime_manifest: Mapping[str, object],
) -> GeometrySpec:
    if isinstance(geometry, GeometrySpec):
        return geometry
    if isinstance(geometry, Mapping):
        return _geometry_spec_from_mapping(geometry)

    geometry_id = _geometry_id(geometry)
    try:
        return structure.get_geometry(geometry_id)
    except KeyError:
        raw_geometry_spec = runtime_manifest.get("geometry_spec")
        if isinstance(raw_geometry_spec, Mapping) and str(raw_geometry_spec.get("id")) == geometry_id:
            return _geometry_spec_from_mapping(raw_geometry_spec)
        raise


def _geometry_spec_from_mapping(payload: Mapping[str, object]) -> GeometrySpec:
    geometry_id = str(payload.get("id", ""))
    if not geometry_id:
        raise ValueError("GV geometry_spec must include an id.")
    raw_parameters = payload.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("GV geometry_spec parameters must be a mapping.")
    return GeometrySpec(
        id=geometry_id,
        label=str(payload.get("label") or geometry_id),
        shape=str(payload.get("shape") or "custom"),
        parameters={str(name): float(value) for name, value in raw_parameters.items()},
        source=str(payload.get("source") or "runtime_manifest"),
    )


def validate_gv_dataset_id(dataset_id: str, *, structure_name: str) -> None:
    if not isinstance(dataset_id, str):
        raise ValueError("GV dataset identifiers must be strings.")
    if not dataset_id.startswith(f"{structure_name}:"):
        raise ValueError(
            f"GV dataset_id must be structure-qualified with '{structure_name}': {dataset_id!r}."
        )
    parts = dataset_id.split(":")
    if len(parts) != _REQUIRED_DATASET_ID_COMPONENTS:
        raise ValueError(f"GV dataset_id must include 4 components: '{structure_name}:<experiment>:<geometry>:<controls>'.")


def _build_observable_schema(context: GVReferenceContext) -> list[dict[str, str]]:
    metadata_schema = context.structure.metadata.get("observable_schema")
    if isinstance(metadata_schema, Sequence):
        allowed = {observable.name for observable in context.experiment.observables}
        discovered = [
            {
                "name": str(item.get("name")),
                "description": str(item.get("description")),
                "units": str(item.get("units")),
            }
            for item in metadata_schema
            if isinstance(item, Mapping) and str(item.get("name")) in allowed
        ]
        if discovered:
            return discovered
    return [
        {"name": observable.name, "description": observable.description, "units": observable.units}
        for observable in context.experiment.observables
    ]


def validate_gv_reference_manifest(manifest: Mapping[str, Any], *, context: GVReferenceContext | None = None) -> None:
    manifest_schema_version = manifest.get("manifest_schema_version")
    if manifest_schema_version not in _SUPPORTED_GV_REFERENCE_MANIFEST_VERSIONS:
        raise ValueError(
            f"Unsupported GV reference manifest schema version {manifest_schema_version!r}. "
            f"Expected one of {_SUPPORTED_GV_REFERENCE_MANIFEST_VERSIONS}."
        )

    required = _GV_REFERENCE_MANIFEST_REQUIRED_FIELDS[manifest_schema_version]
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError("GV reference manifest is missing required schema fields: " + ", ".join(missing))

    structure = manifest["structure"]
    if str(structure) != "gv":
        raise ValueError("GV reference manifest must use structure='gv'.")

    reference_kind = str(manifest["reference_kind"])
    if reference_kind not in GV_REFERENCE_KINDS:
        raise ValueError(f"Unsupported GV reference kind in manifest: {reference_kind!r}.")

    surrogate_backend = str(manifest["surrogate_backend"])
    validate_reference_axes(surrogate_backend=surrogate_backend, reference_kind=reference_kind)

    controls = manifest["controls"]
    if not isinstance(controls, Mapping):
        raise ValueError("GV reference manifest controls must be a mapping.")

    dataset_id = str(manifest["dataset_id"])
    validate_gv_dataset_id(dataset_id, structure_name="gv")

    calibrated = manifest["calibrated_parameter_names"]
    if not isinstance(calibrated, Sequence):
        raise ValueError("GV reference manifest calibrated_parameter_names must be a sequence.")
    expected_calibrated = list(get_structure("gv").parameter_contract.calibrated_names)
    if list(calibrated) != expected_calibrated:
        raise ValueError("GV reference manifest calibrated_parameter_names must match the GV parameter contract.")

    if manifest_schema_version >= _MANIFEST_SCHEMA_VERSION:
        geometry_parameters = manifest["geometry_parameters"]
        if not isinstance(geometry_parameters, Mapping):
            raise ValueError("GV reference manifest geometry_parameters must be a mapping.")
        for axis in _GV_GEOMETRY_PARAMETER_NAMES:
            if axis not in geometry_parameters:
                raise ValueError(f"GV reference manifest geometry_parameters missing '{axis}'.")

    if manifest_schema_version >= _MANIFEST_SCHEMA_VERSION:
        observable_schema = manifest["observable_schema"]
        if not isinstance(observable_schema, Sequence):
            raise ValueError("GV reference manifest observable_schema must be a sequence.")
        if context is not None and len(observable_schema) != len(context.experiment.observables):
            raise ValueError("GV reference manifest observable_schema must match the experiment observables.")

    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("GV reference manifest provenance must be a mapping.")

    if manifest_schema_version >= _MANIFEST_SCHEMA_VERSION:
        geometry_spec = manifest["geometry_spec"]
        if not isinstance(geometry_spec, Mapping):
            raise ValueError("GV reference manifest geometry_spec must be a mapping.")
        geometry_spec_parameters = geometry_spec.get("parameters")
        if not isinstance(geometry_spec_parameters, Mapping):
            raise ValueError("GV reference manifest geometry_spec.parameters must be a mapping.")
        for axis in _GV_GEOMETRY_PARAMETER_NAMES:
            if axis not in geometry_spec_parameters:
                raise ValueError(f"GV reference manifest geometry_spec.parameters missing '{axis}'.")

    noise_model = manifest["noise_model"]
    if not isinstance(noise_model, Mapping):
        raise ValueError("GV reference manifest noise_model must be a mapping.")
    if noise_model.get("kind") != "multiplicative" or noise_model.get("parameter") != "sigma":
        raise ValueError("GV reference manifest noise_model must be multiplicative sigma.")

    if manifest_schema_version >= _MANIFEST_SCHEMA_VERSION:
        generation_seed = manifest["generation_seed"]
        if generation_seed is not None and not isinstance(generation_seed, (int, float)):
            raise ValueError("GV reference manifest generation_seed must be numeric when set.")


def build_manifest_prefix(
    context: GVReferenceContext,
    *,
    reference_kind: str,
    reference_source: str,
    surrogate_backend: str = DEFAULT_SURROGATE_BACKEND,
) -> dict[str, Any]:
    validate_reference_axes(
        surrogate_backend=surrogate_backend,
        reference_kind=reference_kind,
    )
    contract = context.structure.parameter_contract
    noise_model = contract.noise_model
    manifest: dict[str, Any] = {
        "manifest_schema_version": _MANIFEST_SCHEMA_VERSION,
        "structure": context.structure.name,
        "experiment": context.experiment.name,
        "geometry": context.geometry.id,
        "geometry_spec": {
            "id": context.geometry.id,
            "label": context.geometry.label,
            "shape": context.geometry.shape,
            "parameters": dict(context.geometry.parameters),
            "source": context.geometry.source,
        },
        "controls": dict(context.controls),
        "control_id": context.control_id,
        "dataset_id": context.dataset_id,
        "reference_kind": reference_kind,
        "reference_source": reference_source,
        "surrogate_backend": surrogate_backend,
        "observable_names": [observable.name for observable in context.experiment.observables],
        "observable_schema": _build_observable_schema(context),
        "geometry_parameters": dict(context.geometry.parameters),
        "calibrated_parameter_names": list(contract.calibrated_names),
        "nuisance_parameter_names": list(contract.nuisance_names),
        "generation_seed": None,
        "parameter_names": {
            "calibrated": list(contract.calibrated_names),
            "nuisance": list(contract.nuisance_names),
        },
        "controls_policy": "GV controls are design inputs and are excluded from calibrated variables.",
        "noise_model": None
        if noise_model is None
        else {
            "kind": noise_model.kind,
            "parameter": noise_model.parameter,
            "description": noise_model.description,
        },
        "provenance": {
            "geometry_source": context.geometry.source,
            "runtime_provenance_root": context.runtime_manifest.get("provenance_root"),
            "runtime_source_root": context.runtime_manifest.get("source_root")
            or context.runtime_manifest.get("provenance_root"),
            "runtime_legacy_import_root": context.runtime_manifest.get("legacy_import_root"),
            "runtime_source_manifest": context.runtime_manifest.get("source_manifest"),
            "source_files": list(context.runtime_manifest.get("source_files", [])),
            "commands": list(context.runtime_manifest.get("commands", [])),
            "analysis_commands": list(context.runtime_manifest.get("analysis_commands", [])),
            "runtime_package": context.runtime_manifest.get("runtime_package"),
        },
        "outputs": {
            "output_root": context.runtime_manifest.get("output_root"),
            "work_dir": context.runtime_manifest.get("work_dir"),
            "generated_subdirs": list(context.runtime_manifest.get("generated_subdirs", [])),
        },
    }
    if context.runtime_manifest:
        manifest["runtime_manifest"] = _jsonable(context.runtime_manifest)
    return manifest


def build_reference_record(
    *,
    context: GVReferenceContext,
    reference_kind: str,
    reference_source: str,
    observable: str,
    points: Sequence[float],
    values: Sequence[float],
    nuisance_parameters: Mapping[str, float],
    metadata: Mapping[str, object] | None = None,
    surrogate_backend: str = DEFAULT_SURROGATE_BACKEND,
) -> GVReferenceRecord:
    validate_reference_axes(
        surrogate_backend=surrogate_backend,
        reference_kind=reference_kind,
    )
    validated_points = normalize_series("points", points)
    validated_values = normalize_series("values", values)
    if len(validated_points) != len(validated_values):
        raise ValueError("points and values must have the same length.")
    validated_nuisance = normalize_nuisance_parameters(nuisance_parameters)
    return GVReferenceRecord(
        context=context,
        reference_kind=reference_kind,
        reference_source=reference_source,
        surrogate_backend=surrogate_backend,
        observable=str(observable),
        points=validated_points,
        values=validated_values,
        nuisance_parameters=validated_nuisance,
        metadata=_jsonable(dict(metadata or {})),
    )


def materialize_reference_records(
    records: Iterable[GVReferenceRecord],
    *,
    data_path: str | Path,
    metadata_path: str | Path | None = None,
    collection_id: str | None = None,
    experiments: Sequence[str] | None = None,
    geometries: Sequence[str] | None = None,
    dataset_ids: Sequence[str] | None = None,
) -> GVReferenceMaterialization:
    selected = list(
        select_reference_records(
            records,
            experiments=experiments,
            geometries=geometries,
            dataset_ids=dataset_ids,
        )
    )
    if not selected:
        raise ValueError("No GV reference records remain after applying the requested selection.")

    selected.sort(
        key=lambda record: (
            record.context.experiment.name,
            record.context.geometry.id,
            record.context.control_id,
            record.observable,
        )
    )
    _validate_record_collection(selected)

    resolved_data_path = Path(data_path).resolve()
    resolved_metadata_path = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else resolved_data_path.with_suffix(".json")
    )
    resolved_data_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, np.ndarray] = {
        "dataset_ids": np.asarray([record.context.dataset_id for record in selected], dtype=str),
        "experiments": np.asarray([record.context.experiment.name for record in selected], dtype=str),
        "geometries": np.asarray([record.context.geometry.id for record in selected], dtype=str),
        "observables": np.asarray([record.observable for record in selected], dtype=str),
        "point_counts": np.asarray([len(record.points) for record in selected], dtype=np.int64),
    }
    entries: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        key = _record_array_key(record, index=index)
        npz_payload[f"{key}__points"] = np.asarray(record.points, dtype=np.float64)
        npz_payload[f"{key}__values"] = np.asarray(record.values, dtype=np.float64)
        entries.append(
            {
                "dataset_id": record.context.dataset_id,
                "experiment": record.context.experiment.name,
                "geometry": record.context.geometry.id,
                "observable": record.observable,
                "control_id": record.context.control_id,
                "controls": dict(record.context.controls),
                "nuisance_parameters": dict(record.nuisance_parameters),
                "array_key": key,
                "manifest": record.to_manifest(),
            }
        )
    np.savez_compressed(resolved_data_path, **npz_payload)

    root_manifest = selected[0].to_manifest()
    manifest = {
        "manifest_schema_version": _MANIFEST_SCHEMA_VERSION,
        "dataset_id": collection_id or root_manifest["dataset_id"],
        "structure": "gv",
        "reference_kind": root_manifest["reference_kind"],
        "surrogate_backend": root_manifest["surrogate_backend"],
        "noise_model": {
            "kind": "multiplicative",
            "parameter": "sigma",
        },
        "parameter_names": root_manifest["parameter_names"],
        "controls_policy": root_manifest["controls_policy"],
        "entry_count": len(entries),
        "selection": {
            "experiments": list(experiments or []),
            "geometries": list(geometries or []),
            "dataset_ids": list(dataset_ids or []),
        },
        "artifacts": {
            "reference_dataset": str(resolved_data_path),
            "reference_manifest": str(resolved_metadata_path),
        },
        "entries": entries,
    }
    resolved_metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return GVReferenceMaterialization(
        data_path=resolved_data_path,
        metadata_path=resolved_metadata_path,
        manifest=manifest,
    )


def select_reference_records(
    records: Iterable[GVReferenceRecord],
    *,
    experiments: Sequence[str] | None = None,
    geometries: Sequence[str] | None = None,
    dataset_ids: Sequence[str] | None = None,
) -> Iterable[GVReferenceRecord]:
    experiment_filter = set(experiments or ())
    geometry_filter = set(geometries or ())
    dataset_filter = set(dataset_ids or ())
    for record in records:
        if experiment_filter and record.context.experiment.name not in experiment_filter:
            continue
        if geometry_filter and record.context.geometry.id not in geometry_filter:
            continue
        if dataset_filter and record.context.dataset_id not in dataset_filter:
            continue
        yield record


def ensure_existing_reference_path(
    path: str | Path,
    *,
    dataset_id: str,
    reference_kind: str,
) -> Path:
    reference_path = Path(path)
    if reference_path.exists():
        return reference_path
    raise FileNotFoundError(
        f"Missing {reference_kind} GV reference data for dataset '{dataset_id}': {reference_path}. "
        "Generate the DPD reference artifact first or pass a valid existing path."
    )


def normalize_controls(
    *,
    experiment_name: str,
    allowed_controls: Sequence[str],
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
    overlap = sorted(set(controls).intersection(GV_CALIBRATED_PARAMETER_ORDER))
    if overlap:
        raise ValueError(
            "GV controls must remain separate from calibrated parameters: "
            + ", ".join(overlap)
        )
    return {name: float(controls[name]) for name in allowed_controls}


def normalize_series(name: str, values: Sequence[float]) -> tuple[float, ...]:
    series = tuple(float(value) for value in values)
    if not series:
        raise ValueError(f"{name} must contain at least one value.")
    return series


def normalize_nuisance_parameters(
    nuisance_parameters: Mapping[str, float],
) -> dict[str, float]:
    normalized = {str(name): float(value) for name, value in nuisance_parameters.items()}
    if "sigma" not in normalized:
        raise ValueError("GV reference records require a sigma nuisance parameter.")
    overlap = sorted(set(normalized).intersection(GV_CALIBRATED_PARAMETER_ORDER))
    if overlap:
        raise ValueError(
            "GV nuisance parameters must remain separate from calibrated parameters: "
            + ", ".join(overlap)
        )
    return normalized


def validate_reference_axes(*, surrogate_backend: str, reference_kind: str) -> None:
    if surrogate_backend != DEFAULT_SURROGATE_BACKEND:
        raise ValueError("GV reference helpers support only the 'dnn' surrogate backend in this tranche.")
    if reference_kind not in GV_REFERENCE_KINDS:
        raise ValueError(
            f"Unsupported GV reference kind '{reference_kind}'. Expected one of {GV_REFERENCE_KINDS}."
        )


def control_identifier(controls: Mapping[str, float]) -> str:
    parts = [f"{name}_{format_float(value)}" for name, value in sorted(controls.items())]
    return "__".join(parts) or "default"


def format_float(value: float) -> str:
    text = f"{float(value):.12g}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _validate_record_collection(records: Sequence[GVReferenceRecord]) -> None:
    kinds = {record.reference_kind for record in records}
    if len(kinds) != 1:
        raise ValueError(f"Mixed GV reference kinds are not supported in one artifact: {sorted(kinds)}.")
    backends = {record.surrogate_backend for record in records}
    if len(backends) != 1:
        raise ValueError(f"Mixed surrogate backends are not supported in one artifact: {sorted(backends)}.")
    structures = {record.context.structure.name for record in records}
    if structures != {"gv"}:
        raise ValueError(f"Mixed structures are not supported in one GV artifact: {sorted(structures)}.")


def _record_array_key(record: GVReferenceRecord, *, index: int) -> str:
    token = (
        f"{index:03d}_{record.context.experiment.name}_{record.context.geometry.id}_"
        f"{record.context.control_id}_{record.observable}"
    )
    return re.sub(r"[^A-Za-z0-9_]+", "_", token).strip("_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
