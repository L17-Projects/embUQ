from __future__ import annotations

"""Schema and path helpers for GV numerical aggregate HDF5 sweep datasets."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from meso_uq.config.models import EMB_EXPERIMENTS, GV_EXPERIMENTS
from meso_uq.experiments import canonical_dataset_id
from meso_uq.references.gv_common import control_identifier, normalize_controls
from meso_uq.structures import get_structure
from meso_uq.structures.gv import build_geometry

from .material_parameters import GV_ZERO_ALLOWED_MATERIAL_PARAMETERS


GV_NUMERICAL_MANIFEST_SCHEMA_VERSION = 1
GV_NUMERICAL_POSTPROCESSOR_VERSION = "0.1.0"

GV_NUMERICAL_DATA_ROOT = Path("_runs") / "gv" / "numerical_data"
GV_NUMERICAL_DATASET_DIRNAME = "datasets"
GV_NUMERICAL_DATASET_MANIFEST_FILENAME = "numerical_dataset_manifest.json"
GV_NUMERICAL_DATASET_HDF5_FILENAME = "numerical_dataset.h5"
_GV_GEOMETRY_PARAMETER_NAMES = ("radius", "height")


@dataclass(frozen=True)
class GVNumericalUnits:
    material_parameters: Mapping[str, str]
    controls: Mapping[str, str]

    def to_manifest(self) -> dict[str, Mapping[str, str]]:
        return {
            "material_parameters": dict(self.material_parameters),
            "controls": dict(self.controls),
        }


@dataclass(frozen=True)
class GVNumericalNormalization:
    scales: Mapping[str, float]
    offsets: Mapping[str, float]

    def to_manifest(self) -> dict[str, dict[str, float]]:
        return {
            "scales": {name: float(value) for name, value in self.scales.items()},
            "offsets": {name: float(value) for name, value in self.offsets.items()},
        }


@dataclass(frozen=True)
class GVNumericalQualityFlags:
    finite_observables: bool
    finite_observable_ratio: float
    canary_failures: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "finite_observables": self.finite_observables,
            "finite_observable_ratio": float(self.finite_observable_ratio),
            "canary_failures": list(self.canary_failures),
        }


@dataclass(frozen=True)
class GVNumericalDatasetManifest:
    manifest_schema_version: int
    postprocessor_version: str
    structure: str
    experiment: str
    campaign_id: str
    geometry: str
    geometry_parameters: Mapping[str, float]
    control_id: str
    controls: Mapping[str, float]
    material_parameters: Mapping[str, float]
    units: GVNumericalUnits
    normalization: GVNumericalNormalization
    quality_flags: GVNumericalQualityFlags
    raw_provenance: Mapping[str, Any]
    dataset_id: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "postprocessor_version": self.postprocessor_version,
            "structure": self.structure,
            "experiment": self.experiment,
            "campaign_id": self.campaign_id,
            "geometry": self.geometry,
            "geometry_parameters": dict(self.geometry_parameters),
            "control_id": self.control_id,
            "controls": dict(self.controls),
            "material_parameters": dict(self.material_parameters),
            "units": self.units.to_manifest(),
            "normalization": self.normalization.to_manifest(),
            "quality_flags": self.quality_flags.to_manifest(),
            "raw_provenance": _jsonable(self.raw_provenance),
            "dataset_id": self.dataset_id,
        }


def _normalize_float(value: object, context: str) -> float:
    try:
        value_as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric.") from exc
    if not isfinite(value_as_float):
        raise ValueError(f"{context} must be finite.")
    return value_as_float


def _validate_campaign_id(campaign_id: str) -> None:
    if not campaign_id:
        raise ValueError("campaign_id must be a non-empty identifier.")
    if "/" in campaign_id or "\\" in campaign_id:
        raise ValueError(f"campaign_id must not contain path separators: {campaign_id!r}.")
    if ".." in campaign_id:
        raise ValueError("campaign_id must not contain path traversal segments.")


def _validate_structure(structure: str) -> str:
    if structure != "gv":
        raise ValueError(f"GV numerical data is only defined for structure 'gv'; got {structure!r}.")
    return structure


def _validate_experiment(structure: str, experiment: str) -> str:
    if not experiment:
        raise ValueError("experiment must be a non-empty string.")
    if experiment in EMB_EXPERIMENTS and experiment not in GV_EXPERIMENTS:
        raise ValueError(f"Experiment {experiment!r} belongs to structure 'emb'.")
    if experiment in EMB_EXPERIMENTS and experiment in GV_EXPERIMENTS:
        raise ValueError(f"Experiment {experiment!r} is ambiguous across structures.")
    try:
        return get_structure(structure).get_experiment(experiment, include_experimental=True).name
    except KeyError as exc:
        raise ValueError(f"Unknown experiment {experiment!r} for structure {structure!r}.") from exc


def _validate_geometry(*, radius: float, height: float) -> tuple[str, dict[str, float]]:
    if radius <= 0:
        raise ValueError(f"GV geometry radius must be positive, got {radius}.")
    if height <= 0:
        raise ValueError(f"GV geometry height must be positive, got {height}.")
    geometry = build_geometry(radius=radius, height=height, source="numerical_data_schema")
    parameters = {str(name): float(value) for name, value in geometry.parameters.items()}
    missing = sorted(set(_GV_GEOMETRY_PARAMETER_NAMES) - set(parameters))
    if missing:
        raise ValueError(f"GV geometry must include {', '.join(missing)}.")
    return geometry.id, parameters


def _validate_material_parameters(
    structure: str,
    material_parameters: Mapping[str, object],
) -> dict[str, float]:
    calibrated = tuple(get_structure(structure).parameter_contract.calibrated_names)
    normalized: dict[str, float] = {
        str(name): _normalize_float(value, f"material_parameters[{name}]") for name, value in material_parameters.items()
    }
    supplied = set(normalized)
    required = set(calibrated)
    missing = sorted(required - supplied)
    extra = sorted(supplied - required)
    if missing:
        raise ValueError(
            f"GV numerical manifest requires all calibrated material parameters for structure {structure!r}; "
            f"missing: {', '.join(missing)}."
        )
    if extra:
        raise ValueError(f"GV numerical manifest contains unexpected material parameters: {', '.join(extra)}.")
    for name in calibrated:
        if name in GV_ZERO_ALLOWED_MATERIAL_PARAMETERS:
            if normalized[name] < 0.0:
                raise ValueError(f"GV material parameter '{name}' must be non-negative.")
            continue
        if normalized[name] <= 0.0:
            raise ValueError(f"GV material parameter '{name}' must be positive.")
    return normalized


def _normalize_units(
    *,
    experiment: str,
    normalized_controls: Mapping[str, float],
    material_parameters: Mapping[str, float],
    units: Mapping[str, Mapping[str, str]] | None = None,
) -> GVNumericalUnits:
    provided = units or {}
    if not isinstance(provided, Mapping):
        raise ValueError("units must be a mapping.")
    provided_controls = provided.get("controls", {})
    provided_material = provided.get("material_parameters", {})
    if not isinstance(provided_controls, Mapping):
        raise ValueError("units.controls must be a mapping.")
    if not isinstance(provided_material, Mapping):
        raise ValueError("units.material_parameters must be a mapping.")

    for key, unit in provided_controls.items():
        if str(key) not in normalized_controls:
            raise ValueError(f"GV numerical units contains unexpected control {key!r}.")
        if not isinstance(unit, str):
            raise ValueError(f"GV numerical control unit for {key!r} must be a string.")
    for key, unit in provided_material.items():
        if str(key) not in material_parameters:
            raise ValueError(f"GV numerical units contains unexpected material parameter {key!r}.")
        if not isinstance(unit, str):
            raise ValueError(f"GV numerical material unit for {key!r} must be a string.")

    experiment_spec = get_structure("gv").get_experiment(experiment, include_experimental=True)
    control_unit_defaults = {control.name: control.units for control in experiment_spec.controls}
    material_unit_defaults = {
        name: get_structure("gv").parameter_contract.get_parameter(name).units for name in material_parameters
    }
    return GVNumericalUnits(
        material_parameters={
            name: str(provided_material.get(name, material_unit_defaults[name])) for name in material_parameters
        },
        controls={
            name: str(provided_controls.get(name, control_unit_defaults[name])) for name in normalized_controls
        },
    )


def _normalize_float_map(payload: Mapping[str, object], context: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    normalized: dict[str, float] = {}
    for name, value in payload.items():
        normalized[str(name)] = _normalize_float(value, f"{context}[{name!r}]")
    return normalized


def _normalize_normalization(
    material_parameters: Mapping[str, float],
    controls: Mapping[str, float],
    normalization: Mapping[str, Mapping[str, float]] | None = None,
) -> GVNumericalNormalization:
    axes = tuple((*material_parameters.keys(), *controls.keys()))
    expected = set(axes)
    scales = _normalize_float_map((normalization or {}).get("scales", {}), "normalization.scales")
    offsets = _normalize_float_map((normalization or {}).get("offsets", {}), "normalization.offsets")
    extra = set(scales) - expected
    if extra:
        raise ValueError(
            "GV numerical normalization.scales contains unexpected keys: "
            f"{', '.join(sorted(extra))}."
        )
    extra = set(offsets) - expected
    if extra:
        raise ValueError(
            "GV numerical normalization.offsets contains unexpected keys: "
            f"{', '.join(sorted(extra))}."
        )
    base_scales = {name: 1.0 for name in axes}
    base_offsets = {name: 0.0 for name in axes}
    base_scales.update(scales)
    base_offsets.update(offsets)
    return GVNumericalNormalization(scales=base_scales, offsets=base_offsets)


def _normalize_quality_flags(flags: Mapping[str, Any] | None) -> GVNumericalQualityFlags:
    if flags is None:
        return GVNumericalQualityFlags(
            finite_observables=True,
            finite_observable_ratio=1.0,
            canary_failures=(),
        )
    if not isinstance(flags, Mapping):
        raise ValueError("quality_flags must be a mapping.")
    finite = flags.get("finite_observables")
    if finite is None:
        raise ValueError("quality_flags must include 'finite_observables'.")
    if not isinstance(finite, bool):
        raise ValueError("quality_flags.finite_observables must be boolean.")
    finite_observable_ratio = _normalize_float(
        flags.get("finite_observable_ratio", 1.0),
        "quality_flags.finite_observable_ratio",
    )
    if finite_observable_ratio < 0.0 or finite_observable_ratio > 1.0:
        raise ValueError("quality_flags.finite_observable_ratio must be between 0 and 1.")
    canary_failures_raw = flags.get("canary_failures", ())
    if isinstance(canary_failures_raw, str):
        canary_failures = (canary_failures_raw,)
    else:
        try:
            canary_failures = tuple(str(item) for item in canary_failures_raw)
        except TypeError as exc:
            raise ValueError("quality_flags.canary_failures must be a sequence of strings.") from exc
    return GVNumericalQualityFlags(
        finite_observables=bool(finite),
        finite_observable_ratio=finite_observable_ratio,
        canary_failures=canary_failures,
    )


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


def build_gv_numerical_dataset_manifest(
    *,
    campaign_id: str,
    structure: str,
    experiment: str,
    geometry_radius: float,
    geometry_height: float,
    material_parameters: Mapping[str, object],
    controls: Mapping[str, object],
    raw_provenance: Mapping[str, Any],
    quality_flags: Mapping[str, Any] | None = None,
    units: Mapping[str, Mapping[str, str]] | None = None,
    normalization: Mapping[str, Mapping[str, float]] | None = None,
    manifest_schema_version: int = GV_NUMERICAL_MANIFEST_SCHEMA_VERSION,
    postprocessor_version: str = GV_NUMERICAL_POSTPROCESSOR_VERSION,
) -> GVNumericalDatasetManifest:
    _validate_campaign_id(campaign_id)
    structure = _validate_structure(structure)
    experiment = _validate_experiment(structure, experiment)
    geometry_radius = _normalize_float(geometry_radius, "geometry_radius")
    geometry_height = _normalize_float(geometry_height, "geometry_height")
    geometry, geometry_parameters = _validate_geometry(radius=geometry_radius, height=geometry_height)
    experiment_spec = get_structure("gv").get_experiment(experiment, include_experimental=True)
    normalized_controls = normalize_controls(
        experiment_name=experiment,
        allowed_controls=experiment_spec.control_names,
        controls=controls,
    )
    control_id = control_identifier(normalized_controls)
    normalized_material = _validate_material_parameters(structure, material_parameters)
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("raw_provenance must be a mapping.")
    normalized_provenance = _jsonable(raw_provenance)
    normalized_units = _normalize_units(
        experiment=experiment,
        normalized_controls=normalized_controls,
        material_parameters=normalized_material,
        units=units,
    )
    normalized_normalization = _normalize_normalization(
        material_parameters=normalized_material,
        controls=normalized_controls,
        normalization=normalization,
    )
    normalized_flags = _normalize_quality_flags(quality_flags)
    dataset_id = canonical_dataset_id(structure, experiment, geometry, control_id)
    return GVNumericalDatasetManifest(
        manifest_schema_version=int(manifest_schema_version),
        postprocessor_version=str(postprocessor_version),
        structure=structure,
        experiment=experiment,
        campaign_id=campaign_id,
        geometry=geometry,
        geometry_parameters=geometry_parameters,
        control_id=control_id,
        controls=normalized_controls,
        material_parameters=normalized_material,
        units=normalized_units,
        normalization=normalized_normalization,
        quality_flags=normalized_flags,
        raw_provenance=normalized_provenance,
        dataset_id=dataset_id,
    )


def parse_gv_dataset_id(dataset_id: str) -> tuple[str, str, str, str]:
    if not isinstance(dataset_id, str):
        raise ValueError("GV dataset_id must be a string.")
    parts = dataset_id.split(":")
    if len(parts) != 4:
        raise ValueError("GV dataset_id must have exactly 4 components: structure:experiment:geometry:controls.")
    return parts[0], parts[1], parts[2], parts[3]


def _validate_dataset_path_token(component: str, *, field_name: str) -> None:
    if not component:
        raise ValueError(f"GV dataset_id must include a non-empty {field_name} component.")
    if "/" in component or "\\" in component:
        raise ValueError(f"GV dataset_id {field_name} component must not contain path separators.")
    if ".." in component:
        raise ValueError(f"GV dataset_id {field_name} component must not contain path traversal segments.")


def validate_gv_numerical_dataset_id(dataset_id: str, *, structure: str) -> None:
    _validate_structure(structure)
    dataset_structure, experiment, geometry, control_id = parse_gv_dataset_id(dataset_id)
    if dataset_structure != structure:
        raise ValueError(
            f"GV dataset_id structure mismatch: expected {structure!r}, got {dataset_structure!r}."
        )
    _validate_experiment(structure, experiment)
    _validate_dataset_path_token(geometry, field_name="geometry")
    _validate_dataset_path_token(control_id, field_name="control")


def validate_gv_numerical_manifest(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("GV numerical manifest must be a mapping.")
    required = (
        "manifest_schema_version",
        "postprocessor_version",
        "structure",
        "experiment",
        "campaign_id",
        "geometry",
        "geometry_parameters",
        "control_id",
        "controls",
        "material_parameters",
        "units",
        "normalization",
        "quality_flags",
        "raw_provenance",
        "dataset_id",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"GV numerical manifest missing required fields: {', '.join(missing)}.")
    if not isinstance(payload["manifest_schema_version"], int):
        raise ValueError("GV numerical manifest manifest_schema_version must be an integer.")
    if payload["manifest_schema_version"] != GV_NUMERICAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"GV numerical manifest schema version mismatch: {payload['manifest_schema_version']!r}; "
            f"expected {GV_NUMERICAL_MANIFEST_SCHEMA_VERSION}."
        )
    if payload["postprocessor_version"] != GV_NUMERICAL_POSTPROCESSOR_VERSION:
        raise ValueError("GV numerical manifest postprocessor_version is unsupported.")
    _validate_campaign_id(str(payload["campaign_id"]))
    structure = _validate_structure(str(payload["structure"]))
    experiment = _validate_experiment(structure, str(payload["experiment"]))
    validate_gv_numerical_dataset_id(str(payload["dataset_id"]), structure=structure)

    if not isinstance(payload["geometry_parameters"], Mapping):
        raise ValueError("GV numerical manifest geometry_parameters must be a mapping.")
    for axis in _GV_GEOMETRY_PARAMETER_NAMES:
        if axis not in payload["geometry_parameters"]:
            raise ValueError(f"GV numerical geometry_parameters must include {axis!r}.")
    manifest_geometry, geometry_parameters = _validate_geometry(
        radius=_normalize_float(payload["geometry_parameters"]["radius"], "geometry_parameters.radius"),
        height=_normalize_float(payload["geometry_parameters"]["height"], "geometry_parameters.height"),
    )
    if str(payload["geometry"]) != manifest_geometry:
        raise ValueError(
            f"GV numerical manifest geometry does not match geometry_parameters: "
            f"{payload['geometry']!r} != {manifest_geometry!r}."
        )
    geometry_parameters_payload = {str(name): _normalize_float(value, f"geometry_parameters[{name}]")
                                 for name, value in payload["geometry_parameters"].items()}
    if set(geometry_parameters_payload) != set(_GV_GEOMETRY_PARAMETER_NAMES):
        raise ValueError("GV numerical geometry_parameters must include exactly radius and height.")
    if geometry_parameters != geometry_parameters_payload:
        raise ValueError("GV numerical geometry_parameters should include only numeric radius and height values.")

    if not isinstance(payload["controls"], Mapping):
        raise ValueError("GV numerical manifest controls must be a mapping.")
    normalized_controls = normalize_controls(
        experiment_name=experiment,
        allowed_controls=get_structure(structure).get_experiment(experiment, include_experimental=True).control_names,
        controls=payload["controls"],
    )
    control_id = str(payload["control_id"])
    if control_id != control_identifier(normalized_controls):
        raise ValueError(
            "GV numerical manifest control_id must match normalized controls."
        )
    normalized_material = _validate_material_parameters(structure, payload["material_parameters"])
    _normalize_units(
        experiment=experiment,
        normalized_controls=normalized_controls,
        material_parameters=normalized_material,
        units=payload["units"],
    )
    _normalize_normalization(
        material_parameters=normalized_material,
        controls=normalized_controls,
        normalization=payload["normalization"],
    )
    _normalize_quality_flags(payload["quality_flags"])
    if not isinstance(payload["raw_provenance"], Mapping):
        raise ValueError("GV numerical manifest raw_provenance must be a mapping.")
    try:
        json.dumps(_jsonable(payload["raw_provenance"]))
    except TypeError as exc:
        raise ValueError("GV numerical manifest raw_provenance must be JSON-serializable.") from exc


def campaign_dataset_root(*, campaign_id: str) -> Path:
    _validate_campaign_id(campaign_id)
    return GV_NUMERICAL_DATA_ROOT / campaign_id / GV_NUMERICAL_DATASET_DIRNAME


def campaign_dataset_dir(*, campaign_id: str, dataset_id: str) -> Path:
    structure, experiment, geometry, control_id = parse_gv_dataset_id(dataset_id)
    validate_gv_numerical_dataset_id(dataset_id, structure=structure)
    return campaign_dataset_root(campaign_id=campaign_id) / structure / experiment / geometry / control_id


def campaign_dataset_manifest_path(*, campaign_id: str, dataset_id: str) -> Path:
    return campaign_dataset_dir(campaign_id=campaign_id, dataset_id=dataset_id) / GV_NUMERICAL_DATASET_MANIFEST_FILENAME


def campaign_dataset_hdf5_path(*, campaign_id: str, dataset_id: str) -> Path:
    return campaign_dataset_dir(campaign_id=campaign_id, dataset_id=dataset_id) / GV_NUMERICAL_DATASET_HDF5_FILENAME


__all__ = [
    "GV_NUMERICAL_MANIFEST_SCHEMA_VERSION",
    "GV_NUMERICAL_POSTPROCESSOR_VERSION",
    "GV_NUMERICAL_DATA_ROOT",
    "GV_NUMERICAL_DATASET_DIRNAME",
    "GV_NUMERICAL_DATASET_MANIFEST_FILENAME",
    "GV_NUMERICAL_DATASET_HDF5_FILENAME",
    "build_gv_numerical_dataset_manifest",
    "validate_gv_numerical_manifest",
    "validate_gv_numerical_dataset_id",
    "parse_gv_dataset_id",
    "campaign_dataset_root",
    "campaign_dataset_dir",
    "campaign_dataset_manifest_path",
    "campaign_dataset_hdf5_path",
    "GVNumericalDatasetManifest",
    "GVNumericalUnits",
    "GVNumericalNormalization",
    "GVNumericalQualityFlags",
]
