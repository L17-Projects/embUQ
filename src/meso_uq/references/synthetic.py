from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from meso_uq.experiments import canonical_dataset_id
from meso_uq.structures import get_structure

from .gv_common import (
    DEFAULT_SURROGATE_BACKEND,
    GV_REFERENCE_KINDS,
    GVReferenceMaterialization,
    GVReferenceRecord,
    build_reference_record,
    control_identifier,
    materialize_reference_records,
    resolve_gv_reference_context,
    validate_reference_axes,
)

_DEFAULT_CURVES: dict[str, tuple[tuple[float, ...], tuple[float, ...], str]] = {
    "stretching": ((0.0, 0.25, 0.5, 0.75, 1.0), (0.0, 0.08, 0.19, 0.31, 0.44), "extension_curve"),
    "buckling": ((0.0, 0.25, 0.5, 0.75, 1.0), (1.0, 0.92, 0.75, 0.53, 0.28), "buckling_response"),
    "torsion": ((0.0, 0.25, 0.5, 0.75, 1.0), (0.0, 0.04, 0.1, 0.17, 0.25), "torsion_response"),
    "eigenmodes": ((1.0, 2.0, 3.0, 4.0), (0.18, 0.41, 0.73, 1.02), "eigenmode_spectrum"),
    "shear_flow": ((0.0, 0.5, 1.0, 1.5), (0.0, 0.09, 0.21, 0.34), "shear_flow_response"),
}


@dataclass(frozen=True)
class ReferenceIdentity:
    structure: str
    experiment: str
    geometry: str
    controls: Mapping[str, float]
    surrogate_backend: str
    reference_kind: str
    dataset_id: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "structure": self.structure,
            "experiment": self.experiment,
            "geometry": self.geometry,
            "controls": dict(self.controls),
            "surrogate_backend": self.surrogate_backend,
            "reference_kind": self.reference_kind,
            "dataset_id": self.dataset_id,
        }


@dataclass(frozen=True)
class SyntheticReferenceFixture:
    identity: ReferenceIdentity
    observable: str
    points: tuple[float, ...]
    values: tuple[float, ...]
    calibrated_parameter_names: tuple[str, ...]
    nuisance_parameters: Mapping[str, float]
    noise_model: Mapping[str, object]
    metadata: Mapping[str, object]
    runtime_manifest: Mapping[str, object] | None = None

    def to_manifest(self) -> dict[str, object]:
        manifest = {
            **self.identity.to_manifest(),
            "observable": self.observable,
            "points": list(self.points),
            "values": list(self.values),
            "calibrated_parameter_names": list(self.calibrated_parameter_names),
            "nuisance_parameters": dict(self.nuisance_parameters),
            "noise_model": dict(self.noise_model),
            "metadata": dict(self.metadata),
        }
        if self.runtime_manifest:
            manifest["runtime_manifest"] = dict(self.runtime_manifest)
        return manifest

    def to_record(self) -> GVReferenceRecord:
        context = resolve_gv_reference_context(
            runtime_manifest=self.runtime_manifest,
            experiment=self.identity.experiment,
            geometry=self.identity.geometry,
            controls=self.identity.controls,
        )
        return build_reference_record(
            context=context,
            reference_kind=self.identity.reference_kind,
            reference_source=str(self.metadata.get("source", "synthetic_fixture")),
            observable=self.observable,
            points=self.points,
            values=self.values,
            nuisance_parameters=self.nuisance_parameters,
            metadata=self.metadata,
            surrogate_backend=self.identity.surrogate_backend,
        )


@dataclass(frozen=True)
class SyntheticGVReference:
    seed: int
    points: tuple[float, ...]
    values: tuple[float, ...]
    manifest: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return dict(self.manifest)


def _normalize_controls(controls: Mapping[str, Any]) -> dict[str, float]:
    return {str(name): float(value) for name, value in controls.items()}


def _normalize_series(
    name: str,
    values: Sequence[float] | None,
    fallback: Sequence[float],
) -> tuple[float, ...]:
    series = tuple(float(value) for value in (fallback if values is None else values))
    if not series:
        raise ValueError(f"{name} must contain at least one value.")
    return series


def _default_fixture(experiment: str) -> tuple[tuple[float, ...], tuple[float, ...], str]:
    try:
        return _DEFAULT_CURVES[experiment]
    except KeyError as exc:
        raise ValueError(f"No synthetic fixture template registered for experiment '{experiment}'.") from exc


def _gv_calibrated_parameter_names() -> tuple[str, ...]:
    return get_structure("gv").parameter_contract.calibrated_names


def _validate_fixture_request(
    *,
    experiment: str,
    controls: Mapping[str, float],
    surrogate_backend: str,
    reference_kind: str,
) -> None:
    validate_reference_axes(
        surrogate_backend=surrogate_backend,
        reference_kind=reference_kind,
    )
    experiment_spec = get_structure("gv").get_experiment(experiment, include_experimental=True)
    control_names = set(controls)
    expected_controls = set(experiment_spec.control_names)
    unknown = sorted(control_names - expected_controls)
    missing = sorted(expected_controls - control_names)
    if unknown:
        raise ValueError(
            f"Unknown GV controls for experiment '{experiment}': {', '.join(unknown)}. "
            f"Expected only: {', '.join(experiment_spec.control_names)}."
        )
    if missing:
        raise ValueError(
            f"Missing GV controls for experiment '{experiment}': {', '.join(missing)}. "
            "Synthetic fixtures must preserve the full runtime control identity."
        )


def _validate_gv_reference_axes(*, surrogate_backend: str, reference_kind: str) -> None:
    validate_reference_axes(
        surrogate_backend=surrogate_backend,
        reference_kind=reference_kind,
    )


def _identity_from_axes(
    *,
    structure: str,
    experiment: str,
    geometry: str,
    controls: Mapping[str, float],
    surrogate_backend: str,
    reference_kind: str,
    dataset_id: str | None = None,
) -> ReferenceIdentity:
    _validate_gv_reference_axes(
        surrogate_backend=surrogate_backend,
        reference_kind=reference_kind,
    )
    normalized_controls = _normalize_controls(controls)
    return ReferenceIdentity(
        structure=structure,
        experiment=experiment,
        geometry=geometry,
        controls=normalized_controls,
        surrogate_backend=surrogate_backend,
        reference_kind=reference_kind,
        dataset_id=dataset_id
        or canonical_dataset_id(structure, experiment, geometry, controls=_control_id(normalized_controls)),
    )


def _control_id(controls: Mapping[str, float]) -> str:
    return control_identifier(controls)


def build_gv_synthetic_fixture(
    *,
    experiment: str,
    geometry: str,
    controls: Mapping[str, Any],
    sigma: float,
    d0: float | None = None,
    surrogate_backend: str = DEFAULT_SURROGATE_BACKEND,
    reference_kind: str = "synthetic",
    points: Sequence[float] | None = None,
    values: Sequence[float] | None = None,
    observable: str | None = None,
    metadata: Mapping[str, object] | None = None,
    dataset_id: str | None = None,
    runtime_manifest: Mapping[str, object] | None = None,
) -> SyntheticReferenceFixture:
    default_points, default_values, default_observable = _default_fixture(experiment)
    resolved_points = _normalize_series("points", points, default_points)
    resolved_values = _normalize_series("values", values, default_values)
    if len(resolved_points) != len(resolved_values):
        raise ValueError("points and values must have the same length.")
    normalized_controls = _normalize_controls(controls)

    calibrated_parameter_names = _gv_calibrated_parameter_names()
    control_names = set(normalized_controls)
    overlap = control_names.intersection(calibrated_parameter_names)
    if overlap:
        raise ValueError(
            "GV controls must remain separate from calibrated parameters: "
            + ", ".join(sorted(overlap))
        )
    _validate_fixture_request(
        experiment=experiment,
        controls=normalized_controls,
        surrogate_backend=surrogate_backend,
        reference_kind=reference_kind,
    )

    nuisance_parameters: dict[str, float] = {"sigma": float(sigma)}
    if d0 is not None:
        nuisance_parameters["d0"] = float(d0)

    identity = _identity_from_axes(
        structure="gv",
        experiment=experiment,
        geometry=geometry,
        controls=normalized_controls,
        surrogate_backend=surrogate_backend,
        reference_kind=reference_kind,
        dataset_id=dataset_id,
    )
    resolved_metadata = {"source": "deterministic_fixture", **(metadata or {})}
    return SyntheticReferenceFixture(
        identity=identity,
        observable=observable or default_observable,
        points=resolved_points,
        values=resolved_values,
        calibrated_parameter_names=calibrated_parameter_names,
        nuisance_parameters=nuisance_parameters,
        noise_model={"kind": "multiplicative", "parameter": "sigma"},
        metadata=resolved_metadata,
        runtime_manifest=dict(runtime_manifest or {}),
    )


def synthetic_fixture_from_runtime_manifest(
    runtime_manifest: Mapping[str, object],
    *,
    sigma: float,
    d0: float | None = None,
    surrogate_backend: str = DEFAULT_SURROGATE_BACKEND,
    points: Sequence[float] | None = None,
    values: Sequence[float] | None = None,
    observable: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SyntheticReferenceFixture:
    source_files = tuple(str(path) for path in runtime_manifest.get("source_files", ()))
    combined_metadata = {
        "source": "runtime_manifest",
        "runtime_dataset_id": runtime_manifest.get("dataset_id"),
        "runtime_package": runtime_manifest.get("runtime_package"),
        "source_files": source_files,
        **(metadata or {}),
    }
    return build_gv_synthetic_fixture(
        experiment=str(runtime_manifest["experiment"]),
        geometry=str(runtime_manifest["geometry"]),
        controls=runtime_manifest["controls"],
        sigma=sigma,
        d0=d0,
        surrogate_backend=surrogate_backend,
        reference_kind="synthetic",
        points=points,
        values=values,
        observable=observable,
        metadata=combined_metadata,
        dataset_id=str(runtime_manifest["dataset_id"]),
        runtime_manifest=runtime_manifest,
    )


def generate_gv_synthetic_reference(
    *,
    seed: int,
    runtime_manifest: Mapping[str, object] | None = None,
    experiment: str | None = None,
    geometry: str | None = None,
    controls: Mapping[str, float] | None = None,
    point_count: int = 64,
) -> SyntheticGVReference:
    if point_count < 2:
        raise ValueError("Synthetic GV references require at least two sample points.")
    context = resolve_gv_reference_context(
        runtime_manifest=runtime_manifest,
        experiment=experiment,
        geometry=geometry,
        controls=controls,
    )
    points = tuple(index / (point_count - 1) for index in range(point_count))
    values = tuple(
        _seeded_synthetic_value(
            seed,
            context.dataset_id,
            context.experiment.name,
            context.controls,
            point,
        )
        for point in points
    )
    record = build_reference_record(
        context=context,
        reference_kind="synthetic",
        reference_source="deterministic_seeded_generator",
        observable=context.experiment.observables[0].name,
        points=points,
        values=values,
        nuisance_parameters={"sigma": 0.0},
        metadata={"synthetic_seed": int(seed), "point_count": point_count},
    )
    manifest = record.to_manifest()
    manifest.update(
        {
            "synthetic_seed": int(seed),
            "point_count": point_count,
            "points": list(points),
            "values": list(values),
        }
    )
    return SyntheticGVReference(seed=int(seed), points=points, values=values, manifest=manifest)


def build_gv_synthetic_reference_manifest(**kwargs: Any) -> dict[str, Any]:
    return generate_gv_synthetic_reference(**kwargs).to_manifest()


def materialize_gv_synthetic_reference_dataset(
    fixtures: Sequence[SyntheticReferenceFixture],
    *,
    data_path: str | Path,
    metadata_path: str | Path | None = None,
    collection_id: str | None = None,
    experiments: Sequence[str] | None = None,
    geometries: Sequence[str] | None = None,
    dataset_ids: Sequence[str] | None = None,
) -> GVReferenceMaterialization:
    return materialize_reference_records(
        [fixture.to_record() for fixture in fixtures],
        data_path=data_path,
        metadata_path=metadata_path,
        collection_id=collection_id,
        experiments=experiments,
        geometries=geometries,
        dataset_ids=dataset_ids,
    )


def materialize_gv_synthetic_reference(
    *,
    seed: int,
    output_root: str | Path,
    runtime_manifest: Mapping[str, object] | None = None,
    experiment: str | None = None,
    geometry: str | None = None,
    controls: Mapping[str, float] | None = None,
    point_count: int = 64,
    dataset_filename: str = "reference_dataset.npz",
    manifest_filename: str = "reference_manifest.json",
) -> GVReferenceMaterialization:
    reference = generate_gv_synthetic_reference(
        seed=seed,
        runtime_manifest=runtime_manifest,
        experiment=experiment,
        geometry=geometry,
        controls=controls,
        point_count=point_count,
    )
    resolved_root = Path(output_root).resolve()
    dataset_path = resolved_root / dataset_filename
    manifest_path = resolved_root / manifest_filename
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = reference.to_manifest()
    manifest["artifacts"] = {
        "reference_dataset": str(dataset_path),
        "reference_manifest": str(manifest_path),
    }
    np.savez_compressed(
        dataset_path,
        points=np.asarray(reference.points, dtype=np.float64),
        values=np.asarray(reference.values, dtype=np.float64),
        dataset_id=np.asarray(manifest["dataset_id"], dtype="<U256"),
        structure=np.asarray(manifest["structure"], dtype="<U32"),
        experiment=np.asarray(manifest["experiment"], dtype="<U64"),
        geometry=np.asarray(manifest["geometry"], dtype="<U128"),
        controls_json=np.asarray(json.dumps(manifest["controls"], sort_keys=True), dtype="<U4096"),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True), dtype="<U65535"),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return GVReferenceMaterialization(
        data_path=dataset_path,
        metadata_path=manifest_path,
        manifest=manifest,
    )


def _seeded_synthetic_value(
    seed: int,
    dataset_id: str,
    experiment_name: str,
    controls: Mapping[str, float],
    point: float,
) -> float:
    control_scale = sum(abs(value) for value in controls.values()) / max(len(controls), 1)
    baseline = {
        "stretching": 0.15 + 1.35 * point,
        "buckling": 1.2 - 0.9 * point,
        "torsion": 0.2 + 0.6 * point,
        "eigenmodes": 0.7 + 0.25 * point,
        "shear_flow": 0.4 + 0.5 * point,
    }.get(experiment_name, 0.25 + point)
    modulation = _uniform_01(seed, dataset_id, f"{experiment_name}:{point:.12f}") - 0.5
    return round(baseline + 0.0025 * control_scale + 0.08 * modulation, 8)


def _uniform_01(seed: int, dataset_id: str, token: str) -> float:
    digest = hashlib.sha256(f"{seed}:{dataset_id}:{token}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64 - 1)


__all__ = [
    "DEFAULT_SURROGATE_BACKEND",
    "GV_REFERENCE_KINDS",
    "ReferenceIdentity",
    "SyntheticGVReference",
    "SyntheticReferenceFixture",
    "build_gv_synthetic_reference_manifest",
    "build_gv_synthetic_fixture",
    "generate_gv_synthetic_reference",
    "materialize_gv_synthetic_reference",
    "materialize_gv_synthetic_reference_dataset",
    "synthetic_fixture_from_runtime_manifest",
]
