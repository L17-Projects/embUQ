from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from meso_uq.structures.gv.geometries import build_geometry
from meso_uq.structures.gv.material_parameters import validate_material_parameter_overrides
from meso_uq.structures.gv.numerical_data import (
    build_gv_numerical_dataset_manifest,
    campaign_dataset_hdf5_path,
    campaign_dataset_root,
    campaign_dataset_manifest_path,
)
from meso_uq.structures.gv.postprocessing import GVNumericalPostprocessResult, POSTPROCESSORS
from meso_uq.structures.gv.runtime import GV_RUNTIME_DEFAULT_ROOT, plan_runtime
from meso_uq.structures.gv.runtime.base import RuntimeDryRun


_GV_NUMERICAL_GENERATOR_EXPERIMENTS: tuple[str, ...] = (
    "stretching",
    "buckling",
    "torsion",
    "eigenmodes",
)
_GV_NUMERICAL_GENERATOR_NAMESPACE = "gv"


@dataclass(frozen=True)
class GVNumericalGenerationResult:
    experiment: str
    geometry_id: str
    geometry_radius: float
    geometry_height: float
    controls: Mapping[str, float]
    material_parameters: Mapping[str, float]
    runtime_dry_run: RuntimeDryRun
    postprocess_result: GVNumericalPostprocessResult | None
    dataset_id: str
    manifest_path: Path
    hdf5_path: Path
    expected_dataset_id: str
    expected_manifest_path: Path
    expected_hdf5_path: Path
    raw_provenance: Mapping[str, Any]
    provenance: Mapping[str, Any]
    expected_manifest: Mapping[str, Any]


def _coerce_geometry_size(
    *,
    geometry_radius: float | None = None,
    radGV: float | None = None,
    geometry_height: float | None = None,
    height: float | None = None,
) -> tuple[float, float]:
    resolved_radius = geometry_radius if geometry_radius is not None else radGV
    resolved_height = geometry_height if geometry_height is not None else height
    if resolved_radius is None or resolved_height is None:
        raise ValueError(
            "Both geometry size components are required: geometry_radius/radGV and geometry_height/height."
        )
    radius_value = float(resolved_radius)
    height_value = float(resolved_height)
    if not isfinite(radius_value) or not isfinite(height_value):
        raise ValueError("geometry_radius and geometry_height must be finite values.")
    if radius_value <= 0:
        raise ValueError(f"geometry_radius must be positive, got {radius_value}.")
    if height_value <= 0:
        raise ValueError(f"geometry_height must be positive, got {height_value}.")
    return radius_value, height_value


def _coerce_controls(
    controls: Mapping[str, object] | None,
) -> Mapping[str, float] | None:
    if controls is None:
        return None
    if not isinstance(controls, Mapping):
        raise ValueError("controls must be a mapping.")
    if not controls:
        return None
    normalized: dict[str, float] = {}
    for name, value in controls.items():
        normalized_value = float(value)
        if not isfinite(normalized_value):
            raise ValueError(f"control {name!r} must be finite.")
        normalized[str(name)] = normalized_value
    return normalized


def _validate_experiment(experiment: str) -> str:
    experiment_name = str(experiment)
    if experiment_name == "shear_flow":
        raise ValueError(
            "experiment 'shear_flow' is not yet supported by the GV numerical generator API."
        )
    if experiment_name not in _GV_NUMERICAL_GENERATOR_EXPERIMENTS:
        supported = ", ".join(_GV_NUMERICAL_GENERATOR_EXPERIMENTS)
        raise ValueError(
            f"Unsupported GV experiment for numerical generation: {experiment_name!r}. "
            f"Supported: {supported}."
        )
    return experiment_name


def _resolve_postprocessor(experiment: str):
    key = f"{_GV_NUMERICAL_GENERATOR_NAMESPACE}:{experiment}"
    processor = POSTPROCESSORS.get(key)
    if processor is None:
        raise ValueError(f"No postprocessor is registered for experiment {experiment!r}.")
    return processor


def _default_raw_staging_quality_flags(
    fixture_like: Mapping[str, Any] | None,
    requested_quality_flags: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if requested_quality_flags is not None:
        return requested_quality_flags
    if fixture_like is not None:
        return None
    return {
        "finite_observables": False,
        "finite_observable_ratio": 0.0,
        "canary_failures": ("postprocessing_not_run",),
    }


def generate_gv_numerical_data(
    *,
    experiment: str,
    campaign_id: str,
    material_parameters: Mapping[str, object],
    geometry_radius: float | None = None,
    radGV: float | None = None,
    geometry_height: float | None = None,
    height: float | None = None,
    controls: Mapping[str, object] | None = None,
    output_root: str | Path = GV_RUNTIME_DEFAULT_ROOT,
    fixture_like: Mapping[str, Any] | None = None,
    quality_flags: Mapping[str, Any] | None = None,
    units: Mapping[str, Mapping[str, str]] | None = None,
    normalization: Mapping[str, Mapping[str, float]] | None = None,
) -> GVNumericalGenerationResult:
    """Generate GV numerical data staging artifacts for one experiment/geometry pair."""

    experiment_name = _validate_experiment(experiment)
    campaign_dataset_root(campaign_id=campaign_id)
    radius_value, height_value = _coerce_geometry_size(
        geometry_radius=geometry_radius,
        radGV=radGV,
        geometry_height=geometry_height,
        height=height,
    )
    normalized_material = validate_material_parameter_overrides(material_parameters)
    normalized_controls = _coerce_controls(controls)

    geometry = build_geometry(
        radius=radius_value,
        height=height_value,
        source="numerical_generator",
    )

    runtime = plan_runtime(
        experiment_name,
        output_root=output_root,
        geometry=geometry.id,
        controls=normalized_controls,
        material_parameter_overrides=normalized_material,
        include_experimental=False,
    )
    runtime_payload = runtime.to_manifest()
    runtime_payload["generator"] = {
        "campaign_id": campaign_id,
        "experiment": experiment_name,
        "geometry_radius": radius_value,
        "geometry_height": height_value,
        "source": "generator_api",
    }
    manifest_quality_flags = _default_raw_staging_quality_flags(
        fixture_like,
        quality_flags,
    )

    dataset_manifest = build_gv_numerical_dataset_manifest(
        campaign_id=campaign_id,
        structure=_GV_NUMERICAL_GENERATOR_NAMESPACE,
        experiment=experiment_name,
        geometry_radius=radius_value,
        geometry_height=height_value,
        material_parameters=normalized_material,
        controls=runtime.controls,
        raw_provenance=runtime_payload,
        quality_flags=manifest_quality_flags,
        units=units,
        normalization=normalization,
    )
    dataset_payload = dataset_manifest.to_manifest()
    dataset_id = str(dataset_payload["dataset_id"])
    manifest_path = campaign_dataset_manifest_path(campaign_id=campaign_id, dataset_id=dataset_id)
    hdf5_path = campaign_dataset_hdf5_path(campaign_id=campaign_id, dataset_id=dataset_id)

    postprocess_result = None
    if fixture_like is not None:
        if not isinstance(fixture_like, Mapping):
            raise ValueError("fixture_like must be a mapping.")
        postprocess = _resolve_postprocessor(experiment_name)
        postprocess_result = postprocess(
            campaign_id=campaign_id,
            geometry_radius=radius_value,
            geometry_height=height_value,
            material_parameters=normalized_material,
            controls=runtime.controls,
            raw_provenance=runtime_payload,
            fixture_like=fixture_like,
            quality_flags=quality_flags,
            units=units,
            normalization=normalization,
        )

    return GVNumericalGenerationResult(
        experiment=experiment_name,
        geometry_id=geometry.id,
        geometry_radius=radius_value,
        geometry_height=height_value,
        controls=runtime.controls,
        material_parameters=normalized_material,
        runtime_dry_run=runtime,
        postprocess_result=postprocess_result,
        dataset_id=dataset_id,
        manifest_path=manifest_path,
        hdf5_path=hdf5_path,
        expected_dataset_id=dataset_id,
        expected_manifest_path=manifest_path,
        expected_hdf5_path=hdf5_path,
        raw_provenance=runtime_payload,
        provenance=runtime_payload,
        expected_manifest=dataset_payload,
    )


__all__ = [
    "GVNumericalGenerationResult",
    "generate_gv_numerical_data",
]
