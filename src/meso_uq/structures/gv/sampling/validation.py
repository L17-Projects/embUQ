from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from ..material_parameters import validate_material_parameter_overrides
from ..controls import EXPERIMENT_CONTROLS

from .types import GVMaterialGeometry, GVSweep, GVRuntimeOptions


GV_SAMPLING_EXPERIMENTS = ("stretching", "buckling", "torsion", "eigenmodes")
_DEFAULT_GV_SAMPLING_ROOT = "_runs/gv/sampling"


def validate_gv_experiment(experiment: object) -> str:
    experiment_name = str(experiment)
    if experiment_name == "shear_flow":
        raise ValueError("GV experiment 'shear_flow' is not supported for sampling.")
    if experiment_name not in GV_SAMPLING_EXPERIMENTS:
        raise ValueError(
            f"Unsupported GV experiment {experiment_name!r}; "
            f"supported experiments are {', '.join(GV_SAMPLING_EXPERIMENTS)}."
        )
    return experiment_name


def validate_material_parameters(materials: Mapping[str, object]) -> dict[str, float]:
    return validate_material_parameter_overrides(materials)


def _coerce_geometry_value(value: object, *, field_name: str) -> float:
    if value is None:
        raise ValueError(f"GV geometry {field_name} is required.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GV geometry {field_name} must be numeric.") from exc
    return number


def validate_geometry(
    geometry: GVMaterialGeometry | Mapping[str, object] | None,
    *,
    radGV: float | None = None,
    height: float | None = None,
) -> GVMaterialGeometry:
    if geometry is None:
        radius = _coerce_geometry_value(radGV, field_name="radGV")
        geometry_height = _coerce_geometry_value(height, field_name="height")
        return GVMaterialGeometry(radGV=radius, height=geometry_height)

    if isinstance(geometry, GVMaterialGeometry):
        if radGV is not None or height is not None:
            raise ValueError(
                "Provide either geometry directly or explicit radGV/height, not both."
            )
        return geometry

    if not isinstance(geometry, Mapping):
        raise ValueError("GV geometry must be a GVMaterialGeometry or mapping.")

    geometry_payload = dict(geometry)
    radius = _coerce_geometry_value(
        geometry_payload.get("radGV", radGV),
        field_name="radGV",
    )
    geometry_height = _coerce_geometry_value(
        geometry_payload.get("height", geometry_payload.get("geometry_height", height)),
        field_name="height",
    )
    return GVMaterialGeometry(radGV=radius, height=geometry_height)


def _coerce_control_scalar(name: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GV control {name!r} must be numeric.") from exc
    if not isfinite(number):
        raise ValueError(f"GV control {name!r} must be finite.")
    return number


def _coerce_control_sequence(name: str, value: Sequence[object]) -> tuple[float, ...]:
    if not value:
        raise ValueError(
            f"GV control {name!r} sweep values must be a non-empty finite sequence."
        )
    values: list[float] = []
    for raw_value in value:
        values.append(_coerce_control_scalar(name, raw_value))
    return tuple(values)


def _is_sequence(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    return isinstance(value, Sequence)


def _allowed_control_names(experiment: str) -> set[str]:
    if experiment not in EXPERIMENT_CONTROLS:
        raise ValueError(f"Unknown GV experiment {experiment!r}.")
    return {control.name for control in EXPERIMENT_CONTROLS[experiment]}


def validate_explicit_controls(
    *,
    experiment: str,
    controls: Mapping[str, object] | None,
) -> tuple[dict[str, float], GVSweep]:
    allowed_names = _allowed_control_names(experiment)
    if controls is None:
        raise ValueError("GV sampling requires explicit controls.")
    if not isinstance(controls, Mapping):
        raise ValueError("GV controls must be a mapping.")

    fixed_controls: dict[str, float] = {}
    sweep_candidates: list[tuple[str, tuple[float, ...]]] = []
    for raw_name, raw_value in controls.items():
        name = str(raw_name)
        if name not in allowed_names:
            raise ValueError(f"GV control {name!r} is not valid for experiment {experiment!r}.")
        if _is_sequence(raw_value):
            sweep_candidates.append((name, _coerce_control_sequence(name, list(raw_value))))  # type: ignore[arg-type]
        else:
            fixed_controls[name] = _coerce_control_scalar(name, raw_value)

    if len(sweep_candidates) != 1:
        raise ValueError(
            "GV sampling requires exactly one explicit sweep axis with non-empty finite values."
        )
    sweep_axis, sweep_values = sweep_candidates[0]
    return fixed_controls, GVSweep(axis=sweep_axis, values=sweep_values)


def validate_sample_gv_request(
    *,
    experiment: object,
    material_parameters: Mapping[str, object],
    geometry: GVMaterialGeometry | Mapping[str, object] | None,
    controls: Mapping[str, object] | None,
    runtime_options: GVRuntimeOptions | None = None,
    radGV: float | None = None,
    height: float | None = None,
) -> tuple[str, dict[str, float], GVMaterialGeometry, dict[str, float], GVSweep, GVRuntimeOptions]:
    validated_experiment = validate_gv_experiment(experiment)
    validated_materials = validate_material_parameters(material_parameters)
    validated_geometry = validate_geometry(geometry, radGV=radGV, height=height)
    controls_payload = (
        controls
        if controls is not None
        else (runtime_options.controls if runtime_options is not None else None)
    )
    fixed_controls, sweep = validate_explicit_controls(
        experiment=validated_experiment,
        controls=controls_payload,
    )
    if runtime_options is None:
        runtime_options = GVRuntimeOptions(
            controls=fixed_controls,
            output_root=_DEFAULT_GV_SAMPLING_ROOT,
        )
    else:
        runtime_options = GVRuntimeOptions(
            controls=fixed_controls,
            output_root=str(runtime_options.output_root),
            timeout_seconds=runtime_options.timeout_seconds,
        )
    return (
        validated_experiment,
        validated_materials,
        validated_geometry,
        fixed_controls,
        sweep,
        runtime_options,
    )


__all__ = [
    "GV_SAMPLING_EXPERIMENTS",
    "validate_explicit_controls",
    "validate_gv_experiment",
    "validate_geometry",
    "validate_material_parameters",
    "validate_sample_gv_request",
]
