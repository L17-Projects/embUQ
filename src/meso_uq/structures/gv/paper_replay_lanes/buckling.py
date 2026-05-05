from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..sampling import GVRuntimeOptions, sample_gv
from ..sampling.buckling import parse_buckling_lane_channels
from ..parameters import GV_MATERIAL_PARAMETER_NAMES
from ..sampling.validation import validate_geometry


BUCKLING_FIGURE_ID = "Figure 7"
_BUCKLING_SWEEP_AXIS = "bpress"
_REQUIRED_CHANNELS = ("bpress", "relative_volume")
_OPTIONAL_CHANNELS = (
    "buckling_response",
    "force_response",
    "pressure_response",
    "shape_amplitude",
    "deformation_amplitude",
)
_DEFAULT_OUTPUT_ROOT = Path("_runs/gv/paper_replay/buckling")
_PRESSURE_ALIASES = {
    "pressure_difference": "bpress",
    "pressure": "bpress",
    "background_pressure": "bpress",
}
_CHANNEL_ALIASES = {
    "pressure_difference": "bpress",
    "pressure": "bpress",
    "background_pressure": "bpress",
    "volumetric_strain": "relative_volume",
    "volume_strain": "relative_volume",
}
_ZERO_ALLOWED_MATERIAL_PARAMETERS = frozenset({"a3", "a4", "b1", "b2"})
_MATERIAL_ALIASES = {"muL": "mu_l"}


@dataclass(frozen=True)
class BucklingPaperReplayPlan:
    figure_id: str
    experiment: str
    geometry: dict[str, float]
    material_parameters: dict[str, float]
    controls: dict[str, object]
    runtime_options: GVRuntimeOptions
    mapping_assumptions: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "figure_id": self.figure_id,
            "experiment": self.experiment,
            "geometry": dict(self.geometry),
            "material_parameters": dict(self.material_parameters),
            "controls": {
                name: list(values) if isinstance(values, tuple) else values
                for name, values in self.controls.items()
            },
            "runtime_options": {
                "output_root": str(self.runtime_options.output_root),
                "controls": dict(self.runtime_options.controls),
                "timeout_seconds": self.runtime_options.timeout_seconds,
            },
            "mapping_assumptions": list(self.mapping_assumptions),
        }


@dataclass(frozen=True)
class BucklingPaperReplayResult:
    figure_id: str
    experiment: str
    axis: str
    controls: dict[str, float]
    geometry: dict[str, float]
    material_parameters: dict[str, float]
    channels: dict[str, np.ndarray]
    provenance: dict[str, Any]

    def to_manifest(self) -> dict[str, object]:
        return {
            "figure_id": self.figure_id,
            "experiment": self.experiment,
            "axis": self.axis,
            "controls": dict(self.controls),
            "geometry": dict(self.geometry),
            "material_parameters": dict(self.material_parameters),
            "channels": {
                name: np.asarray(values).tolist()
                for name, values in self.channels.items()
            },
            "provenance": dict(self.provenance),
        }


def plan_buckling_paper_replay_lane(
    *,
    material_parameters: Mapping[str, object],
    geometry: Mapping[str, object] | None = None,
    radGV: float | None = None,
    height: float | None = None,
    pressure_differences: tuple[float, ...] | list[float],
    buck: float = 0.75,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    timeout_seconds: int = 7200,
) -> BucklingPaperReplayPlan:
    validated_geometry = validate_geometry(geometry, radGV=radGV, height=height)
    validated_materials = _validate_paper_replay_material_parameters(material_parameters)
    pressures = _coerce_finite_1d(pressure_differences, name="pressure_differences")
    if pressures.size < 2:
        raise ValueError("Buckling paper replay requires at least two pressure-difference samples.")
    buck_value = _coerce_scalar(buck, name="buck")
    validated_output_root = _validate_runs_root(output_root)
    runtime_options = GVRuntimeOptions(
        controls={"buck": buck_value},
        output_root=str(validated_output_root),
        timeout_seconds=timeout_seconds,
    )
    return BucklingPaperReplayPlan(
        figure_id=BUCKLING_FIGURE_ID,
        experiment="buckling",
        geometry={"radGV": validated_geometry.radGV, "height": validated_geometry.height},
        material_parameters=validated_materials,
        controls={"buck": buck_value, _BUCKLING_SWEEP_AXIS: tuple(float(value) for value in pressures)},
        runtime_options=runtime_options,
        mapping_assumptions=(
            "Paper replay sweeps `bpress` as the pressure-difference proxy while holding `buck` fixed.",
            "This assumes the staged Mirheo buckling runtime can be interpreted qualitatively against Figure 7 via relative volume versus pressure difference.",
            "The legacy runtime descriptor still advertises `buck` as its native sweep axis; verify this mapping against MES-113 data provenance before production replay runs.",
        ),
    )


def run_buckling_paper_replay_lane(
    plan: BucklingPaperReplayPlan,
    *,
    sampler: Callable[..., object] = sample_gv,
) -> BucklingPaperReplayResult:
    sample_result = sampler(
        experiment=plan.experiment,
        material_parameters=plan.material_parameters,
        geometry=plan.geometry,
        controls=plan.controls,
        runtime_options=plan.runtime_options,
        write_artifacts=False,
    )
    return postprocess_buckling_paper_replay_lane(sample_result, plan=plan)


def postprocess_buckling_paper_replay_lane(
    sample_result: object,
    *,
    plan: BucklingPaperReplayPlan | None = None,
) -> BucklingPaperReplayResult:
    payload = _coerce_mapping(sample_result, context="sample_result")
    channels = _normalize_buckling_channels(payload.get("channels"))

    missing = [name for name in _REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise ValueError(
            "Buckling paper replay requires channels: " + ", ".join(_REQUIRED_CHANNELS) + "."
        )

    if channels["bpress"].shape != channels["relative_volume"].shape:
        raise ValueError("Buckling paper replay requires 'bpress' and 'relative_volume' to share a shape.")

    figure_id = plan.figure_id if plan is not None else BUCKLING_FIGURE_ID
    experiment = str(payload.get("experiment", plan.experiment if plan is not None else "buckling"))
    geometry = _resolve_geometry(payload, plan)
    material_parameters = _resolve_material_parameters(payload, plan)
    controls = _resolve_controls(payload, plan, required=("buck",))
    provenance = _resolve_provenance(payload, plan)
    provenance["required_channels"] = list(_REQUIRED_CHANNELS)
    provenance["mapping_assumptions"] = list(
        plan.mapping_assumptions
        if plan is not None
        else (
            "Paper replay interprets `bpress` as pressure difference with fixed `buck`.",
        )
    )

    return BucklingPaperReplayResult(
        figure_id=figure_id,
        experiment=experiment,
        axis=_BUCKLING_SWEEP_AXIS,
        controls=controls,
        geometry=geometry,
        material_parameters=material_parameters,
        channels=channels,
        provenance=provenance,
    )


def plot_buckling_paper_replay(
    result: BucklingPaperReplayResult,
    *,
    output_path: str | Path,
) -> Path:
    target = _validate_runs_output_path(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pressure = result.channels["bpress"]
    relative_volume = result.channels["relative_volume"]
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(pressure, relative_volume, marker="o", linewidth=1.8, color="#0b7285")
    axis.set_xlabel("Pressure difference proxy (`bpress`)")
    axis.set_ylabel("Relative volume")
    axis.set_title(f"GV buckling paper replay: {result.figure_id}")
    axis.grid(True, alpha=0.3)
    assumption = result.provenance.get("mapping_assumptions", [""])[0]
    figure.text(
        0.02,
        0.02,
        f"experiment={result.experiment} | buck={result.controls['buck']} | {assumption}",
        fontsize=8,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _normalize_buckling_channels(channels_like: object) -> dict[str, np.ndarray]:
    payload = _coerce_mapping(channels_like, context="sample_result.channels")
    remapped: dict[str, Any] = {}
    for raw_name, values in payload.items():
        canonical_name = _CHANNEL_ALIASES.get(str(raw_name), str(raw_name))
        remapped[canonical_name] = values
    parsed = parse_buckling_lane_channels({"channels": remapped})
    normalized: dict[str, np.ndarray] = {}
    for channel_name in (*_REQUIRED_CHANNELS, *_OPTIONAL_CHANNELS):
        if channel_name in parsed:
            normalized[channel_name] = np.asarray(parsed[channel_name], dtype=float)
    return normalized


def _coerce_mapping(payload: object, *, context: str) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    extracted: dict[str, Any] = {}
    for name in (
        "experiment",
        "geometry",
        "material_parameters",
        "controls",
        "sweep",
        "channels",
        "manifest",
        "runtime_manifests",
        "plan_manifests",
        "work_dirs",
        "runtime_seconds",
        "status",
    ):
        if hasattr(payload, name):
            extracted[name] = getattr(payload, name)
    if extracted:
        geometry = extracted.get("geometry")
        if hasattr(geometry, "radGV") and hasattr(geometry, "height"):
            extracted["geometry"] = {"radGV": geometry.radGV, "height": geometry.height}
        return extracted
    if hasattr(payload, "as_manifest"):
        manifest = payload.as_manifest()
        if isinstance(manifest, Mapping):
            return manifest
    raise ValueError(f"{context} must be a mapping or GVSampleResult-like object.")


def _coerce_finite_1d(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D numeric sequence.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _coerce_scalar(value: object, *, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        scalar = float(array.item())
    elif array.size == 1:
        scalar = float(array.reshape(-1)[0])
    else:
        raise ValueError(f"{name} must be a scalar value.")
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _validate_paper_replay_material_parameters(overrides: Mapping[str, object]) -> dict[str, float]:
    canonicalized: dict[str, float] = {}
    for raw_name, raw_value in overrides.items():
        canonical_name = _MATERIAL_ALIASES.get(str(raw_name), str(raw_name))
        if canonical_name not in GV_MATERIAL_PARAMETER_NAMES:
            raise ValueError(f"Unexpected material parameter '{raw_name}'.")
        if canonical_name in canonicalized:
            raise ValueError(f"Material parameter '{canonical_name}' is duplicated in overrides.")
        value = _coerce_scalar(raw_value, name=f"material_parameters.{canonical_name}")
        if not isfinite(value):
            raise ValueError(f"Material parameter '{canonical_name}' must be finite.")
        if value < 0.0:
            raise ValueError(f"Material parameter '{canonical_name}' must be finite and >= 0.")
        if value == 0.0 and canonical_name not in _ZERO_ALLOWED_MATERIAL_PARAMETERS:
            raise ValueError(f"Material parameter '{canonical_name}' must be finite and > 0.")
        canonicalized[canonical_name] = value

    missing = [name for name in GV_MATERIAL_PARAMETER_NAMES if name not in canonicalized]
    if missing:
        raise ValueError("Missing required GV material parameters: " + ", ".join(missing))
    return {name: canonicalized[name] for name in GV_MATERIAL_PARAMETER_NAMES}


def _resolve_geometry(
    payload: Mapping[str, Any],
    plan: BucklingPaperReplayPlan | None,
) -> dict[str, float]:
    geometry_like = payload.get("geometry", {})
    if isinstance(geometry_like, Mapping) and {"radGV", "height"} <= set(geometry_like):
        return {
            "radGV": _coerce_scalar(geometry_like["radGV"], name="geometry.radGV"),
            "height": _coerce_scalar(geometry_like["height"], name="geometry.height"),
        }
    if plan is not None:
        return dict(plan.geometry)
    raise ValueError("Buckling paper replay requires geometry metadata.")


def _resolve_material_parameters(
    payload: Mapping[str, Any],
    plan: BucklingPaperReplayPlan | None,
) -> dict[str, float]:
    if isinstance(payload.get("material_parameters"), Mapping):
        return _validate_paper_replay_material_parameters(payload["material_parameters"])
    if plan is not None:
        return dict(plan.material_parameters)
    raise ValueError("Buckling paper replay requires material_parameters metadata.")


def _resolve_controls(
    payload: Mapping[str, Any],
    plan: BucklingPaperReplayPlan | None,
    *,
    required: tuple[str, ...],
) -> dict[str, float]:
    controls_like = payload.get("controls", {})
    normalized: dict[str, float] = {}
    if isinstance(controls_like, Mapping):
        for raw_name, value in controls_like.items():
            canonical_name = _PRESSURE_ALIASES.get(str(raw_name), str(raw_name))
            if canonical_name == _BUCKLING_SWEEP_AXIS and isinstance(value, (list, tuple)):
                continue
            if canonical_name in {required[0], _BUCKLING_SWEEP_AXIS}:
                normalized[canonical_name] = _coerce_scalar(value, name=f"controls.{canonical_name}")
    if plan is not None:
        for name in required:
            normalized.setdefault(name, _coerce_scalar(plan.controls[name], name=f"plan.controls.{name}"))
    missing = [name for name in required if name not in normalized]
    if missing:
        raise ValueError("Buckling paper replay requires scalar controls: " + ", ".join(required) + ".")
    return normalized


def _resolve_provenance(
    payload: Mapping[str, Any],
    plan: BucklingPaperReplayPlan | None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    manifest = payload.get("manifest")
    if isinstance(manifest, Mapping):
        provenance["sampling_manifest"] = dict(manifest)
    runtime_manifests = payload.get("runtime_manifests")
    if isinstance(runtime_manifests, tuple | list):
        provenance["runtime_manifests"] = list(runtime_manifests)
    plan_manifests = payload.get("plan_manifests")
    if isinstance(plan_manifests, tuple | list):
        provenance["plan_manifests"] = list(plan_manifests)
    work_dirs = payload.get("work_dirs")
    if isinstance(work_dirs, tuple | list):
        provenance["work_dirs"] = [str(path) for path in work_dirs]
    if "runtime_seconds" in payload:
        provenance["runtime_seconds"] = payload["runtime_seconds"]
    if "status" in payload:
        provenance["status"] = payload["status"]
    if plan is not None:
        provenance["lane_plan"] = plan.to_manifest()
    return provenance


def _validate_runs_output_path(output_path: str | Path) -> Path:
    return _validate_runs_root(output_path, label="plots")


def _validate_runs_root(output_path: str | Path, *, label: str = "outputs") -> Path:
    path = Path(output_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Paper replay {label} must be written under a relative _runs/ path.")
    if path.parts[:1] != ("_runs",):
        raise ValueError(f"Paper replay {label} must be written under _runs/.")
    return path


__all__ = [
    "BUCKLING_FIGURE_ID",
    "BucklingPaperReplayPlan",
    "BucklingPaperReplayResult",
    "plan_buckling_paper_replay_lane",
    "plot_buckling_paper_replay",
    "postprocess_buckling_paper_replay_lane",
    "run_buckling_paper_replay_lane",
]
