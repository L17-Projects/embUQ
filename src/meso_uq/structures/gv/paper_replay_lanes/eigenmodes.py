from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..sampling import GVRuntimeOptions, sample_gv
from ..sampling.eigenmodes import parse_eigenmodes_lane_channels
from ..parameters import GV_MATERIAL_PARAMETER_NAMES
from ..sampling.validation import validate_geometry


EIGENMODES_FIGURE_ID = "Figure 8"
_EIGENMODES_AXIS = "mode_index"
_REQUIRED_ONE_OF = ("eigenfrequencies", "eigenvalues")
_OPTIONAL_CHANNELS = ("eigenfrequencies", "eigenvalues", "eigenvectors")
_DEFAULT_OUTPUT_ROOT = Path("_runs/gv/paper_replay/eigenmodes")
_PRESSURE_ALIASES = {
    "pressure_difference": "bpress",
    "pressure": "bpress",
    "background_pressure": "bpress",
}
_ZERO_ALLOWED_MATERIAL_PARAMETERS = frozenset({"a3", "a4", "b1", "b2"})
_MATERIAL_ALIASES = {"muL": "mu_l"}


@dataclass(frozen=True)
class EigenmodesPaperReplayPlan:
    figure_id: str
    experiment: str
    geometry: dict[str, float]
    material_parameters: dict[str, float]
    controls: dict[str, float]
    runtime_options: GVRuntimeOptions
    mode_count: int

    def to_manifest(self) -> dict[str, object]:
        return {
            "figure_id": self.figure_id,
            "experiment": self.experiment,
            "geometry": dict(self.geometry),
            "material_parameters": dict(self.material_parameters),
            "controls": dict(self.controls),
            "runtime_options": {
                "output_root": str(self.runtime_options.output_root),
                "controls": dict(self.runtime_options.controls),
                "timeout_seconds": self.runtime_options.timeout_seconds,
            },
            "mode_count": self.mode_count,
        }


@dataclass(frozen=True)
class EigenmodesPaperReplayResult:
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


def plan_eigenmodes_paper_replay_lane(
    *,
    material_parameters: Mapping[str, object],
    geometry: Mapping[str, object] | None = None,
    radGV: float | None = None,
    height: float | None = None,
    bpress: float = -91.0,
    mode_count: int = 30,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    timeout_seconds: int = 7200,
) -> EigenmodesPaperReplayPlan:
    validated_geometry = validate_geometry(geometry, radGV=radGV, height=height)
    validated_materials = _validate_paper_replay_material_parameters(material_parameters)
    pressure = _coerce_scalar(bpress, name="bpress")
    resolved_mode_count = int(mode_count)
    if resolved_mode_count <= 0:
        raise ValueError("Eigenmodes paper replay mode_count must be positive.")
    validated_output_root = _validate_runs_root(output_root)
    runtime_options = GVRuntimeOptions(
        controls={"bpress": pressure},
        output_root=str(validated_output_root),
        timeout_seconds=timeout_seconds,
    )
    return EigenmodesPaperReplayPlan(
        figure_id=EIGENMODES_FIGURE_ID,
        experiment="eigenmodes",
        geometry={"radGV": validated_geometry.radGV, "height": validated_geometry.height},
        material_parameters=validated_materials,
        controls={"bpress": pressure},
        runtime_options=runtime_options,
        mode_count=resolved_mode_count,
    )


def run_eigenmodes_paper_replay_lane(
    plan: EigenmodesPaperReplayPlan,
    *,
    sampler: Callable[..., object] = sample_gv,
) -> EigenmodesPaperReplayResult:
    sampling_controls = {"bpress": (plan.controls["bpress"],)}
    sample_result = sampler(
        experiment=plan.experiment,
        material_parameters=plan.material_parameters,
        geometry=plan.geometry,
        controls=sampling_controls,
        runtime_options=plan.runtime_options,
        write_artifacts=False,
    )
    return postprocess_eigenmodes_paper_replay_lane(sample_result, plan=plan)


def postprocess_eigenmodes_paper_replay_lane(
    sample_result: object,
    *,
    plan: EigenmodesPaperReplayPlan | None = None,
) -> EigenmodesPaperReplayResult:
    payload = _coerce_mapping(sample_result, context="sample_result")
    mode_count = plan.mode_count if plan is not None else 30
    try:
        channels = _normalize_eigenmode_channels(payload.get("channels"), mode_count=mode_count)
    except ValueError as exc:
        if "requires eigenvalues or eigenfrequencies" in str(exc):
            raise ValueError(
                "Eigenmodes paper replay requires at least one spectral channel: "
                + ", ".join(_REQUIRED_ONE_OF)
                + "."
            ) from exc
        raise

    if not any(name in channels for name in _REQUIRED_ONE_OF):
        raise ValueError(
            "Eigenmodes paper replay requires at least one spectral channel: "
            + ", ".join(_REQUIRED_ONE_OF)
            + "."
        )
    if _EIGENMODES_AXIS not in channels:
        raise ValueError("Eigenmodes paper replay requires channel 'mode_index'.")

    figure_id = plan.figure_id if plan is not None else EIGENMODES_FIGURE_ID
    experiment = str(payload.get("experiment", plan.experiment if plan is not None else "eigenmodes"))
    geometry = _resolve_geometry(payload, plan)
    material_parameters = _resolve_material_parameters(payload, plan)
    controls = _resolve_controls(payload, plan)
    provenance = _resolve_provenance(payload, plan)
    provenance["required_channels"] = list(_REQUIRED_ONE_OF)

    return EigenmodesPaperReplayResult(
        figure_id=figure_id,
        experiment=experiment,
        axis=_EIGENMODES_AXIS,
        controls=controls,
        geometry=geometry,
        material_parameters=material_parameters,
        channels=channels,
        provenance=provenance,
    )


def plot_eigenmodes_paper_replay(
    result: EigenmodesPaperReplayResult,
    *,
    output_path: str | Path,
) -> Path:
    target = _validate_runs_output_path(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spectrum_name = "eigenfrequencies" if "eigenfrequencies" in result.channels else "eigenvalues"
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        result.channels["mode_index"],
        result.channels[spectrum_name],
        marker="o",
        linewidth=1.6,
        color="#c92a2a",
    )
    axis.set_xlabel("Mode index")
    axis.set_ylabel(spectrum_name.replace("_", " ").title())
    axis.set_title(f"GV eigenmodes paper replay: {result.figure_id}")
    axis.grid(True, alpha=0.3)
    figure.text(
        0.02,
        0.02,
        f"experiment={result.experiment} | bpress={result.controls['bpress']} | modes={len(result.channels['mode_index'])}",
        fontsize=8,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _normalize_eigenmode_channels(channels_like: object, *, mode_count: int) -> dict[str, np.ndarray]:
    payload = _coerce_mapping(channels_like, context="sample_result.channels")
    parsed = parse_eigenmodes_lane_channels(
        {"channels": dict(payload)},
        mode_count=mode_count,
        include_eigenvectors=True,
    )
    normalized: dict[str, np.ndarray] = {}
    normalized[_EIGENMODES_AXIS] = np.asarray(parsed[_EIGENMODES_AXIS], dtype=float)
    for name in _OPTIONAL_CHANNELS:
        if name in parsed:
            normalized[name] = np.asarray(parsed[name], dtype=float)
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
    plan: EigenmodesPaperReplayPlan | None,
) -> dict[str, float]:
    geometry_like = payload.get("geometry", {})
    if isinstance(geometry_like, Mapping) and {"radGV", "height"} <= set(geometry_like):
        return {
            "radGV": _coerce_scalar(geometry_like["radGV"], name="geometry.radGV"),
            "height": _coerce_scalar(geometry_like["height"], name="geometry.height"),
        }
    if plan is not None:
        return dict(plan.geometry)
    raise ValueError("Eigenmodes paper replay requires geometry metadata.")


def _resolve_material_parameters(
    payload: Mapping[str, Any],
    plan: EigenmodesPaperReplayPlan | None,
) -> dict[str, float]:
    if isinstance(payload.get("material_parameters"), Mapping):
        return _validate_paper_replay_material_parameters(payload["material_parameters"])
    if plan is not None:
        return dict(plan.material_parameters)
    raise ValueError("Eigenmodes paper replay requires material_parameters metadata.")


def _resolve_controls(
    payload: Mapping[str, Any],
    plan: EigenmodesPaperReplayPlan | None,
) -> dict[str, float]:
    controls_like = payload.get("controls", {})
    normalized: dict[str, float] = {}
    if isinstance(controls_like, Mapping):
        for raw_name, value in controls_like.items():
            canonical_name = _PRESSURE_ALIASES.get(str(raw_name), str(raw_name))
            if canonical_name == "bpress":
                normalized[canonical_name] = _coerce_scalar(value, name="controls.bpress")
    if plan is not None:
        normalized.setdefault("bpress", plan.controls["bpress"])
    if "bpress" not in normalized:
        raise ValueError("Eigenmodes paper replay requires scalar control 'bpress'.")
    return normalized


def _resolve_provenance(
    payload: Mapping[str, Any],
    plan: EigenmodesPaperReplayPlan | None,
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
    "EIGENMODES_FIGURE_ID",
    "EigenmodesPaperReplayPlan",
    "EigenmodesPaperReplayResult",
    "plan_eigenmodes_paper_replay_lane",
    "plot_eigenmodes_paper_replay",
    "postprocess_eigenmodes_paper_replay_lane",
    "run_eigenmodes_paper_replay_lane",
]
