from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from meso_uq.experiments import canonical_dataset_id
from meso_uq.references.gv_common import format_float

from .. import build_geometry
from ..postprocessing.common import validate_numeric_channels
from ..runtime import plan_runtime
from ..sampling import GVRuntimeOptions, sample_gv
from ..sampling.executor import execute_sampling_plan
from ..sampling.extraction import extract_sampling_channels
from ..sampling.planner import build_sampling_plan
from ..sampling.torsion import extract_torsion_lane, reconstruct_torsion_anchor_geometry
from ..sampling.types import GVMaterialGeometry, GVSweep

_DEFAULT_OUTPUT_ROOT = Path("_runs/gv/paper_replay/torsion")
_DEFAULT_THETA = (0.01, 0.03, 0.05, 0.075, 0.1)
_PAPER_EXACT_THETA = tuple(np.round(np.linspace(0.01, 0.10, 10), 2))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.number, np.bool_)):
        return value.item()
    return value


def _normalize_output_root(output_root: str | Path) -> Path:
    path = Path(output_root)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Paper replay outputs must live under _runs/.")
    if path.parts[:1] != ("_runs",):
        raise ValueError("Paper replay outputs must live under _runs/.")
    return path


def _coerce_mapping(payload: object, *, name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return {str(key): value for key, value in payload.items()}


def _coerce_float_mapping(payload: Mapping[str, Any], *, name: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in payload.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{key!r}] must be numeric.") from exc
        if not np.isfinite(number):
            raise ValueError(f"{name}[{key!r}] must be finite.")
        normalized[str(key)] = number
    return normalized


def _is_sequence_control(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, (str, bytes, bytearray))


def _coerce_control_value(value: Any, *, name: str) -> float | tuple[float, ...]:
    if _is_sequence_control(value):
        values = tuple(float(item) for item in value)
        if not values:
            raise ValueError(f"controls[{name!r}] must be a non-empty sweep.")
        if not all(np.isfinite(item) for item in values):
            raise ValueError(f"controls[{name!r}] must contain only finite values.")
        return values
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"controls[{name!r}] must be numeric.") from exc
    if not np.isfinite(number):
        raise ValueError(f"controls[{name!r}] must be finite.")
    return number


def _coerce_control_mapping(payload: Mapping[str, Any]) -> dict[str, float | tuple[float, ...]]:
    return {str(key): _coerce_control_value(value, name=str(key)) for key, value in payload.items()}


def _representative_controls(payload: Mapping[str, Any]) -> dict[str, float]:
    representative: dict[str, float] = {}
    for key, value in payload.items():
        normalized = _coerce_control_value(value, name=str(key))
        if isinstance(normalized, tuple):
            representative[str(key)] = float(normalized[0])
        else:
            representative[str(key)] = float(normalized)
    return representative


def _control_identifier(payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for name, value in sorted(payload.items()):
        normalized = _coerce_control_value(value, name=str(name))
        if isinstance(normalized, tuple):
            value_token = "sweep_" + "_".join(format_float(item) for item in normalized)
        else:
            value_token = format_float(normalized)
        parts.append(f"{name}_{value_token}")
    return "__".join(parts) or "default"


def _result_payload(sample_result: object) -> dict[str, Any]:
    if isinstance(sample_result, Mapping):
        return dict(sample_result)
    payload: dict[str, Any] = {}
    for name in (
        "channels",
        "controls",
        "manifest",
        "geometry",
        "experiment",
        "runtime_manifests",
        "plan_manifests",
        "execution_manifests",
        "work_dirs",
        "runtime_seconds",
        "status",
    ):
        if hasattr(sample_result, name):
            payload[name] = getattr(sample_result, name)
    geometry = payload.get("geometry")
    if hasattr(geometry, "radGV") and hasattr(geometry, "height"):
        payload["geometry"] = {"radGV": geometry.radGV, "height": geometry.height}
    if not payload:
        raise ValueError("sample_result must be a mapping or expose lane-like attributes.")
    return payload


def _import_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        return pyplot
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency-dependent
        raise RuntimeError(
            "matplotlib is required to generate GV paper replay plots."
        ) from exc


def _short_plot_slug(dataset_id: object) -> str:
    slug = str(dataset_id).replace(":", "__")
    if len(slug) <= 120:
        return slug
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:96]}__{digest}"


@dataclass(frozen=True)
class TorsionPaperReplayPlan:
    campaign_id: str
    geometry_radius: float
    geometry_height: float
    material_parameters: Mapping[str, float]
    controls: Mapping[str, Any]
    output_root: Path = _DEFAULT_OUTPUT_ROOT
    include_plot: bool = True
    paper_exact: bool = False
    raw_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", _normalize_output_root(self.output_root))
        object.__setattr__(self, "material_parameters", _coerce_float_mapping(_coerce_mapping(self.material_parameters, name="material_parameters"), name="material_parameters"))
        object.__setattr__(self, "controls", _coerce_control_mapping(_coerce_mapping(self.controls, name="controls")))
        if "theta" not in self.controls:
            raise ValueError("Torsion paper replay requires control theta.")
        object.__setattr__(self, "raw_provenance", _coerce_mapping(self.raw_provenance, name="raw_provenance"))

    def sampling_kwargs(self) -> dict[str, Any]:
        controls = dict(self.controls)
        return {
            "experiment": "torsion",
            "campaign_id": self.campaign_id,
            "geometry": {"radGV": self.geometry_radius, "height": self.geometry_height},
            "material_parameters": dict(self.material_parameters),
            "controls": controls,
            "runtime_options": GVRuntimeOptions(
                controls=controls,
                output_root=str(self.output_root),
            ),
            "write_artifacts": False,
        }


@dataclass(frozen=True)
class TorsionPaperReplayResult:
    plan: TorsionPaperReplayPlan
    manifest: Mapping[str, Any]
    channels: Mapping[str, np.ndarray]
    summary: Mapping[str, Any]
    raw_sample_result: Mapping[str, Any]
    plot_path: Path | None = None
    plot_metadata_path: Path | None = None


def plan_torsion_paper_replay_lane(
    *,
    campaign_id: str,
    geometry_radius: float,
    geometry_height: float,
    material_parameters: Mapping[str, float],
    controls: Mapping[str, Any] | None = None,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    include_plot: bool = True,
    paper_exact: bool = False,
    raw_provenance: Mapping[str, Any] | None = None,
) -> TorsionPaperReplayPlan:
    default_theta = _PAPER_EXACT_THETA if paper_exact else _DEFAULT_THETA
    resolved_controls = {"theta": default_theta, **dict(controls or {})}
    return TorsionPaperReplayPlan(
        campaign_id=campaign_id,
        geometry_radius=float(geometry_radius),
        geometry_height=float(geometry_height),
        material_parameters=material_parameters,
        controls=resolved_controls,
        output_root=output_root,
        include_plot=include_plot,
        paper_exact=paper_exact,
        raw_provenance=raw_provenance,
    )


def postprocess_torsion_paper_replay_lane(
    sample_result: object,
    *,
    plan: TorsionPaperReplayPlan,
) -> TorsionPaperReplayResult:
    payload = _result_payload(sample_result)
    lane_controls = _representative_controls(plan.controls)
    _enforce_paper_raw_anchor_contract(payload, plan=plan, controls=lane_controls)
    lane = extract_torsion_lane(
        {
            "controls": lane_controls,
            "geometry": {"radius": plan.geometry_radius, "height": plan.geometry_height},
            "channels": payload.get("channels", payload),
        },
        controls=lane_controls,
        geometry={"radius": plan.geometry_radius, "height": plan.geometry_height},
    )
    channels = validate_numeric_channels(lane["channels"])
    gamma = channels["gamma"]
    sigma_phi_r = channels["sigma_phi_r"]
    if gamma.shape != sigma_phi_r.shape:
        raise ValueError("Torsion paper replay requires gamma and sigma_phi_r to share the same shape.")

    geometry = build_geometry(
        radius=plan.geometry_radius,
        height=plan.geometry_height,
        source="meso_uq.structures.gv.paper_replay_lanes.torsion",
    )
    control_id = _control_identifier(plan.controls)
    manifest = {
        "manifest_schema_version": 1,
        "structure": "gv",
        "experiment": "torsion",
        "campaign_id": plan.campaign_id,
        "geometry": geometry.id,
        "geometry_parameters": {str(name): float(value) for name, value in geometry.parameters.items()},
        "control_id": control_id,
        "controls": _jsonable(dict(plan.controls)),
        "material_parameters": dict(plan.material_parameters),
        "quality_flags": {
            "finite_observables": True,
            "finite_observable_ratio": 1.0,
            "canary_failures": [],
        },
        "raw_provenance": {
            "workflow": "gv_paper_replay_lane",
            "lane": "torsion",
            "qualitative_target": "paper_si_torsion_dpd_only",
            "paper_protocol": "mirheo_anchor_torque_reconstruction",
            "paper_exact": plan.paper_exact,
            "plot_requested": plan.include_plot,
            "sample_manifest": payload.get("manifest"),
            **dict(plan.raw_provenance or {}),
        },
        "dataset_id": canonical_dataset_id("gv", "torsion", geometry.id, control_id),
    }
    summary = {
        "lane": "torsion",
        "axis_channel": "gamma",
        "response_channel": "sigma_phi_r",
        "sample_count": int(gamma.size),
        "gamma_min": float(np.min(gamma)),
        "gamma_max": float(np.max(gamma)),
        "sigma_phi_r_min": float(np.min(sigma_phi_r)),
        "sigma_phi_r_max": float(np.max(sigma_phi_r)),
        "paper_exact": bool(plan.paper_exact),
        "available_channels": tuple(sorted(channels)),
    }
    return TorsionPaperReplayResult(
        plan=plan,
        manifest=manifest,
        channels=channels,
        summary=summary,
        raw_sample_result=_jsonable(payload),
    )


def plot_torsion_paper_replay_lane(
    result: TorsionPaperReplayResult,
) -> TorsionPaperReplayResult:
    pyplot = _import_matplotlib_pyplot()
    plots_dir = result.plan.output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = _short_plot_slug(result.manifest["dataset_id"])
    plot_path = plots_dir / f"{dataset_slug}.png"
    metadata_path = plots_dir / f"{dataset_slug}.json"

    figure, axis = pyplot.subplots(figsize=(7.2, 5.0))
    gamma = np.asarray(result.channels["gamma"], dtype=float)
    sigma_phi_r = np.asarray(result.channels["sigma_phi_r"], dtype=float)
    gamma_line = np.linspace(0.0, max(float(np.max(gamma)) * 1.1, 1.0e-12), 100)
    axis.plot(
        gamma_line,
        float(result.plan.material_parameters["mu"]) * gamma_line,
        linestyle="--",
        linewidth=2.0,
        color="crimson",
        label="Theory",
    )
    sigma_std = result.channels.get("sigma_phi_r_std", result.channels.get("sigma_std"))
    if sigma_std is not None and np.asarray(sigma_std, dtype=float).shape == sigma_phi_r.shape:
        axis.errorbar(
            gamma,
            sigma_phi_r,
            yerr=np.asarray(sigma_std, dtype=float),
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            linestyle="None",
            color="royalblue",
            capsize=2,
            label="Numerical",
        )
    else:
        axis.plot(
            gamma,
            sigma_phi_r,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            linestyle="None",
            color="royalblue",
            label="Numerical",
        )
    axis.set_xlabel(r"$\gamma$")
    axis.set_ylabel(r"$\sigma_{\varphi r}$ [$k_BT_0/r_c^2$]")
    axis.set_title("GV torsion paper replay")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    pyplot.close(figure)

    metadata_path.write_text(
        json.dumps(
            {
                "manifest": _jsonable(result.manifest),
                "summary": _jsonable(result.summary),
                "plot_path": str(plot_path),
                "plot_provenance": {
                    "x_label": "gamma = 2 * theta * radGV / dz",
                    "y_label": "sigma_phi_r = tau_z / (2 * pi * radGV^2)",
                    "response_definition": "average absolute top/bottom anchor stress over the last 25% timesteps",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TorsionPaperReplayResult(
        plan=result.plan,
        manifest=result.manifest,
        channels=result.channels,
        summary=result.summary,
        raw_sample_result=result.raw_sample_result,
        plot_path=plot_path,
        plot_metadata_path=metadata_path,
    )


def _enforce_paper_raw_anchor_contract(
    payload: Mapping[str, Any],
    *,
    plan: TorsionPaperReplayPlan,
    controls: Mapping[str, float],
) -> None:
    channels = payload.get("channels", payload)
    if not isinstance(channels, Mapping):
        return
    flattened = _flatten_channel_names(channels)
    canonical_names = {_canonical_channel_name(name) for name in flattened}
    has_canonical = {"gamma", "sigma_phi_r"} <= canonical_names
    has_paper_anchor_payload = bool(
        {
            "mesh_vertices",
            "constrained_vertex_forces_min",
            "constrained_vertex_forces_max",
        }
        & canonical_names
    )
    if not has_paper_anchor_payload or has_canonical:
        return
    reconstruct_torsion_anchor_geometry(
        {
            "controls": dict(controls),
            "geometry": {"radius": plan.geometry_radius, "height": plan.geometry_height},
            "channels": dict(channels),
        },
        geometry={"radius": plan.geometry_radius, "height": plan.geometry_height},
    )


def _flatten_channel_names(payload: Mapping[str, Any], *, prefix: str = "") -> tuple[str, ...]:
    names: list[str] = []
    for raw_name, value in payload.items():
        if not isinstance(raw_name, str):
            continue
        name = f"{prefix}_{raw_name}" if prefix else raw_name
        if isinstance(value, Mapping):
            names.extend(_flatten_channel_names(value, prefix=name))
        else:
            names.append(name)
    return tuple(names)


def _canonical_channel_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "anchor_min_forces": "constrained_vertex_forces_min",
        "anchor_min_force": "constrained_vertex_forces_min",
        "bottom_anchor_forces": "constrained_vertex_forces_min",
        "bottom_anchor_force": "constrained_vertex_forces_min",
        "anchor_max_forces": "constrained_vertex_forces_max",
        "anchor_max_force": "constrained_vertex_forces_max",
        "top_anchor_forces": "constrained_vertex_forces_max",
        "top_anchor_force": "constrained_vertex_forces_max",
        "vertices": "mesh_vertices",
    }
    return aliases.get(normalized, normalized)


def run_torsion_paper_replay_lane(
    plan: TorsionPaperReplayPlan,
    *,
    sampling_callable: Callable[..., object] = sample_gv,
) -> TorsionPaperReplayResult:
    if plan.paper_exact and sampling_callable is sample_gv:
        sample_result = _run_torsion_paper_exact_forward_sweep(plan)
    else:
        sample_result = sampling_callable(**plan.sampling_kwargs())
    result = postprocess_torsion_paper_replay_lane(sample_result, plan=plan)
    if not plan.include_plot:
        return result
    return plot_torsion_paper_replay_lane(result)


def _run_torsion_paper_exact_forward_sweep(plan: TorsionPaperReplayPlan) -> dict[str, Any]:
    theta_values = tuple(float(value) for value in _coerce_control_value(plan.controls["theta"], name="theta"))  # type: ignore[arg-type]
    geometry = build_geometry(
        radius=plan.geometry_radius,
        height=plan.geometry_height,
        source="meso_uq.structures.gv.paper_replay_lanes.torsion.forward_sweep",
    )
    runtime = plan_runtime(
        "torsion",
        output_root=plan.output_root,
        geometry=geometry.id,
        controls=None,
        material_parameter_overrides=dict(plan.material_parameters),
        include_experimental=True,
    )
    sampling_plan = build_sampling_plan(
        runtime,
        control_axis="theta",
        values=theta_values,
        timeout_seconds=GVRuntimeOptions().timeout_seconds,
    )
    started = time.perf_counter()
    execution = execute_sampling_plan(
        sampling_plan,
        timeout_seconds=GVRuntimeOptions().timeout_seconds,
        env={
            "MESOUQ_GV_MATERIAL_OVERRIDES_JSON": json.dumps(
                dict(plan.material_parameters),
                sort_keys=True,
            ),
        },
    )
    runtime_seconds = time.perf_counter() - started
    channels = extract_sampling_channels(
        experiment="torsion",
        work_dir=runtime.work_dir,
        controls={"theta": theta_values[0]},
        sweep=GVSweep("theta", theta_values),
        geometry=GVMaterialGeometry(radGV=plan.geometry_radius, height=plan.geometry_height),
    )
    return {
        "experiment": "torsion",
        "geometry": {"radGV": plan.geometry_radius, "height": plan.geometry_height},
        "material_parameters": dict(plan.material_parameters),
        "controls": {"theta": theta_values},
        "channels": channels,
        "manifest": None,
        "runtime_manifests": (runtime.to_manifest(),),
        "plan_manifests": (sampling_plan.to_manifest(),),
        "execution_manifests": (
            {
                "executed_commands": list(execution.executed_commands),
                "return_codes": list(execution.return_codes),
            },
        ),
        "work_dirs": (Path(runtime.work_dir),),
        "runtime_seconds": runtime_seconds,
        "status": "completed",
    }


__all__ = [
    "TorsionPaperReplayPlan",
    "TorsionPaperReplayResult",
    "plan_torsion_paper_replay_lane",
    "plot_torsion_paper_replay_lane",
    "postprocess_torsion_paper_replay_lane",
    "run_torsion_paper_replay_lane",
]
