from __future__ import annotations

import hashlib
import json
import os
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
from ..sampling.extraction import extract_sampling_channels, merge_sampling_channels
from ..sampling.planner import build_sampling_plan
from ..sampling.types import GVMaterialGeometry, GVSweep

_DEFAULT_OUTPUT_ROOT = Path("_runs/gv/paper_replay/stretching")
_PAPER_EXACT_TOTAL_FORCE = tuple(np.linspace(500.0, 50_000.0, 80))
_DEFAULT_TOTAL_FORCE = tuple(np.linspace(500.0, 50_000.0, 9))
_DEFAULT_BPRESS = -91.0
_PAPER_PROTOCOL = {
    "radGV": 2.0,
    "height": 14.28,
    "dt": 5.0e-05,
    "numsteps": 20_000,
    "nevery": 200,
    "equilibration_runs": 1,
    "production_runs": 80,
    "particle_band_fraction": 0.15,
}


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
class StretchingPaperReplayPlan:
    campaign_id: str
    geometry_radius: float
    geometry_height: float
    material_parameters: Mapping[str, float]
    controls: Mapping[str, Any]
    paper_exact: bool = False
    output_root: Path = _DEFAULT_OUTPUT_ROOT
    include_plot: bool = True
    raw_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", _normalize_output_root(self.output_root))
        object.__setattr__(self, "material_parameters", _coerce_float_mapping(_coerce_mapping(self.material_parameters, name="material_parameters"), name="material_parameters"))
        object.__setattr__(self, "controls", _coerce_control_mapping(_coerce_mapping(self.controls, name="controls")))
        if "tot_force" not in self.controls or "bpress" not in self.controls:
            raise ValueError("Stretching paper replay requires controls tot_force and bpress.")
        object.__setattr__(self, "raw_provenance", _coerce_mapping(self.raw_provenance, name="raw_provenance"))

    def sampling_kwargs(self) -> dict[str, Any]:
        controls = dict(self.controls)
        return {
            "experiment": "stretching",
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
class StretchingPaperReplayResult:
    plan: StretchingPaperReplayPlan
    manifest: Mapping[str, Any]
    channels: Mapping[str, np.ndarray]
    summary: Mapping[str, Any]
    raw_sample_result: Mapping[str, Any]
    plot_path: Path | None = None
    plot_metadata_path: Path | None = None


def plan_stretching_paper_replay_lane(
    *,
    campaign_id: str,
    geometry_radius: float,
    geometry_height: float,
    material_parameters: Mapping[str, float],
    controls: Mapping[str, Any] | None = None,
    paper_exact: bool = False,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    include_plot: bool = True,
    raw_provenance: Mapping[str, Any] | None = None,
    point_start: int | None = None,
    point_stop: int | None = None,
) -> StretchingPaperReplayPlan:
    resolved_controls = {
        "tot_force": _PAPER_EXACT_TOTAL_FORCE if paper_exact else _DEFAULT_TOTAL_FORCE,
        "bpress": _DEFAULT_BPRESS,
        **dict(controls or {}),
    }
    if point_start is not None or point_stop is not None:
        resolved_controls["tot_force"] = _slice_point_controls(
            resolved_controls["tot_force"],
            point_start=point_start,
            point_stop=point_stop,
        )
    return StretchingPaperReplayPlan(
        campaign_id=campaign_id,
        geometry_radius=float(geometry_radius),
        geometry_height=float(geometry_height),
        material_parameters=material_parameters,
        controls=resolved_controls,
        paper_exact=paper_exact,
        output_root=output_root,
        include_plot=include_plot,
        raw_provenance=raw_provenance,
    )


def _slice_point_controls(
    values: Any,
    *,
    point_start: int | None,
    point_stop: int | None,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in _coerce_control_value(values, name="tot_force"))  # type: ignore[arg-type]
    start = 0 if point_start is None else int(point_start)
    stop = len(normalized) if point_stop is None else int(point_stop)
    if start < 0 or stop < 0:
        raise ValueError("Stretching point bounds must be >= 0.")
    if stop <= start:
        raise ValueError("Stretching point bounds must satisfy stop > start.")
    sliced = normalized[start:stop]
    if not sliced:
        raise ValueError("Stretching point bounds selected no control points.")
    return sliced


def _extract_channel_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    channels = payload.get("channels", payload)
    if not isinstance(channels, Mapping):
        raise ValueError("Stretching paper replay requires a channel mapping.")
    return {str(name): value for name, value in channels.items()}


def _require_same_shape(channels: Mapping[str, np.ndarray], names: tuple[str, ...]) -> None:
    shapes = {name: np.asarray(channels[name], dtype=float).shape for name in names if name in channels}
    if len(set(shapes.values())) > 1:
        raise ValueError(
            "Stretching paper replay requires channels "
            + ", ".join(names)
            + " to share the same shape."
        )


def _first_available(channels: Mapping[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in channels:
            return np.asarray(channels[name], dtype=float)
    return None


def _normalize_stretching_channels(
    raw_channels: Mapping[str, Any],
    *,
    geometry_radius: float,
) -> dict[str, np.ndarray]:
    channels = validate_numeric_channels(raw_channels)
    normalized = dict(channels)

    if "mean_length" in normalized and "mean_radius" in normalized:
        mean_length = np.asarray(normalized["mean_length"], dtype=float)
        mean_radius = np.asarray(normalized["mean_radius"], dtype=float)
        if mean_length.ndim == 0:
            mean_length = mean_length.reshape(1)
        if mean_radius.ndim == 0:
            mean_radius = mean_radius.reshape(1)
        if mean_length.shape != mean_radius.shape:
            raise ValueError("Stretching mean_length and mean_radius must share the same shape.")
        if mean_length[0] <= 0.0 or mean_radius[0] <= 0.0:
            raise ValueError("Stretching paper replay baseline length/radius must be positive.")
        normalized["epsilon_zz"] = (mean_length - mean_length[0]) / mean_length[0]
        epsilon_phi = (mean_radius - mean_radius[0]) / mean_radius[0]
        normalized["epsilon_phi"] = epsilon_phi
        normalized["epsilon_phi_plot"] = -epsilon_phi
        normalized["minus_epsilon_phi"] = -epsilon_phi
        if "std_length" in normalized:
            normalized["std_epsilon_zz"] = np.asarray(normalized["std_length"], dtype=float) / mean_length[0]
        if "std_radius" in normalized:
            normalized["std_epsilon_phi"] = np.asarray(normalized["std_radius"], dtype=float) / mean_radius[0]

    if "sigma_zz" not in normalized:
        if "tot_force" not in normalized:
            raise ValueError("Stretching paper replay requires sigma_zz or tot_force.")
        normalized["sigma_zz"] = np.asarray(normalized["tot_force"], dtype=float) / (
            2.0 * np.pi * float(geometry_radius)
        )

    if "epsilon_zz" not in normalized:
        epsilon_zz = _first_available(
            normalized,
            ("longitudinal_strain", "strain_z", "axial_strain"),
        )
        if epsilon_zz is None:
            raise ValueError("Stretching paper replay requires epsilon_zz.")
        normalized["epsilon_zz"] = epsilon_zz

    if "epsilon_phi_plot" not in normalized and "minus_epsilon_phi" in normalized:
        normalized["epsilon_phi_plot"] = np.asarray(normalized["minus_epsilon_phi"], dtype=float)
    if "minus_epsilon_phi" not in normalized and "epsilon_phi_plot" in normalized:
        normalized["minus_epsilon_phi"] = np.asarray(normalized["epsilon_phi_plot"], dtype=float)
    if "epsilon_phi_plot" not in normalized:
        epsilon_phi = _first_available(
            normalized,
            ("epsilon_phi", "circumferential_strain", "hoop_strain", "strain_phi"),
        )
        if epsilon_phi is None:
            raise ValueError(
                "Stretching paper replay requires epsilon_phi_plot or minus_epsilon_phi."
            )
        plot_ready = -np.asarray(epsilon_phi, dtype=float)
        normalized["epsilon_phi_plot"] = plot_ready
        normalized["minus_epsilon_phi"] = plot_ready

    _require_same_shape(
        normalized,
        ("sigma_zz", "epsilon_zz", "epsilon_phi_plot", "minus_epsilon_phi"),
    )

    if "std_epsilon_zz" in normalized:
        _require_same_shape(normalized, ("sigma_zz", "std_epsilon_zz"))
    if "std_epsilon_phi" in normalized:
        _require_same_shape(normalized, ("sigma_zz", "std_epsilon_phi"))
    if "tot_force" in normalized:
        _require_same_shape(normalized, ("sigma_zz", "tot_force"))
    if "force" in normalized and "displacement" in normalized:
        _require_same_shape(normalized, ("force", "displacement"))

    return normalized


def postprocess_stretching_paper_replay_lane(
    sample_result: object,
    *,
    plan: StretchingPaperReplayPlan,
) -> StretchingPaperReplayResult:
    payload = _result_payload(sample_result)
    channels = _normalize_stretching_channels(
        _extract_channel_mapping(payload),
        geometry_radius=plan.geometry_radius,
    )
    sigma_zz = np.asarray(channels["sigma_zz"], dtype=float)
    epsilon_zz = np.asarray(channels["epsilon_zz"], dtype=float)
    minus_epsilon_phi = np.asarray(channels["minus_epsilon_phi"], dtype=float)

    geometry = build_geometry(
        radius=plan.geometry_radius,
        height=plan.geometry_height,
        source="meso_uq.structures.gv.paper_replay_lanes.stretching",
    )
    control_id = _control_identifier(plan.controls)
    manifest = {
        "manifest_schema_version": 1,
        "structure": "gv",
        "experiment": "stretching",
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
            "lane": "stretching",
            "qualitative_target": "paper_figure_3_stress_strain",
            "plot_requested": plan.include_plot,
            "paper_exact": plan.paper_exact,
            "paper_protocol_reference_only": True,
            "paper_protocol_reference_source": "dropped_gv_paper_scripts/stretching/gv_stretching.zip",
            "canonical_replay_data_source": "Mirheo reruns",
            "paper_protocol": dict(_PAPER_PROTOCOL),
            "sample_manifest": payload.get("manifest"),
            **dict(plan.raw_provenance or {}),
        },
        "dataset_id": canonical_dataset_id("gv", "stretching", geometry.id, control_id),
    }
    summary = {
        "lane": "stretching",
        "axis_channel": "epsilon_zz",
        "auxiliary_axis_channel": "minus_epsilon_phi",
        "response_channel": "sigma_zz",
        "sample_count": int(sigma_zz.size),
        "epsilon_zz_min": float(np.min(epsilon_zz)),
        "epsilon_zz_max": float(np.max(epsilon_zz)),
        "minus_epsilon_phi_min": float(np.min(minus_epsilon_phi)),
        "minus_epsilon_phi_max": float(np.max(minus_epsilon_phi)),
        "sigma_zz_min": float(np.min(sigma_zz)),
        "sigma_zz_max": float(np.max(sigma_zz)),
        "has_auxiliary_force_displacement": "force" in channels and "displacement" in channels,
        "available_channels": tuple(sorted(channels)),
    }
    return StretchingPaperReplayResult(
        plan=plan,
        manifest=manifest,
        channels=channels,
        summary=summary,
        raw_sample_result=_jsonable(payload),
    )


def plot_stretching_paper_replay_lane(
    result: StretchingPaperReplayResult,
    *,
    reference_curve: Mapping[str, Any] | None = None,
) -> StretchingPaperReplayResult:
    pyplot = _import_matplotlib_pyplot()
    plots_dir = result.plan.output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = _short_plot_slug(result.manifest["dataset_id"])
    plot_path = plots_dir / f"{dataset_slug}.png"
    metadata_path = plots_dir / f"{dataset_slug}.json"

    has_auxiliary_panel = "force" in result.channels and "displacement" in result.channels
    if has_auxiliary_panel:
        figure, axes = pyplot.subplots(1, 2, figsize=(12.0, 5.0))
        main_axis, auxiliary_axis = axes
    else:
        figure, main_axis = pyplot.subplots(figsize=(8.0, 5.0))
        auxiliary_axis = None

    sigma_zz = np.asarray(result.channels["sigma_zz"], dtype=float)
    epsilon_zz = np.asarray(result.channels["epsilon_zz"], dtype=float)
    epsilon_phi_plot = np.asarray(result.channels["epsilon_phi_plot"], dtype=float)
    theory = _resolve_stretching_theory(result)
    if theory is not None:
        x = np.linspace(-0.001, 0.022, 100)
        main_axis.plot(
            x,
            theory["el_fac"] * x,
            color="royalblue",
            linestyle="-",
            linewidth=2.0,
            label=r"Theory: $\sigma_{zz}=E_l\varepsilon_{zz}$",
        )
        main_axis.plot(
            x,
            theory["el_fac"] / theory["poisson"] * x,
            color="crimson",
            linestyle="-",
            linewidth=2.0,
            label=r"Theory: $\sigma_{zz}=-E_l/\nu_{lt}\varepsilon_{\varphi\varphi}$",
        )

    xerr_zz = result.channels.get("std_epsilon_zz")
    xerr_phi = result.channels.get("std_epsilon_phi")
    main_axis.errorbar(
        epsilon_zz,
        sigma_zz,
        xerr=np.asarray(xerr_zz, dtype=float) if xerr_zz is not None else None,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linestyle="None",
        color="royalblue",
        capsize=2,
        label="Numerical",
    )
    main_axis.errorbar(
        epsilon_phi_plot,
        sigma_zz,
        xerr=np.asarray(xerr_phi, dtype=float) if xerr_phi is not None else None,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linestyle="None",
        color="crimson",
        capsize=2,
        label="Numerical",
    )

    if reference_curve is not None:
        ref = validate_numeric_channels(reference_curve)
        has_stress_strain_reference = (
            "sigma_zz" in ref
            and "epsilon_zz" in ref
            and ("epsilon_phi_plot" in ref or "minus_epsilon_phi" in ref)
        )
        has_force_displacement_reference = "displacement" in ref and "force" in ref
        if not has_stress_strain_reference and not has_force_displacement_reference:
            raise ValueError(
                "Stretching reference_curve requires stress-strain channels or displacement and force channels."
            )
        if has_stress_strain_reference:
            ref_phi = np.asarray(
                ref.get("epsilon_phi_plot", ref.get("minus_epsilon_phi")),
                dtype=float,
            )
            main_axis.plot(
                ref["epsilon_zz"],
                ref["sigma_zz"],
                linestyle="--",
                linewidth=1.5,
                label="Reference: sigma_zz(epsilon_zz)",
            )
            main_axis.plot(
                ref_phi,
                ref["sigma_zz"],
                linestyle=":",
                linewidth=1.5,
                label="Reference: sigma_zz(-epsilon_phi)",
            )
        if has_force_displacement_reference and auxiliary_axis is not None:
            auxiliary_axis.plot(
                ref["displacement"],
                ref["force"],
                linestyle="--",
                linewidth=1.5,
                label="Reference",
            )

    main_axis.set_xlabel(r"$\varepsilon_{zz}, -\varepsilon_{\varphi\varphi}$")
    main_axis.set_ylabel(r"$\sigma_{zz}$ [$k_BT_0/r_c^2$]")
    main_axis.set_title("GV stretching stress-strain replay")
    main_axis.set_xlim(-0.001, 0.022)
    main_axis.set_ylim(-25.0, max(1.0, 0.45 * float(np.max(sigma_zz))))
    main_axis.grid(True, alpha=0.25)
    main_axis.legend(loc="best")
    if auxiliary_axis is not None:
        auxiliary_axis.plot(
            np.asarray(result.channels["displacement"], dtype=float),
            np.asarray(result.channels["force"], dtype=float),
            marker="o",
            linewidth=2.0,
            label="DPD replay",
        )
        auxiliary_axis.set_xlabel("Displacement")
        auxiliary_axis.set_ylabel("Force")
        auxiliary_axis.set_title("Auxiliary Force-Displacement")
        auxiliary_axis.grid(True, alpha=0.25)
        auxiliary_axis.legend(loc="best")
    figure.suptitle(f"GV stretching paper replay: {result.plan.campaign_id}")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    pyplot.close(figure)

    metadata_path.write_text(
        json.dumps(
            {
                "manifest": _jsonable(result.manifest),
                "summary": _jsonable(result.summary),
                "plot_path": str(plot_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return StretchingPaperReplayResult(
        plan=result.plan,
        manifest=result.manifest,
        channels=result.channels,
        summary=result.summary,
        raw_sample_result=result.raw_sample_result,
        plot_path=plot_path,
        plot_metadata_path=metadata_path,
    )


def _resolve_stretching_theory(result: StretchingPaperReplayResult) -> dict[str, float] | None:
    parameters = _load_first_runtime_parameters(result)
    if parameters is None:
        return None
    defaults, resolved = parameters
    try:
        ul = float(defaults["ul"])
        ue = float(resolved.get("ue", defaults.get("ue", defaults["kbol"] * defaults["t0"])))
        shell_th = float(defaults.get("th_fac", 1.0)) * float(defaults["shell_th"])
        el_fac = float(defaults["fscale"]) * float(defaults["Yl"]) * shell_th / (ue / ul**2)
        poisson = float(defaults["nu"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if el_fac <= 0.0 or poisson == 0.0:
        return None
    return {"el_fac": el_fac, "poisson": poisson}


def _load_first_runtime_parameters(
    result: StretchingPaperReplayResult,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    raw_work_dirs = result.raw_sample_result.get("work_dirs", ())
    if not isinstance(raw_work_dirs, tuple | list):
        return None
    for raw_work_dir in raw_work_dirs:
        work_dir = Path(raw_work_dir)
        defaults_path = work_dir / "parameter" / "parameters-default00001.yaml"
        parameters_path = work_dir / "parameter" / "parameters00001.yaml"
        if not defaults_path.is_file() or not parameters_path.is_file():
            continue
        try:
            import yaml

            defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
            parameters = yaml.safe_load(parameters_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(defaults, Mapping) and isinstance(parameters, Mapping):
            return defaults, parameters
    return None


def run_stretching_paper_replay_lane(
    plan: StretchingPaperReplayPlan,
    *,
    sampling_callable: Callable[..., object] = sample_gv,
    reference_curve: Mapping[str, Any] | None = None,
) -> StretchingPaperReplayResult:
    if plan.paper_exact and sampling_callable is sample_gv:
        sample_result = _run_stretching_paper_exact_forward_sweep(plan)
    else:
        sample_result = sampling_callable(**plan.sampling_kwargs())
    result = postprocess_stretching_paper_replay_lane(sample_result, plan=plan)
    if not plan.include_plot:
        return result
    return plot_stretching_paper_replay_lane(result, reference_curve=reference_curve)


def _run_stretching_paper_exact_forward_sweep(plan: StretchingPaperReplayPlan) -> dict[str, Any]:
    tot_force_values = tuple(float(value) for value in _coerce_control_value(plan.controls["tot_force"], name="tot_force"))  # type: ignore[arg-type]
    bpress = float(_coerce_control_value(plan.controls["bpress"], name="bpress"))  # type: ignore[arg-type]
    geometry = build_geometry(
        radius=plan.geometry_radius,
        height=plan.geometry_height,
        source="meso_uq.structures.gv.paper_replay_lanes.stretching.forward_sweep",
    )
    timeout_seconds = GVRuntimeOptions().timeout_seconds
    material_env = {
        "MESOUQ_GV_PAPER_EXACT": "1",
        "MESOUQ_GV_MATERIAL_OVERRIDES_JSON": json.dumps(dict(plan.material_parameters), sort_keys=True),
    }
    geometry_payload = GVMaterialGeometry(radGV=plan.geometry_radius, height=plan.geometry_height)
    runtime_manifests: list[dict[str, Any]] = []
    plan_manifests: list[dict[str, Any]] = []
    execution_manifests: list[dict[str, Any]] = []
    work_dirs: list[Path] = []
    channel_sets: list[Mapping[str, np.ndarray]] = []
    started = time.perf_counter()

    for index, tot_force in enumerate(tot_force_values):
        runtime_controls = {"tot_force": tot_force}
        if bpress != _DEFAULT_BPRESS:
            runtime_controls["bpress"] = bpress
        runtime = plan_runtime(
            "stretching",
            output_root=plan.output_root,
            geometry=geometry.id,
            controls=runtime_controls,
            material_parameter_overrides=dict(plan.material_parameters),
            include_experimental=True,
        )
        work_dir = Path(runtime.work_dir)
        work_dirs.append(work_dir)
        runtime_manifests.append(runtime.to_manifest())

        existing_channels = _try_extract_existing_stretching_point(
            work_dir=work_dir,
            tot_force=tot_force,
            bpress=bpress,
            geometry=geometry_payload,
        )
        if existing_channels is not None:
            channel_sets.append(existing_channels)
            execution_manifests.append(
                {
                    "control_index": index,
                    "controls": {"tot_force": tot_force, "bpress": bpress},
                    "status": "reused_existing",
                    "work_dir": str(work_dir),
                }
            )
            continue

        sampling_plan = build_sampling_plan(
            runtime,
            control_axis="tot_force",
            values=(tot_force,),
            timeout_seconds=timeout_seconds,
        )
        plan_manifests.append(sampling_plan.to_manifest())
        execution = execute_sampling_plan(
            sampling_plan,
            timeout_seconds=timeout_seconds,
            env=material_env,
        )
        point_channels = extract_sampling_channels(
            experiment="stretching",
            work_dir=work_dir,
            controls={"tot_force": tot_force, "bpress": bpress},
            sweep=GVSweep("tot_force", (tot_force,)),
            geometry=geometry_payload,
        )
        channel_sets.append(point_channels)
        execution_manifests.append(
            {
                "control_index": index,
                "controls": {"tot_force": tot_force, "bpress": bpress},
                "status": "completed",
                "work_dir": str(work_dir),
                "executed_commands": list(execution.executed_commands),
                "return_codes": list(execution.return_codes),
            }
        )

    runtime_seconds = time.perf_counter() - started
    channels = merge_sampling_channels(tuple(channel_sets))
    return {
        "experiment": "stretching",
        "geometry": {"radGV": plan.geometry_radius, "height": plan.geometry_height},
        "material_parameters": dict(plan.material_parameters),
        "controls": {"tot_force": tot_force_values, "bpress": bpress},
        "channels": channels,
        "manifest": None,
        "runtime_manifests": tuple(runtime_manifests),
        "plan_manifests": tuple(plan_manifests),
        "execution_manifests": tuple(execution_manifests),
        "work_dirs": tuple(work_dirs),
        "runtime_seconds": runtime_seconds,
        "status": "completed",
    }


def _try_extract_existing_stretching_point(
    *,
    work_dir: Path,
    tot_force: float,
    bpress: float,
    geometry: GVMaterialGeometry,
) -> dict[str, np.ndarray] | None:
    if not work_dir.is_dir():
        return None
    try:
        return extract_sampling_channels(
            experiment="stretching",
            work_dir=work_dir,
            controls={"tot_force": tot_force, "bpress": bpress},
            sweep=GVSweep("tot_force", (tot_force,)),
            geometry=geometry,
        )
    except Exception:
        return None


__all__ = [
    "StretchingPaperReplayPlan",
    "StretchingPaperReplayResult",
    "plan_stretching_paper_replay_lane",
    "plot_stretching_paper_replay_lane",
    "postprocess_stretching_paper_replay_lane",
    "run_stretching_paper_replay_lane",
]
