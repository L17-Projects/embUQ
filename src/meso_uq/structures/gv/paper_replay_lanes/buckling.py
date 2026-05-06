from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from .. import build_geometry
from ..runtime import plan_runtime
from ..sampling import GVRuntimeOptions, sample_gv
from ..sampling.executor import execute_sampling_plan
from ..sampling.extraction import extract_sampling_channels
from ..sampling.planner import build_sampling_plan
from ..sampling.types import GVMaterialGeometry, GVSweep
from ..sampling.buckling import parse_buckling_lane_channels
from ..parameters import GV_MATERIAL_PARAMETER_NAMES
from ..sampling.validation import validate_geometry


BUCKLING_FIGURE_ID = "Figure 7"
_PLOT_AXIS = "pressure_difference"
_REQUIRED_CHANNELS = ("buck", "pressure_difference", "relative_volume")
_OPTIONAL_CHANNELS = (
    "bpress",
    "buckling_response",
    "force_response",
    "pressure_response",
    "shape_amplitude",
    "deformation_amplitude",
    "initial_volume",
    "mean_volume",
    "std_volume",
    "relative_volume_std",
    "analyzed_volume_frame_count",
)
_DEFAULT_OUTPUT_ROOT = Path("_runs/gv/paper_replay/buckling")
_CONTROL_ALIASES = {
    "pressure": "bpress",
    "background_pressure": "bpress",
}
_CHANNEL_ALIASES = {
    "delta_p": "pressure_difference",
    "delta_pressure": "pressure_difference",
    "pressure": "pressure_difference",
    "pressure_difference_proxy": "pressure_difference",
    "volumetric_strain": "relative_volume",
    "volume_strain": "relative_volume",
}
_ZERO_ALLOWED_MATERIAL_PARAMETERS = frozenset({"a3", "a4", "b1", "b2"})
_MATERIAL_ALIASES = {"muL": "mu_l"}
_DEFAULT_BPRESS = -91.0
_DEFAULT_BUCK_MAX = 0.75
_DEFAULT_PAPER_SWEEP_POINTS = 20
_EXACT_PAPER_SWEEP_POINTS = 25
_BUCK_TO_PRESSURE_SCALE = 100.0 * 3.0**2 * 0.101


@dataclass(frozen=True)
class BucklingPaperReplayPlan:
    figure_id: str
    experiment: str
    geometry: dict[str, float]
    material_parameters: dict[str, float]
    controls: dict[str, object]
    runtime_options: GVRuntimeOptions
    mapping_assumptions: tuple[str, ...]
    paper_exact: bool = False

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
            "paper_exact": self.paper_exact,
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
    pressure_differences: tuple[float, ...] | list[float] | None = None,
    buck: float | tuple[float, ...] | list[float] = _DEFAULT_BUCK_MAX,
    bpress: float = _DEFAULT_BPRESS,
    paper_exact: bool = False,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    timeout_seconds: int = 7200,
) -> BucklingPaperReplayPlan:
    validated_geometry = validate_geometry(geometry, radGV=radGV, height=height)
    validated_materials = _validate_paper_replay_material_parameters(material_parameters)
    buck_sweep = _resolve_buck_sweep(buck, paper_exact=paper_exact)
    if pressure_differences is None:
        pressures = _derive_pressure_difference(buck_sweep)
    else:
        pressures = _coerce_finite_1d(pressure_differences, name="pressure_differences")
        if pressures.size < 2:
            raise ValueError("Buckling paper replay requires at least two pressure-difference samples.")
        if pressures.shape != buck_sweep.shape:
            raise ValueError(
                "Buckling paper replay requires 'pressure_differences' to match the buck sweep length."
            )
    fixed_bpress = _coerce_scalar(bpress, name="bpress")
    validated_output_root = _validate_runs_root(output_root)
    runtime_options = GVRuntimeOptions(
        controls={"bpress": fixed_bpress},
        output_root=str(validated_output_root),
        timeout_seconds=timeout_seconds,
    )
    return BucklingPaperReplayPlan(
        figure_id=BUCKLING_FIGURE_ID,
        experiment="buckling",
        geometry={"radGV": validated_geometry.radGV, "height": validated_geometry.height},
        material_parameters=validated_materials,
        controls={
            "buck": tuple(float(value) for value in buck_sweep),
            "bpress": fixed_bpress,
            "pressure_difference": tuple(float(value) for value in pressures),
        },
        runtime_options=runtime_options,
        mapping_assumptions=(
            "Paper replay sweeps `buck` over the canonical Figure 7 range while holding `bpress=-91.0` fixed as a runtime control.",
            "Pressure difference is derived as Delta p = buck*aii*rhow**2*alpha with aii=100, rhow=3, alpha=0.101, so Delta p = 90.9*buck.",
            "Dropped scripts under gv_paper_scripts are protocol references only; canonical replay artifacts for this lane come from Mirheo reruns.",
        ),
        paper_exact=bool(paper_exact),
    )


def run_buckling_paper_replay_lane(
    plan: BucklingPaperReplayPlan,
    *,
    sampler: Callable[..., object] = sample_gv,
) -> BucklingPaperReplayResult:
    if plan.paper_exact and sampler is sample_gv:
        sample_result = _run_buckling_paper_exact_forward_sweep(plan)
    else:
        sampling_controls = {
            name: value
            for name, value in plan.controls.items()
            if name != "pressure_difference"
        }
        sample_result = sampler(
            experiment=plan.experiment,
            material_parameters=plan.material_parameters,
            geometry=plan.geometry,
            controls=sampling_controls,
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
    if plan is not None:
        channels.setdefault("buck", np.asarray(plan.controls["buck"], dtype=float))
        channels.setdefault(
            "pressure_difference",
            np.asarray(plan.controls["pressure_difference"], dtype=float),
        )

    missing = [name for name in _REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise ValueError(
            "Buckling paper replay requires channels: " + ", ".join(_REQUIRED_CHANNELS) + "."
        )

    if channels["buck"].shape != channels["pressure_difference"].shape:
        raise ValueError(
            "Buckling paper replay requires 'buck' and 'pressure_difference' to share a shape."
        )
    if channels["pressure_difference"].shape != channels["relative_volume"].shape:
        raise ValueError(
            "Buckling paper replay requires 'pressure_difference' and 'relative_volume' to share a shape."
        )

    figure_id = plan.figure_id if plan is not None else BUCKLING_FIGURE_ID
    experiment = str(payload.get("experiment", plan.experiment if plan is not None else "buckling"))
    geometry = _resolve_geometry(payload, plan)
    material_parameters = _resolve_material_parameters(payload, plan)
    controls = _resolve_controls(payload, plan, required=("bpress",))
    provenance = _resolve_provenance(payload, plan)
    provenance["required_channels"] = list(_REQUIRED_CHANNELS)
    provenance["mapping_assumptions"] = list(
        plan.mapping_assumptions
        if plan is not None
        else (
            "Dropped scripts are protocol references only; canonical replay data should come from Mirheo reruns.",
        )
    )
    provenance["protocol_reference"] = "Dropped scripts are protocol references only."
    provenance["canonical_replay_source"] = "Canonical replay data come from Mirheo reruns."

    return BucklingPaperReplayResult(
        figure_id=figure_id,
        experiment=experiment,
        axis=_PLOT_AXIS,
        controls=controls,
        geometry=geometry,
        material_parameters=material_parameters,
        channels=channels,
        provenance=provenance,
    )


def _run_buckling_paper_exact_forward_sweep(plan: BucklingPaperReplayPlan) -> dict[str, Any]:
    buck_values = tuple(float(value) for value in _coerce_finite_1d(plan.controls["buck"], name="buck"))
    bpress = _coerce_scalar(plan.controls["bpress"], name="bpress")
    geometry = build_geometry(
        radius=plan.geometry["radGV"],
        height=plan.geometry["height"],
        source="meso_uq.structures.gv.paper_replay_lanes.buckling.forward_sweep",
    )
    runtime = plan_runtime(
        "buckling",
        output_root=plan.runtime_options.output_root,
        geometry=geometry.id,
        controls=None,
        material_parameter_overrides=dict(plan.material_parameters),
        include_experimental=True,
    )
    sampling_plan = build_sampling_plan(
        runtime,
        control_axis="buck",
        values=buck_values,
        timeout_seconds=plan.runtime_options.timeout_seconds,
    )
    started = time.perf_counter()
    execution = execute_sampling_plan(
        sampling_plan,
        timeout_seconds=plan.runtime_options.timeout_seconds,
        env={
            "MESOUQ_GV_PAPER_EXACT": "1",
            "MESOUQ_GV_MATERIAL_OVERRIDES_JSON": json.dumps(
                dict(plan.material_parameters),
                sort_keys=True,
            ),
        },
    )
    runtime_seconds = time.perf_counter() - started
    channels = extract_sampling_channels(
        experiment="buckling",
        work_dir=runtime.work_dir,
        controls={"buck": buck_values[0], "bpress": bpress},
        sweep=GVSweep("buck", buck_values),
        geometry=GVMaterialGeometry(radGV=plan.geometry["radGV"], height=plan.geometry["height"]),
    )
    return {
        "experiment": "buckling",
        "geometry": dict(plan.geometry),
        "material_parameters": dict(plan.material_parameters),
        "controls": {"buck": buck_values, "bpress": bpress},
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


def plot_buckling_paper_replay(
    result: BucklingPaperReplayResult,
    *,
    output_path: str | Path,
) -> Path:
    target = _validate_runs_output_path(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pressure = np.asarray(result.channels["pressure_difference"], dtype=float)
    relative_volume = np.asarray(result.channels["relative_volume"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axis, zoom_axis = axes
    yerr = result.channels.get("relative_volume_std")
    yerr_array = (
        np.asarray(yerr, dtype=float)
        if yerr is not None and np.asarray(yerr, dtype=float).shape == relative_volume.shape
        else None
    )
    theory = _resolve_buckling_theory_slope(result)

    for selected_axis in (axis, zoom_axis):
        if theory is not None:
            x = np.linspace(float(np.min(pressure)), float(np.max(pressure)), 100)
            selected_axis.plot(
                x,
                1.0 + theory * x,
                linestyle="--",
                linewidth=2.0,
                color="crimson",
                label="Theory",
            )
        if yerr_array is not None:
            selected_axis.errorbar(
                pressure,
                relative_volume,
                yerr=yerr_array,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=1.5,
                linestyle="None",
                color="royalblue",
                capsize=2,
                label="Numerical",
            )
        else:
            selected_axis.plot(
                pressure,
                relative_volume,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=1.5,
                linestyle="None",
                color="royalblue",
                label="Numerical",
            )
        selected_axis.set_xlabel(r"$\Delta p$ [$k_BT_0/r_c^3$]")
        selected_axis.set_ylabel(r"$V/V_0$")
        selected_axis.grid(True, alpha=0.25)
        selected_axis.legend(loc="best")

    axis.axvline(x=31.0, color="black", linestyle="--", linewidth=2.0)
    zoom_axis.set_xlim(0.0, min(24.5, float(np.max(pressure))))
    zoom_axis.set_ylim(0.994, 1.0015)
    axis.set_title("Full pressure sweep")
    zoom_axis.set_title("Low-pressure zoom")
    figure.suptitle(f"GV buckling paper replay: {result.figure_id}")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=220)
    plt.close(figure)
    return target


def _resolve_buckling_theory_slope(result: BucklingPaperReplayResult) -> float | None:
    parameters = _load_first_runtime_parameters(result)
    if parameters is None:
        return None
    defaults, resolved = parameters
    try:
        ul = float(defaults["ul"])
        ue = float(resolved.get("ue", defaults.get("ue", defaults["kbol"] * defaults["t0"])))
        shell_th = float(defaults["shell_th"])
        fscale = float(defaults["fscale"])
        radgv = float(defaults.get("radGV", result.geometry["radGV"]))
        nu = float(defaults["nu"])
        yl = fscale * float(defaults["Yl"]) * shell_th / (ue / ul**2)
        yt = fscale * float(defaults["Yt"]) * shell_th / (ue / ul**2)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if yl <= 0.0 or yt <= 0.0:
        return None
    return -radgv / (2.0 * yl) * (1.0 - 4.0 * nu + 4.0 * yl / yt)


def _load_first_runtime_parameters(
    result: BucklingPaperReplayResult,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    for raw_work_dir in result.provenance.get("work_dirs", ()):
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


def _normalize_buckling_channels(channels_like: object) -> dict[str, np.ndarray]:
    payload = _coerce_mapping(channels_like, context="sample_result.channels")
    remapped: dict[str, Any] = {}
    for raw_name, values in payload.items():
        canonical_name = _CHANNEL_ALIASES.get(str(raw_name), str(raw_name))
        remapped[canonical_name] = values
    parsed = parse_buckling_lane_channels({"channels": remapped})
    if "buck" in parsed and "pressure_difference" not in parsed:
        parsed["pressure_difference"] = _derive_pressure_difference(parsed["buck"])
    elif "pressure_difference" in parsed and "buck" not in parsed:
        parsed["buck"] = _derive_buck_from_pressure_difference(parsed["pressure_difference"])
    elif "bpress" in parsed and "pressure_difference" not in parsed:
        parsed["pressure_difference"] = np.asarray(parsed["bpress"], dtype=float)
        parsed["buck"] = _derive_buck_from_pressure_difference(parsed["pressure_difference"])
    if "mean_volume" in parsed:
        mean_volume = _coerce_finite_1d(parsed["mean_volume"], name="mean_volume")
        if mean_volume[0] <= 0.0:
            raise ValueError("Buckling paper replay requires positive first mean_volume for normalization.")
        parsed["relative_volume"] = mean_volume / mean_volume[0]
        if "std_volume" in parsed:
            std_volume = _coerce_finite_1d(parsed["std_volume"], name="std_volume")
            if std_volume.shape != mean_volume.shape:
                raise ValueError("Buckling paper replay requires std_volume and mean_volume to share a shape.")
            parsed["relative_volume_std"] = std_volume / mean_volume[0]
    normalized: dict[str, np.ndarray] = {}
    for channel_name in (*_REQUIRED_CHANNELS, *_OPTIONAL_CHANNELS):
        if channel_name in parsed:
            normalized[channel_name] = np.asarray(parsed[channel_name], dtype=float)
    return normalized


def _resolve_buck_sweep(
    buck: object,
    *,
    paper_exact: bool,
) -> np.ndarray:
    array = np.asarray(buck, dtype=float)
    if array.ndim == 0 or array.size == 1:
        upper = _coerce_scalar(array, name="buck")
        points = _EXACT_PAPER_SWEEP_POINTS if paper_exact else _DEFAULT_PAPER_SWEEP_POINTS
        sweep = np.linspace(0.0, upper, points, dtype=float)
    else:
        sweep = _coerce_finite_1d(buck, name="buck")
    if sweep.size < 2:
        raise ValueError("Buckling paper replay requires at least two buck samples.")
    return sweep


def _derive_pressure_difference(buck_values: object) -> np.ndarray:
    return _coerce_finite_1d(buck_values, name="buck") * _BUCK_TO_PRESSURE_SCALE


def _derive_buck_from_pressure_difference(pressure_differences: object) -> np.ndarray:
    return _coerce_finite_1d(pressure_differences, name="pressure_difference") / _BUCK_TO_PRESSURE_SCALE


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
            canonical_name = _CONTROL_ALIASES.get(str(raw_name), str(raw_name))
            if canonical_name in {"buck", "pressure_difference"} and isinstance(value, (list, tuple)):
                continue
            if canonical_name in set(required):
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
