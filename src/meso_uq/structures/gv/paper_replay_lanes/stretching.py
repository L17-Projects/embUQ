from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from meso_uq.experiments import canonical_dataset_id
from meso_uq.references.gv_common import format_float

from .. import build_geometry
from ..postprocessing.common import validate_numeric_channels
from ..sampling import GVRuntimeOptions, sample_gv
from ..sampling.stretching import extract_stretching_lane

_DEFAULT_OUTPUT_ROOT = Path("_runs/gv/paper_replay/stretching")
_DEFAULT_TOTAL_FORCE = (500.0, 5_000.0, 15_000.0)
_DEFAULT_BPRESS = -91.0


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


@dataclass(frozen=True)
class StretchingPaperReplayPlan:
    campaign_id: str
    geometry_radius: float
    geometry_height: float
    material_parameters: Mapping[str, float]
    controls: Mapping[str, Any]
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
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    include_plot: bool = True,
    raw_provenance: Mapping[str, Any] | None = None,
) -> StretchingPaperReplayPlan:
    resolved_controls = {
        "tot_force": _DEFAULT_TOTAL_FORCE,
        "bpress": _DEFAULT_BPRESS,
        **dict(controls or {}),
    }
    return StretchingPaperReplayPlan(
        campaign_id=campaign_id,
        geometry_radius=float(geometry_radius),
        geometry_height=float(geometry_height),
        material_parameters=material_parameters,
        controls=resolved_controls,
        output_root=output_root,
        include_plot=include_plot,
        raw_provenance=raw_provenance,
    )


def postprocess_stretching_paper_replay_lane(
    sample_result: object,
    *,
    plan: StretchingPaperReplayPlan,
) -> StretchingPaperReplayResult:
    payload = _result_payload(sample_result)
    lane_controls = _representative_controls(plan.controls)
    lane = extract_stretching_lane(
        {
            "controls": lane_controls,
            "channels": payload.get("channels", payload),
        },
        controls=lane_controls,
    )
    channels = validate_numeric_channels(lane["channels"])
    displacement = channels["displacement"]
    force = channels["force"]
    if displacement.shape != force.shape:
        raise ValueError("Stretching paper replay requires displacement and force to share the same shape.")

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
            "qualitative_target": "paper_figure_3_force_displacement",
            "plot_requested": plan.include_plot,
            "sample_manifest": payload.get("manifest"),
            **dict(plan.raw_provenance or {}),
        },
        "dataset_id": canonical_dataset_id("gv", "stretching", geometry.id, control_id),
    }
    summary = {
        "lane": "stretching",
        "axis_channel": "displacement",
        "response_channel": "force",
        "sample_count": int(displacement.size),
        "displacement_min": float(np.min(displacement)),
        "displacement_max": float(np.max(displacement)),
        "force_min": float(np.min(force)),
        "force_max": float(np.max(force)),
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
    dataset_slug = str(result.manifest["dataset_id"]).replace(":", "__")
    plot_path = plots_dir / f"{dataset_slug}.png"
    metadata_path = plots_dir / f"{dataset_slug}.json"

    figure, axis = pyplot.subplots(figsize=(8.0, 5.0))
    displacement = np.asarray(result.channels["displacement"], dtype=float)
    force = np.asarray(result.channels["force"], dtype=float)
    axis.plot(displacement, force, marker="o", linewidth=2.0, label="DPD replay")

    if reference_curve is not None:
        ref = validate_numeric_channels(reference_curve)
        if "displacement" not in ref or "force" not in ref:
            raise ValueError("Stretching reference_curve requires displacement and force channels.")
        axis.plot(ref["displacement"], ref["force"], linestyle="--", linewidth=1.5, label="Paper reference")

    axis.set_xlabel("Displacement")
    axis.set_ylabel("Force")
    axis.set_title(
        f"GV Stretching Paper Replay\n{result.manifest['dataset_id']}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    axis.text(
        0.02,
        0.02,
        (
            f"campaign={result.plan.campaign_id}\n"
            f"geometry=(R={result.plan.geometry_radius}, H={result.plan.geometry_height})\n"
            f"controls={json.dumps(dict(result.plan.controls), sort_keys=True)}\n"
            f"samples={result.summary['sample_count']}"
        ),
        transform=axis.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
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


def run_stretching_paper_replay_lane(
    plan: StretchingPaperReplayPlan,
    *,
    sampling_callable: Callable[..., object] = sample_gv,
    reference_curve: Mapping[str, Any] | None = None,
) -> StretchingPaperReplayResult:
    sample_result = sampling_callable(**plan.sampling_kwargs())
    result = postprocess_stretching_paper_replay_lane(sample_result, plan=plan)
    if not plan.include_plot:
        return result
    return plot_stretching_paper_replay_lane(result, reference_curve=reference_curve)


__all__ = [
    "StretchingPaperReplayPlan",
    "StretchingPaperReplayResult",
    "plan_stretching_paper_replay_lane",
    "plot_stretching_paper_replay_lane",
    "postprocess_stretching_paper_replay_lane",
    "run_stretching_paper_replay_lane",
]
