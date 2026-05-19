from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np

from .. import build_geometry
from ..runtime import plan_runtime
from ..sampling import (
    GVMaterialGeometry,
    GVRuntimeOptions,
    GVSweep,
    build_sampling_plan,
    execute_sampling_plan,
    extract_sampling_channels,
    sample_gv,
)
from ..sampling.eigenmodes import parse_eigenmodes_lane_channels
from ..parameters import GV_MATERIAL_PARAMETER_NAMES
from ..sampling.validation import validate_geometry


EIGENMODES_FIGURE_ID = "Figure 8"
_EIGENMODES_AXIS = "mode_index"
_REQUIRED_ONE_OF = ("frequency", "eigenvalues")
_DEFAULT_OUTPUT_ROOT = Path("_runs/gv/paper_replay/eigenmodes")
_PRESSURE_ALIASES = {
    "pressure_difference": "bpress",
    "pressure": "bpress",
    "background_pressure": "bpress",
}
_ZERO_ALLOWED_MATERIAL_PARAMETERS = frozenset({"a3", "a4", "b1", "b2"})
_MATERIAL_ALIASES = {"muL": "mu_l"}
_REFERENCE_POSITION_KEYS = ("reference_positions", "mesh_vertices", "vertices", "positions", "ref_positions")
_MESH_FACE_KEYS = ("mesh_faces", "faces", "triangles")
_KBT_KEYS = ("kBT", "kbt")
_SELECTED_SURFACE_MODE_INDICES = (0, 4, 6, 7, 18, 24)
_SELECTED_AXIAL_MODE_INDICES = (0, 6, 20)
_SELECTED_SURFACE_MODE_TITLES = (
    "Mode 0",
    "Mode 4",
    "Mode 6",
    "Mode 7",
    "Mode 18",
    "Mode 24",
)
_SELECTED_AXIAL_MODE_LABELS = (
    "$(m, n) = (1, 2)$",
    "$(m, n) = (2, 3)$",
    "$(m, n) = (3, 4)$",
)
_SPECTRUM_ANNOTATION_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20)
_SPECTRUM_ANNOTATION_LABELS = (
    "(1, 2)",
    "(1, 2)*",
    "(1, 3)",
    "(1, 3)*",
    "(1, 4)",
    "(1, 4)*",
    "(2, 3)",
    "(2, 3)*",
    "(2, 4)",
    "(2, 4)*",
    "(2, 2)",
    "(2, 2)*",
    "(1, 5)",
    "(1, 5)*",
    "(1, 1)",
    "(3, 4)",
    "(3, 4)*",
)
_EXACT_MODE_COUNT = 30
_PAPER_BOX_CENTER = 12.5
_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "references"
_FIGURE8G_REFERENCE_CSV = _REFERENCE_DIR / "eigenmodes_fig8g_digitized.csv"
_FIGURE8G_MEAN_ABS_TOLERANCE = 1.0
_FIGURE8G_MAX_ABS_TOLERANCE = 5.0
_MODE_WINDOW_METADATA_CHANNELS = (
    "raw_eigenpair_count",
    "final_mode_count",
    "final_mode_indices",
    "selected_paper_mode_indices",
    "selected_raw_mode_indices",
    "mode_window_min_frequency_tau_inv",
)


@dataclass(frozen=True)
class EigenmodesPaperReplayPlan:
    figure_id: str
    experiment: str
    geometry: dict[str, float]
    material_parameters: dict[str, float]
    controls: dict[str, float]
    runtime_options: GVRuntimeOptions
    mode_count: int
    paper_exact: bool
    runtime_profile: str
    mode_window_policy: str
    mode_min_frequency_tau_inv: float

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
            "paper_exact": self.paper_exact,
            "runtime_profile": self.runtime_profile,
            "mode_window_policy": self.mode_window_policy,
            "mode_min_frequency_tau_inv": self.mode_min_frequency_tau_inv,
            "paper_exact_requirements": _paper_exact_requirements_manifest(),
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
    selected_modes: dict[str, Any]
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
            "selected_modes": dict(self.selected_modes),
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
    paper_exact: bool = False,
    runtime_profile: str | None = None,
    mode_window_policy: str = "frequency-min",
    mode_min_frequency_tau_inv: float = 22.5,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    timeout_seconds: int = 7200,
) -> EigenmodesPaperReplayPlan:
    validated_geometry = validate_geometry(geometry, radGV=radGV, height=height)
    validated_materials = _validate_paper_replay_material_parameters(material_parameters)
    pressure = _coerce_scalar(bpress, name="bpress")
    resolved_paper_exact = bool(paper_exact)
    resolved_mode_count = _EXACT_MODE_COUNT if resolved_paper_exact else int(mode_count)
    if resolved_mode_count <= 0:
        raise ValueError("Eigenmodes paper replay mode_count must be positive.")
    resolved_profile = _resolve_runtime_profile(runtime_profile, paper_exact=resolved_paper_exact)
    if mode_window_policy not in {"frequency-min", "raw-head", "explicit-indices"}:
        raise ValueError("Eigenmodes mode_window_policy must be frequency-min, raw-head, or explicit-indices.")
    if float(mode_min_frequency_tau_inv) <= 0.0:
        raise ValueError("Eigenmodes mode_min_frequency_tau_inv must be positive.")
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
        paper_exact=resolved_paper_exact,
        runtime_profile=resolved_profile,
        mode_window_policy=str(mode_window_policy),
        mode_min_frequency_tau_inv=float(mode_min_frequency_tau_inv),
    )


def _resolve_runtime_profile(runtime_profile: str | None, *, paper_exact: bool) -> str:
    selected = (runtime_profile or ("paper" if paper_exact else "canary")).strip().lower()
    if selected not in {"paper", "canary"}:
        raise ValueError("Eigenmodes runtime_profile must be 'paper' or 'canary'.")
    return selected


def run_eigenmodes_paper_replay_lane(
    plan: EigenmodesPaperReplayPlan,
    *,
    sampler: Callable[..., object] = sample_gv,
) -> EigenmodesPaperReplayResult:
    if plan.paper_exact and sampler is sample_gv:
        sample_result = _run_eigenmodes_paper_exact_single_point(plan)
    else:
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


def _run_eigenmodes_paper_exact_single_point(plan: EigenmodesPaperReplayPlan) -> dict[str, Any]:
    bpress = _coerce_scalar(plan.controls["bpress"], name="bpress")
    geometry = build_geometry(
        radius=plan.geometry["radGV"],
        height=plan.geometry["height"],
        source="meso_uq.structures.gv.paper_replay_lanes.eigenmodes.paper_exact",
    )
    runtime = plan_runtime(
        "eigenmodes",
        output_root=plan.runtime_options.output_root,
        geometry=geometry.id,
        controls={"bpress": bpress},
        material_parameter_overrides=dict(plan.material_parameters),
        include_experimental=True,
    )
    sampling_plan = build_sampling_plan(
        runtime,
        control_axis="bpress",
        values=(bpress,),
        timeout_seconds=plan.runtime_options.timeout_seconds,
    )
    started = time.perf_counter()
    execution = execute_sampling_plan(
        sampling_plan,
        timeout_seconds=plan.runtime_options.timeout_seconds,
        env={
            "MESOUQ_GV_PAPER_EXACT": "1",
            "MESOUQ_GV_EIGENMODES_PROFILE": plan.runtime_profile,
            "MESOUQ_GV_EIGENMODES_MODE_WINDOW_POLICY": plan.mode_window_policy,
            "MESOUQ_GV_EIGENMODES_MODE_MIN_FREQUENCY": str(plan.mode_min_frequency_tau_inv),
            "MESOUQ_GV_MATERIAL_OVERRIDES_JSON": json.dumps(
                dict(plan.material_parameters),
                sort_keys=True,
            ),
        },
    )
    runtime_seconds = time.perf_counter() - started
    geometry_payload = GVMaterialGeometry(radGV=plan.geometry["radGV"], height=plan.geometry["height"])
    channels = extract_sampling_channels(
        experiment="eigenmodes",
        work_dir=runtime.work_dir,
        controls={"bpress": bpress},
        sweep=GVSweep("bpress", (bpress,)),
        geometry=geometry_payload,
    )
    window_manifest_path = Path(runtime.work_dir) / "analysis" / "output" / "mode_window_manifest.json"
    window_manifest = (
        json.loads(window_manifest_path.read_text(encoding="utf-8"))
        if window_manifest_path.is_file()
        else None
    )
    return {
        "experiment": "eigenmodes",
        "geometry": dict(plan.geometry),
        "material_parameters": dict(plan.material_parameters),
        "controls": {"bpress": bpress},
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
        "eigenmode_window_manifest": window_manifest,
        "status": "completed",
    }


def postprocess_eigenmodes_paper_replay_lane(
    sample_result: object,
    *,
    plan: EigenmodesPaperReplayPlan | None = None,
) -> EigenmodesPaperReplayResult:
    payload = _coerce_mapping(sample_result, context="sample_result")
    mode_count = plan.mode_count if plan is not None else _EXACT_MODE_COUNT
    try:
        channels = _normalize_eigenmode_channels(payload, mode_count=mode_count)
    except ValueError as exc:
        if "requires eigenvalues or eigenfrequencies" in str(exc):
            raise ValueError(
                "Eigenmodes paper replay requires at least one spectral channel: "
                + ", ".join(_REQUIRED_ONE_OF)
                + "."
            ) from exc
        if plan is not None and plan.paper_exact and _is_spectrum_payload_error(exc):
            raise ValueError(
                "Paper-exact eigenmodes replay requires 30 finite eigenfrequencies/eigenvalues "
                f"for {EIGENMODES_FIGURE_ID} first-30 spectrum evidence; {exc}"
            ) from exc
        if plan is not None and plan.paper_exact and _is_mode_shape_payload_error(exc):
            raise ValueError(
                "Paper-exact eigenmodes replay requires unambiguous selected Figure 8 "
                f"surface-mode evidence and axial-profile evidence; {exc}"
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

    if plan is not None and plan.paper_exact:
        _validate_paper_exact_eigenmode_evidence(channels)

    figure_id = plan.figure_id if plan is not None else EIGENMODES_FIGURE_ID
    experiment = str(payload.get("experiment", plan.experiment if plan is not None else "eigenmodes"))
    geometry = _resolve_geometry(payload, plan)
    material_parameters = _resolve_material_parameters(payload, plan)
    controls = _resolve_controls(payload, plan)
    selected_modes = _resolve_selected_modes(channels)
    provenance = _resolve_provenance(payload, plan, selected_modes=selected_modes)
    provenance["required_channels"] = list(_REQUIRED_ONE_OF)

    return EigenmodesPaperReplayResult(
        figure_id=figure_id,
        experiment=experiment,
        axis=_EIGENMODES_AXIS,
        controls=controls,
        geometry=geometry,
        material_parameters=material_parameters,
        channels=channels,
        selected_modes=selected_modes,
        provenance=provenance,
    )


def plot_eigenmodes_paper_replay(
    result: EigenmodesPaperReplayResult,
    *,
    output_path: str | Path,
    full_mode_shape_plot: bool = False,
) -> Path:
    target = _validate_runs_output_path(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mode_shape_source = _extract_mode_shape_source(result.channels)
    requested_mode_indices = _requested_mode_indices(result, full_mode_shape_plot=full_mode_shape_plot)
    if full_mode_shape_plot and mode_shape_source is None:
        raise ValueError(
            "Full eigenmode shape plot requested, but eigenvectors and reference_positions/mesh_vertices are unavailable."
        )
    if (
        full_mode_shape_plot
        and mode_shape_source is not None
        and "mesh_faces" in mode_shape_source
        and requested_mode_indices
    ):
        return _plot_eigenmodes_paper_figure(
            result,
            mode_shape_source=mode_shape_source,
            mode_indices=requested_mode_indices,
            target=target,
            plt=plt,
        )
    if mode_shape_source is not None:
        return _plot_eigenmodes_spectrum_panels(
            result,
            mode_shape_source=mode_shape_source,
            target=target,
            plt=plt,
        )

    if mode_shape_source is None:
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        axes = np.asarray([[axis]], dtype=object)
        spectrum_axis = axis
    else:
        columns = 3
        total_mode_panels = len(requested_mode_indices)
        rows = 1 + int(np.ceil(total_mode_panels / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.4 * rows))
        axes = np.asarray(axes, dtype=object)
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        spectrum_axis = axes[0, 0]
        for extra_axis in axes[0, 1:]:
            extra_axis.axis("off")

    _plot_frequency_panel(spectrum_axis, result, annotate=False)
    spectrum_axis.set_title(f"GV eigenmodes paper replay: {result.figure_id}")

    if mode_shape_source is not None:
        for axis, mode_index in zip(axes.reshape(-1)[3:], requested_mode_indices):
            _plot_mode_shape_panel(axis, result, mode_shape_source, mode_index=mode_index)
        for axis in axes.reshape(-1)[3 + len(requested_mode_indices):]:
            axis.axis("off")

    figure.text(
        0.02,
        0.02,
        (
            f"experiment={result.experiment} | bpress={result.controls['bpress']} | "
            f"modes={len(result.channels['mode_index'])} | exact={result.provenance.get('paper_exact', False)}"
        ),
        fontsize=8,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target

def _normalize_eigenmode_channels(sample_payload: Mapping[str, Any], *, mode_count: int) -> dict[str, np.ndarray]:
    channels_like = sample_payload.get("channels")
    payload = _coerce_mapping(channels_like, context="sample_result.channels")
    parser_payload = dict(payload)
    if "lambdas" in parser_payload and "eigenvalues" not in parser_payload:
        parser_payload["eigenvalues"] = parser_payload["lambdas"]
    parsed = parse_eigenmodes_lane_channels(
        {"channels": parser_payload},
        mode_count=mode_count,
        include_eigenvectors=True,
    )
    normalized: dict[str, np.ndarray] = {}
    normalized[_EIGENMODES_AXIS] = np.asarray(parsed[_EIGENMODES_AXIS], dtype=float)
    if "eigenvalues" in parsed:
        normalized["eigenvalues"] = np.asarray(parsed["eigenvalues"], dtype=float)
        if "lambdas" in payload:
            normalized["lambdas"] = np.asarray(parsed["eigenvalues"], dtype=float)
    if "eigenvectors" in parsed:
        normalized["eigenvectors"] = np.asarray(parsed["eigenvectors"], dtype=float)
    frequency = _resolve_frequency(sample_payload, payload, parsed, mode_count=mode_count)
    if frequency is not None:
        normalized["frequency"] = np.asarray(frequency, dtype=float)
    for channel_name in ("reference_positions", "mesh_faces"):
        preserved = _extract_auxiliary_channel(payload, channel_name, mode_count=mode_count)
        if preserved is not None:
            normalized[channel_name] = preserved
    for channel_name in _MODE_WINDOW_METADATA_CHANNELS:
        if channel_name in payload:
            normalized[channel_name] = np.asarray(payload[channel_name], dtype=float)
    return normalized


def _paper_exact_requirements_manifest() -> dict[str, object]:
    return {
        "first_spectrum_mode_count": _EXACT_MODE_COUNT,
        "selected_surface_mode_indices": list(_SELECTED_SURFACE_MODE_INDICES),
        "selected_axial_mode_indices": list(_SELECTED_AXIAL_MODE_INDICES),
    }


def _is_spectrum_payload_error(exc: ValueError) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in (
            "contains non-finite values",
            "must contain data",
            "must be 1D",
            "frequency channel must be finite",
        )
    )


def _is_mode_shape_payload_error(exc: ValueError) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in (
            "eigenvectors",
            "reference_positions",
            "mesh_vertices",
            "mesh_faces",
            "faces",
        )
    )


def _validate_paper_exact_eigenmode_evidence(channels: MutableMapping[str, np.ndarray]) -> None:
    mode_index = np.asarray(channels.get(_EIGENMODES_AXIS, ()), dtype=float)
    if mode_index.ndim != 1 or mode_index.size < _EXACT_MODE_COUNT or not np.all(np.isfinite(mode_index)):
        raise ValueError(
            "Paper-exact eigenmodes replay requires first 30 finite modes for "
            f"{EIGENMODES_FIGURE_ID}; received {int(mode_index.size)} mode indices."
        )

    frequency = np.asarray(channels.get("frequency", ()), dtype=float)
    has_first_30_frequency = (
        frequency.ndim == 1
        and frequency.size >= _EXACT_MODE_COUNT
        and np.all(np.isfinite(frequency[:_EXACT_MODE_COUNT]))
    )
    if not has_first_30_frequency:
        raise ValueError(
            "Paper-exact eigenmodes replay requires 30 finite eigenfrequencies for "
            f"{EIGENMODES_FIGURE_ID} first-30 spectrum evidence; received {int(frequency.size)}."
        )

    _validate_paper_exact_mode_shape_payload(channels)


def _validate_paper_exact_mode_shape_payload(channels: MutableMapping[str, np.ndarray]) -> None:
    if "eigenvectors" not in channels:
        raise ValueError(
            "Paper-exact eigenmodes replay requires mode-shape evidence: "
            "eigenvectors for the first 30 modes are missing."
        )
    eigenvectors = np.asarray(channels["eigenvectors"], dtype=float)
    if eigenvectors.ndim != 2:
        raise ValueError(
            "Paper-exact eigenmodes replay has ambiguous mode-shape evidence: "
            "eigenvectors must be a 2D (mode, vertex_displacement) array."
        )
    if eigenvectors.shape[0] < _EXACT_MODE_COUNT or not np.all(np.isfinite(eigenvectors[:_EXACT_MODE_COUNT])):
        raise ValueError(
            "Paper-exact eigenmodes replay requires finite eigenvectors for the first 30 modes; "
            f"received {int(eigenvectors.shape[0])} mode-shape rows."
        )

    if "reference_positions" not in channels:
        raise ValueError(
            "Paper-exact eigenmodes replay requires axial-profile evidence: "
            "reference_positions/mesh_vertices are missing."
        )
    reference_positions = np.asarray(channels["reference_positions"], dtype=float)
    if (
        reference_positions.ndim != 2
        or reference_positions.shape[1] != 3
        or reference_positions.size == 0
        or not np.all(np.isfinite(reference_positions))
    ):
        raise ValueError(
            "Paper-exact eigenmodes replay has ambiguous axial-profile evidence: "
            "reference_positions/mesh_vertices must be a finite Nx3 array."
        )
    if eigenvectors.shape[1] != reference_positions.size:
        raise ValueError(
            "Paper-exact eigenmodes replay has ambiguous mode-shape evidence: "
            f"eigenvector width {int(eigenvectors.shape[1])} does not match "
            f"3D reference_positions size {int(reference_positions.size)}."
        )

    if "mesh_faces" not in channels:
        raise ValueError(
            "Paper-exact eigenmodes replay requires selected Figure 8 surface-mode evidence: "
            "mesh_faces/faces are missing."
        )
    raw_mesh_faces = np.asarray(channels["mesh_faces"])
    if raw_mesh_faces.ndim != 2 or raw_mesh_faces.shape[1] != 3 or raw_mesh_faces.size == 0:
        raise ValueError(
            "Paper-exact eigenmodes replay has ambiguous selected Figure 8 surface-mode evidence: "
            "mesh_faces/faces must be a non-empty Nx3 triangle array."
        )
    try:
        mesh_face_values = raw_mesh_faces.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Paper-exact eigenmodes replay has ambiguous selected Figure 8 surface-mode evidence: "
            "mesh_faces/faces must contain finite integer vertex indices."
        ) from exc
    if (
        not np.all(np.isfinite(mesh_face_values))
        or not np.all(np.equal(mesh_face_values, np.floor(mesh_face_values)))
    ):
        raise ValueError(
            "Paper-exact eigenmodes replay has ambiguous selected Figure 8 surface-mode evidence: "
            "mesh_faces/faces must contain finite integer vertex indices."
        )
    mesh_faces = mesh_face_values.astype(int)
    if np.any(mesh_faces < 0) or np.any(mesh_faces >= reference_positions.shape[0]):
        raise ValueError(
            "Paper-exact eigenmodes replay has ambiguous selected Figure 8 surface-mode evidence: "
            "mesh_faces/faces must contain valid vertex indices into reference_positions."
        )
    channels["mesh_faces"] = mesh_faces

    paper_reference = _paper_reference_coordinates(reference_positions)
    try:
        _paper_axial_slice_indices(paper_reference)
    except ValueError as exc:
        raise ValueError(
            "Paper-exact eigenmodes replay requires axial-profile evidence for Figure 8 panel (h): "
            + str(exc)
        ) from exc


def _resolve_frequency(
    sample_payload: Mapping[str, Any],
    raw_channels: Mapping[str, Any],
    parsed_channels: Mapping[str, np.ndarray],
    *,
    mode_count: int,
) -> np.ndarray | None:
    if "eigenvalues" in parsed_channels:
        kbt = _resolve_kbt(sample_payload, raw_channels)
        if kbt is None:
            raise ValueError("Eigenmodes paper replay requires kBT to derive frequency from eigenvalues/lambdas.")
        eigenvalues = np.asarray(parsed_channels["eigenvalues"], dtype=float)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("Eigenmodes eigenvalues/lambdas must be positive to derive frequency.")
        return np.sqrt(np.sort(kbt / eigenvalues)[:mode_count])

    frequency = _select_frequency_payload(raw_channels)
    if frequency is None:
        return None
    return frequency[:mode_count]


def _resolve_kbt(sample_payload: Mapping[str, Any], raw_channels: Mapping[str, Any]) -> float | None:
    for name in _KBT_KEYS:
        if name in raw_channels:
            return _coerce_scalar(raw_channels[name], name=name)
    for container_name in ("parameters", "manifest", "metadata"):
        container = sample_payload.get(container_name)
        if isinstance(container, Mapping):
            for name in _KBT_KEYS:
                if name in container:
                    return _coerce_scalar(container[name], name=f"{container_name}.{name}")
    return None


def _select_frequency_payload(raw_channels: Mapping[str, Any]) -> np.ndarray | None:
    lowered = {str(name).strip().lower(): value for name, value in raw_channels.items()}
    for name in ("frequency", "frequencies", "eigenfrequency", "eigenfrequencies"):
        if name in lowered:
            values = np.asarray(lowered[name], dtype=float)
            if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError("Eigenmodes frequency channel must be finite 1D data.")
            return values
    return None


def _extract_auxiliary_channel(
    payload: Mapping[str, Any],
    channel_name: str,
    *,
    mode_count: int,
) -> np.ndarray | None:
    keys = _REFERENCE_POSITION_KEYS if channel_name == "reference_positions" else _MESH_FACE_KEYS
    lowered = {str(name).strip().lower(): value for name, value in payload.items()}
    for key in keys:
        if key in lowered:
            array = np.asarray(lowered[key], dtype=float)
            if channel_name == "reference_positions" and array.ndim != 2:
                raise ValueError("Eigenmodes reference_positions/mesh_vertices must be a 2D array.")
            if channel_name == "mesh_faces" and array.ndim != 2:
                raise ValueError("Eigenmodes mesh_faces must be a 2D array.")
            return array
    return None


def _resolve_selected_modes(channels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    available_count = int(np.asarray(channels[_EIGENMODES_AXIS]).size)
    mode_shape_count = int(np.asarray(channels.get("eigenvectors", ())).shape[0]) if "eigenvectors" in channels else 0
    available_shape_count = min(available_count, mode_shape_count) if mode_shape_count else 0
    selected: dict[str, Any] = {
        "surface_mode_indices": list(_SELECTED_SURFACE_MODE_INDICES),
        "surface_mode_indices_available": [index for index in _SELECTED_SURFACE_MODE_INDICES if index < available_shape_count],
        "axial_mode_indices": list(_SELECTED_AXIAL_MODE_INDICES),
        "axial_mode_indices_available": [index for index in _SELECTED_AXIAL_MODE_INDICES if index < available_shape_count],
    }
    if "raw_eigenpair_count" in channels:
        selected["raw_eigenpair_count"] = int(np.asarray(channels["raw_eigenpair_count"]).reshape(-1)[0])
    if "final_mode_count" in channels:
        selected["final_mode_count"] = int(np.asarray(channels["final_mode_count"]).reshape(-1)[0])
    if "selected_raw_mode_indices" in channels:
        selected["selected_raw_mode_indices"] = [
            int(value) for value in np.asarray(channels["selected_raw_mode_indices"]).reshape(-1)
        ]
    if "final_mode_indices" in channels:
        selected["final_mode_indices"] = [
            int(value) for value in np.asarray(channels["final_mode_indices"]).reshape(-1)
        ]
    if "selected_paper_mode_indices" in channels:
        selected["selected_paper_mode_indices"] = [
            int(value) for value in np.asarray(channels["selected_paper_mode_indices"]).reshape(-1)
        ]
    if "mode_window_min_frequency_tau_inv" in channels:
        selected["mode_window_min_frequency_tau_inv"] = float(
            np.asarray(channels["mode_window_min_frequency_tau_inv"]).reshape(-1)[0]
        )
    return selected


def _extract_mode_shape_source(channels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    if "eigenvectors" not in channels or "reference_positions" not in channels:
        return None
    source = {
        "eigenvectors": np.asarray(channels["eigenvectors"], dtype=float),
        "reference_positions": np.asarray(channels["reference_positions"], dtype=float),
    }
    if "mesh_faces" in channels:
        source["mesh_faces"] = np.asarray(channels["mesh_faces"], dtype=int)
    return source


def _requested_mode_indices(
    result: EigenmodesPaperReplayResult,
    *,
    full_mode_shape_plot: bool,
) -> list[int]:
    if full_mode_shape_plot:
        selected = list(result.selected_modes.get("surface_mode_indices_available", ()))
        if selected:
            return selected
        return list(range(len(result.channels["mode_index"])))
    return list(result.selected_modes.get("surface_mode_indices_available", ()))


def _plot_mode_shape_panel(
    axis: Any,
    result: EigenmodesPaperReplayResult,
    mode_shape_source: Mapping[str, np.ndarray],
    *,
    mode_index: int,
) -> None:
    reference_positions = np.asarray(mode_shape_source["reference_positions"], dtype=float)
    eigenvectors = np.asarray(mode_shape_source["eigenvectors"], dtype=float)
    vectors = np.asarray(eigenvectors[mode_index], dtype=float).reshape(-1, 3)
    radial_distance = np.linalg.norm(reference_positions[:, :2], axis=1)
    displacement_norm = np.linalg.norm(vectors, axis=1)
    axis.plot(radial_distance, displacement_norm, marker="o", linestyle="None", color="#1f77b4")
    axis.set_title(f"Surface mode {mode_index}")
    axis.set_xlabel("Radius")
    axis.set_ylabel("Disp.")
    axis.grid(True, alpha=0.2)


def _plot_eigenmodes_paper_figure(
    result: EigenmodesPaperReplayResult,
    *,
    mode_shape_source: Mapping[str, np.ndarray],
    mode_indices: list[int],
    target: Path,
    plt: Any,
) -> Path:
    from matplotlib.gridspec import GridSpec

    figure = plt.figure(figsize=(12.0, 12.0))
    grid = GridSpec(3, 3, figure=figure, height_ratios=(1.0, 1.0, 0.72), hspace=0.25, wspace=0.22)
    surface_axes = [
        figure.add_subplot(grid[row, col], projection="3d")
        for row in range(2)
        for col in range(3)
    ]
    spectrum_axis = figure.add_subplot(grid[2, 0])
    axial_axis = figure.add_subplot(grid[2, 1:])

    for panel_index, (axis, mode_index) in enumerate(zip(surface_axes, mode_indices)):
        title = (
            _SELECTED_SURFACE_MODE_TITLES[panel_index]
            if panel_index < len(_SELECTED_SURFACE_MODE_TITLES)
            else f"Mode {mode_index}"
        )
        _plot_mode_surface(axis, mode_shape_source, mode_index=mode_index, title=title)
        axis.text2D(0.06, 0.95, f"({chr(ord('a') + panel_index)})", transform=axis.transAxes, fontsize=14)
    for axis in surface_axes[len(mode_indices):]:
        axis.set_axis_off()

    _plot_frequency_panel(spectrum_axis, result, annotate=True)
    spectrum_axis.text(-0.12, 1.02, "(g)", transform=spectrum_axis.transAxes, fontsize=14)

    _plot_axial_modes(axial_axis, result, mode_shape_source)
    axial_axis.text(-0.08, 1.02, "(h)", transform=axial_axis.transAxes, fontsize=14)

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight", dpi=220)
    plt.close(figure)
    return target


def _plot_eigenmodes_spectrum_panels(
    result: EigenmodesPaperReplayResult,
    *,
    mode_shape_source: Mapping[str, np.ndarray],
    target: Path,
    plt: Any,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    spectrum_axis, axial_axis = np.asarray(axes, dtype=object).reshape(-1)[:2]

    _plot_frequency_panel(spectrum_axis, result, annotate=True)
    spectrum_axis.text(-0.12, 1.02, "(g)", transform=spectrum_axis.transAxes, fontsize=16, fontweight="bold")

    _plot_axial_modes(axial_axis, result, mode_shape_source)
    axial_axis.text(-0.08, 1.02, "(h)", transform=axial_axis.transAxes, fontsize=16, fontweight="bold")

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, bbox_inches="tight", dpi=220)
    plt.close(figure)
    return target


def _plot_frequency_panel(
    axis: Any,
    result: EigenmodesPaperReplayResult,
    *,
    annotate: bool,
) -> None:
    mode_index = np.asarray(result.channels["mode_index"], dtype=float)
    frequency = np.asarray(result.channels["frequency"], dtype=float)
    axis.plot(
        mode_index,
        frequency,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linestyle="None",
        color="royalblue",
        label="QHA frequencies",
    )
    if annotate:
        _annotate_frequency_panel(axis, mode_index, frequency)
    axis.set_xlabel("$N$")
    axis.set_ylabel(r"$\omega$ [$\tau^{-1}$]")
    axis.set_ylim(10.0, 80.0)
    axis.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])
    axis.legend(loc="upper left")


def _annotate_frequency_panel(axis: Any, mode_index: np.ndarray, frequency: np.ndarray) -> None:
    if not hasattr(axis, "annotate"):
        return
    for label_index, spectrum_index in enumerate(_SPECTRUM_ANNOTATION_INDICES):
        if spectrum_index >= mode_index.size or label_index >= len(_SPECTRUM_ANNOTATION_LABELS):
            continue
        pair_index = spectrum_index // 2
        if pair_index % 2 == 0:
            offset = (-0.4, 6.0) if spectrum_index % 2 == 0 else (0.1, 10.0)
        else:
            offset = (-0.1, -10.0) if spectrum_index % 2 == 0 else (0.4, -6.0)
        x = float(mode_index[spectrum_index])
        y = float(frequency[spectrum_index])
        axis.annotate(
            _SPECTRUM_ANNOTATION_LABELS[label_index],
            (x, y),
            xytext=(x + offset[0], y + offset[1]),
            arrowprops={"facecolor": "black", "arrowstyle": "->"},
            fontsize=8,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
            textcoords="data",
        )


def _plot_mode_surface(
    axis: Any,
    mode_shape_source: Mapping[str, np.ndarray],
    *,
    mode_index: int,
    title: str,
) -> None:
    reference = np.asarray(mode_shape_source["reference_positions"], dtype=float)
    faces = np.asarray(mode_shape_source["mesh_faces"], dtype=int)
    vectors = np.asarray(mode_shape_source["eigenvectors"][mode_index], dtype=float).reshape(-1, 3)
    centered = reference - np.mean(reference, axis=0)
    scale = 10.0
    displacement = scale * vectors
    signed_amplitude = np.sum(centered * displacement, axis=1)
    signed_norm = np.sign(signed_amplitude) * np.linalg.norm(displacement, axis=1)
    face_values = np.mean(signed_norm[faces], axis=1)
    deformed = centered + displacement
    surface = axis.plot_trisurf(
        deformed[:, 0],
        deformed[:, 1],
        deformed[:, 2],
        triangles=faces,
        linewidth=0.25,
        edgecolor="black",
        cmap="viridis",
    )
    surface.set_array(face_values)
    limit = max(float(np.max(np.abs(face_values))), 1.0e-12)
    surface.set_clim(-limit, limit)
    axis.view_init(elev=20, azim=0)
    axis.set_title(title, fontsize=12)
    axis.set_axis_off()
    span = np.ptp(centered, axis=0)
    axis.set_box_aspect(tuple(float(value) if value > 0.0 else 1.0 for value in span))


def _plot_axial_modes(
    axis: Any,
    result: EigenmodesPaperReplayResult,
    mode_shape_source: Mapping[str, np.ndarray],
) -> None:
    reference = np.asarray(mode_shape_source["reference_positions"], dtype=float)
    paper_reference = _paper_reference_coordinates(reference)
    height = float(result.geometry.get("height", np.ptp(reference[:, 2])))
    selected_indices = _paper_axial_slice_indices(paper_reference)
    z = np.linspace(-0.5 * height, 0.5 * height, int(selected_indices.size))
    colors = ("royalblue", "crimson", "forestgreen")
    for label, color, mode_index in zip(
        _SELECTED_AXIAL_MODE_LABELS,
        colors,
        result.selected_modes.get("axial_mode_indices_available", ()),
    ):
        vectors = np.asarray(mode_shape_source["eigenvectors"][mode_index], dtype=float).reshape(-1, 3)
        radial_amplitude = np.sum(vectors[selected_indices, :2] * paper_reference[selected_indices, :2], axis=1)
        axis.plot(
            z,
            radial_amplitude,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            linestyle="--",
            color=color,
            label=label,
        )
    if height > 0.0:
        axis.axvline(-0.5 * height + 0.15 * height, color="gray", linestyle="--", linewidth=1.0)
        axis.axvline(0.5 * height - 0.15 * height, color="gray", linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"$z$ [$r_c$]")
    axis.set_ylabel(r"$u_r$ [a.u.]")
    axis.set_xlim(-8.0, 8.0)
    axis.set_ylim(-1.5, 2.0)
    axis.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    axis.legend(loc="lower right", fontsize=9)


def _paper_reference_coordinates(reference: np.ndarray) -> np.ndarray:
    raw_indices = np.where((np.abs(reference[:, 1] - _PAPER_BOX_CENTER) < 0.02) & (reference[:, 0] > _PAPER_BOX_CENTER))[0]
    if raw_indices.size >= 3:
        return reference

    # The current extractor stores centered positions, while the paper script
    # computes panel (h) in the original 25 rc simulation box coordinates.
    return np.asarray(reference, dtype=float) + _PAPER_BOX_CENTER


def _paper_axial_slice_indices(reference: np.ndarray) -> np.ndarray:
    raw_indices = np.where((np.abs(reference[:, 1] - _PAPER_BOX_CENTER) < 0.02) & (reference[:, 0] > _PAPER_BOX_CENTER))[0]
    if raw_indices.size >= 3:
        return raw_indices[np.argsort(reference[raw_indices, 2])]
    raise ValueError("Eigenmodes panel (h) requires at least three vertices on the paper axial slice.")


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
    *,
    selected_modes: Mapping[str, Any],
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
    window_manifest = payload.get("eigenmode_window_manifest")
    if isinstance(window_manifest, Mapping):
        provenance["eigenmode_window_manifest"] = dict(window_manifest)
    provenance["protocol_references"] = {
        "dropped_scripts_role": "Protocol references only; not operational source.",
        "canonical_replay_data": "Canonical eigenmode replay data come from Mirheo reruns.",
    }
    provenance["paper_exact_requirements"] = _paper_exact_requirements_manifest()
    provenance["selected_modes"] = dict(selected_modes)
    if plan is not None:
        provenance["lane_plan"] = plan.to_manifest()
        provenance["paper_exact"] = plan.paper_exact
    if plan is not None and plan.paper_exact and "raw_eigenpair_count" in selected_modes:
        provenance["figure8g_acceptance"] = evaluate_eigenmodes_figure8g_acceptance_from_channels(
            payload.get("channels", {}),
            reference_csv=_FIGURE8G_REFERENCE_CSV,
        )
    return provenance


def evaluate_eigenmodes_figure8g_acceptance_from_channels(
    channels_like: object,
    *,
    reference_csv: str | Path = _FIGURE8G_REFERENCE_CSV,
    mean_abs_tolerance: float = _FIGURE8G_MEAN_ABS_TOLERANCE,
    max_abs_tolerance: float = _FIGURE8G_MAX_ABS_TOLERANCE,
) -> dict[str, Any]:
    channels = _coerce_mapping(channels_like, context="channels")
    frequency = _select_frequency_payload(channels)
    if frequency is None:
        if "eigenvalues" not in channels:
            raise ValueError("Figure 8(g) acceptance requires frequency or eigenvalues.")
        kbt = _resolve_kbt({"channels": channels}, channels)
        if kbt is None:
            raise ValueError("Figure 8(g) acceptance requires kBT when eigenvalues are used.")
        eigenvalues = np.asarray(channels["eigenvalues"], dtype=float)
        frequency = np.sqrt(np.sort(kbt / eigenvalues)[: eigenvalues.size])
    reference = _load_figure8g_reference(reference_csv)
    count = min(int(frequency.size), int(reference.size), _EXACT_MODE_COUNT)
    if count < _EXACT_MODE_COUNT:
        raise ValueError(
            f"Figure 8(g) acceptance requires {_EXACT_MODE_COUNT} modes; got {count}."
        )
    delta = np.asarray(frequency[:count], dtype=float) - np.asarray(reference[:count], dtype=float)
    abs_delta = np.abs(delta)
    mean_abs = float(np.mean(abs_delta))
    max_abs = float(np.max(abs_delta))
    return {
        "reference": str(reference_csv),
        "mode_count": count,
        "mean_abs_error_tau_inv": mean_abs,
        "max_abs_error_tau_inv": max_abs,
        "mean_abs_tolerance_tau_inv": float(mean_abs_tolerance),
        "max_abs_tolerance_tau_inv": float(max_abs_tolerance),
        "passed": bool(mean_abs <= mean_abs_tolerance and max_abs <= max_abs_tolerance),
        "tolerance_rationale": (
            "Archive-backed MesoUQ replay matched digitized Figure 8(g) with mean absolute error "
            "about 0.067 tau^-1 and max absolute error about 0.143 tau^-1; operational reruns "
            "get wider stochastic tolerance but must remain near the paper spectrum."
        ),
    }


def _load_figure8g_reference(path: str | Path) -> np.ndarray:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if rows.size == 0:
        raise ValueError(f"Figure 8(g) reference is empty: {path}")
    return np.atleast_1d(rows["omega_tau_inv"]).astype(float)


def _validate_runs_output_path(output_path: str | Path) -> Path:
    return _validate_runs_root(output_path, label="plots")


def _validate_runs_root(output_path: str | Path, *, label: str = "outputs") -> Path:
    path = Path(output_path)
    if ".." in path.parts:
        raise ValueError(f"Paper replay {label} must be written under a relative _runs/ path.")
    if path.is_absolute():
        resolved = path.resolve()
        runs_root = os.environ.get("MESOUQ_RUNS_ROOT")
        if runs_root:
            allowed_root = Path(runs_root).expanduser().resolve()
            if resolved == allowed_root or allowed_root in resolved.parents:
                return resolved
        if any(parent.name == "_runs" for parent in resolved.parents):
            return resolved
        raise ValueError(f"Paper replay {label} must be written under _runs/ or MESOUQ_RUNS_ROOT.")
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
