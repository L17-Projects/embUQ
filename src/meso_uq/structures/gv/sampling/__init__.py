"""GV sampling helpers."""

import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import buckling, stretching
from .. import build_geometry
from ..numerical_data import build_gv_numerical_dataset_manifest
from ..runtime import plan_runtime
from .artifacts import write_sampling_artifacts
from .buckling import (
    _BUCKLING_AXIS,
    _REQUIRED_CONTROLS,
    extract_buckling_lane,
    parse_buckling_lane_channels,
    parse_buckling_lane_controls,
    parse_buckling_sampling_lane,
)
from .stretching import (
    _REQUIRED_CONTROLS as _STRETCHING_REQUIRED_CONTROLS,
    _STRETCHING_AXIS,
    extract_stretching_lane,
    parse_stretching_lane_channels,
    parse_stretching_lane_controls,
    parse_stretching_sampling_lane,
)
from .executor import SamplingExecutionResult, execute_sampling_plan
from .extraction import extract_sampling_channels, merge_sampling_channels
from .failures import (
    GVCommandFailure,
    GVLogScanFailure,
    GVSamplingExtractionError,
    GVSamplingFailure,
    GVSamplingPlanError,
    GVCwdValidationError,
    GVTimeoutFailure,
)
from .logs import LogScanIssue, scan_log_file, scan_log_text
from .planner import (
    GVSamplingPlan,
    SamplingCommand,
    SamplingRun,
    build_sampling_plan,
    default_sampling_timeout_seconds,
)
from .types import GVMaterialGeometry, GVSweep, GVRuntimeOptions, GVSampleResult
from .validation import (
    GV_SAMPLING_EXPERIMENTS,
    validate_explicit_controls,
    validate_geometry,
    validate_gv_experiment,
    validate_material_parameters,
    validate_sample_gv_request,
)


def sample_gv(
    *,
    experiment: str,
    material_parameters: Mapping[str, object],
    geometry: GVMaterialGeometry | Mapping[str, object] | None = None,
    controls: Mapping[str, object] | None = None,
    runtime_options: GVRuntimeOptions | None = None,
    radGV: float | None = None,
    height: float | None = None,
    campaign_id: str | None = None,
    dry_run: bool = False,
    write_artifacts: bool = True,
) -> GVSampleResult:
    """Run one explicit GV control sweep synchronously and return finite channels."""

    validated_experiment, normalized_material, validated_geometry, fixed_controls, sweep, validated_options = (
        validate_sample_gv_request(
            experiment=experiment,
            material_parameters=material_parameters,
            geometry=geometry,
            controls=controls,
            runtime_options=runtime_options,
            radGV=radGV,
            height=height,
        )
    )
    if write_artifacts and not dry_run and len(sweep.values) > 1:
        raise GVSamplingPlanError(
            "GV sampling artifact writing currently requires a single sweep value so the scalar "
            "numerical dataset manifest cannot mislabel a multi-point sweep. Disable write_artifacts "
            "or call sample_gv once per control value until sweep-aware dataset IDs are available."
        )
    resolved_campaign = campaign_id or _default_campaign_id()
    gv_geometry = build_geometry(
        radius=validated_geometry.radGV,
        height=validated_geometry.height,
        source="meso_uq.structures.gv.sampling.sample_gv",
    )
    runtime_manifests: list[Mapping[str, object]] = []
    plan_manifests: list[Mapping[str, object]] = []
    execution_manifests: list[Mapping[str, object]] = []
    work_dirs: list[Path] = []
    channel_sets = []

    started = time.perf_counter()
    material_env = {
        "MESOUQ_GV_MATERIAL_OVERRIDES_JSON": json.dumps(normalized_material, sort_keys=True),
    }
    for value in sweep.values:
        run_controls = dict(fixed_controls)
        run_controls[sweep.axis] = float(value)
        runtime = plan_runtime(
            validated_experiment,
            output_root=validated_options.output_root,
            geometry=gv_geometry.id,
            controls=run_controls,
            material_parameter_overrides=normalized_material,
            include_experimental=True,
        )
        plan = build_sampling_plan(
            runtime,
            control_axis=sweep.axis,
            values=(float(value),),
            timeout_seconds=validated_options.timeout_seconds,
        )
        runtime_manifest = runtime.to_manifest()
        runtime_manifests.append(runtime_manifest)
        plan_manifests.append(plan.to_manifest())
        work_dirs.append(Path(runtime.work_dir))
        if dry_run:
            continue
        execution = execute_sampling_plan(
            plan,
            timeout_seconds=validated_options.timeout_seconds,
            env=material_env,
        )
        execution_manifests.append(
            {
                "executed_commands": list(execution.executed_commands),
                "return_codes": list(execution.return_codes),
            }
        )
        channel_sets.append(
            extract_sampling_channels(
                experiment=validated_experiment,
                work_dir=runtime.work_dir,
                controls=run_controls,
                sweep=sweep,
                geometry=validated_geometry,
            )
        )

    runtime_seconds = time.perf_counter() - started
    channels = {} if dry_run else merge_sampling_channels(tuple(channel_sets))
    manifest_payload = None
    manifest_path = None
    hdf5_path = None
    if not dry_run and write_artifacts:
        representative_controls = dict(fixed_controls)
        representative_controls[sweep.axis] = float(sweep.values[0])
        manifest = build_gv_numerical_dataset_manifest(
            campaign_id=resolved_campaign,
            structure="gv",
            experiment=validated_experiment,
            geometry_radius=validated_geometry.radGV,
            geometry_height=validated_geometry.height,
            material_parameters=normalized_material,
            controls=representative_controls,
            raw_provenance={
                "workflow": "gv_model_sampling_block",
                "sweep": {"axis": sweep.axis, "values": list(sweep.values)},
                "runtime_manifests": runtime_manifests,
                "plan_manifests": plan_manifests,
                "work_dirs": [str(path) for path in work_dirs],
            },
            quality_flags={
                "finite_observables": True,
                "finite_observable_ratio": 1.0,
                "canary_failures": [],
            },
        )
        written = write_sampling_artifacts({"manifest": manifest, "channels": channels})
        manifest_payload = written.manifest
        manifest_path = written.manifest_path
        hdf5_path = written.hdf5_path
    return GVSampleResult(
        experiment=validated_experiment,
        geometry=validated_geometry,
        material_parameters=normalized_material,
        runtime_options=GVRuntimeOptions(
            controls=fixed_controls,
            output_root=validated_options.output_root,
            timeout_seconds=validated_options.timeout_seconds,
        ),
        controls=fixed_controls,
        sweep=sweep,
        channels=channels,
        manifest=manifest_payload,
        manifest_path=manifest_path,
        hdf5_path=hdf5_path,
        runtime_manifests=tuple(runtime_manifests),
        plan_manifests=tuple(plan_manifests),
        execution_manifests=tuple(execution_manifests),
        work_dirs=tuple(work_dirs),
        runtime_seconds=runtime_seconds,
        status="planned" if dry_run else "completed",
    )


def _default_campaign_id() -> str:
    return "gv-sampling-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "_STRETCHING_AXIS",
    "_STRETCHING_REQUIRED_CONTROLS",
    "_BUCKLING_AXIS",
    "_REQUIRED_CONTROLS",
    "GVCommandFailure",
    "GVCwdValidationError",
    "GVLogScanFailure",
    "GVSamplingFailure",
    "GVSamplingExtractionError",
    "GVSamplingPlanError",
    "GVTimeoutFailure",
    "LogScanIssue",
    "SamplingCommand",
    "SamplingExecutionResult",
    "SamplingRun",
    "GVSamplingPlan",
    "build_sampling_plan",
    "execute_sampling_plan",
    "extract_sampling_channels",
    "merge_sampling_channels",
    "scan_log_file",
    "scan_log_text",
    "extract_buckling_lane",
    "parse_buckling_lane_channels",
    "parse_buckling_lane_controls",
    "parse_buckling_sampling_lane",
    "extract_stretching_lane",
    "parse_stretching_lane_channels",
    "parse_stretching_lane_controls",
    "parse_stretching_sampling_lane",
    "GVMaterialGeometry",
    "GVRuntimeOptions",
    "GVSampleResult",
    "GVSweep",
    "GV_SAMPLING_EXPERIMENTS",
    "validate_gv_experiment",
    "sample_gv",
    "validate_explicit_controls",
    "validate_geometry",
    "validate_material_parameters",
    "buckling",
    "stretching",
    "default_sampling_timeout_seconds",
]
