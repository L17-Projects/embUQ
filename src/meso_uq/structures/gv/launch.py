from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from meso_uq.core import Platform, coerce_platform
from meso_uq.experiments import canonical_dataset_id
from meso_uq.platforms import validate_platform_path_policy
from meso_uq.references.gv_common import control_identifier, format_float
from meso_uq.scheduler_routing import parse_slurm_time_limit

from .geometries import build_geometry
from .numerical_data import (
    GV_NUMERICAL_DATASET_DIRNAME,
    GV_NUMERICAL_DATASET_HDF5_FILENAME,
    GV_NUMERICAL_DATASET_MANIFEST_FILENAME,
    parse_gv_dataset_id,
    validate_gv_numerical_dataset_id,
)
from .parameters import GV_MATERIAL_PARAMETER_NAMES
from .sampling.types import GVMaterialGeometry, GVSweep
from .sampling.validation import validate_sample_gv_request


GV_LAUNCH_MANIFEST_SCHEMA_VERSION = 1
GV_LAUNCH_RENDER_MANIFEST_SCHEMA_VERSION = 1
GV_LAUNCH_RENDER_MANIFEST_FILENAME = "gv_launch_campaign_manifest.json"
_GV_LAUNCH_SUPPORTED_PLATFORMS = frozenset(
    {
        Platform.VEGA,
        Platform.KAROLINA,
        Platform.GENERIC_SLURM,
    }
)
_GV_LAUNCH_RENDER_PLATFORMS = frozenset(
    {
        Platform.VEGA,
        Platform.KAROLINA,
    }
)


@dataclass(frozen=True)
class GVLaunchRequest:
    experiment: str
    geometry: GVMaterialGeometry
    geometry_id: str
    material_parameters: Mapping[str, float]
    fixed_controls: Mapping[str, float]
    sweep: GVSweep
    platform: Platform
    output_root: Path
    campaign_id: str
    walltime: str
    walltime_seconds: int
    gpu_count: int
    provenance_tags: Mapping[str, str]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": GV_LAUNCH_MANIFEST_SCHEMA_VERSION,
            "structure": "gv",
            "experiment": self.experiment,
            "geometry": {
                "id": self.geometry_id,
                "radGV": self.geometry.radGV,
                "height": self.geometry.height,
            },
            "material_parameters": dict(self.material_parameters),
            "fixed_controls": dict(self.fixed_controls),
            "sweep": {
                "axis": self.sweep.axis,
                "values": list(self.sweep.values),
            },
            "platform": self.platform.value,
            "output_root": str(self.output_root),
            "campaign_id": self.campaign_id,
            "walltime": self.walltime,
            "walltime_seconds": self.walltime_seconds,
            "gpu_count": self.gpu_count,
            "provenance_tags": dict(self.provenance_tags),
        }


@dataclass(frozen=True)
class GVLaunchRunManifest:
    output_id: str
    controls: Mapping[str, float]
    dataset_id: str
    manifest_path: Path
    hdf5_path: Path

    def to_manifest(self) -> dict[str, Any]:
        return {
            "output_id": self.output_id,
            "controls": dict(self.controls),
            "dataset_id": self.dataset_id,
            "manifest_path": str(self.manifest_path),
            "hdf5_path": str(self.hdf5_path),
        }


@dataclass(frozen=True)
class GVLaunchCampaignManifest:
    request: GVLaunchRequest
    artifact_scope: str
    output_id: str
    dataset_id: str
    manifest_path: Path
    hdf5_path: Path
    runs: tuple[GVLaunchRunManifest, ...]

    def to_manifest(self) -> dict[str, Any]:
        payload = self.request.to_manifest()
        payload.update(
            {
                "artifact_scope": self.artifact_scope,
                "output_id": self.output_id,
                "dataset_id": self.dataset_id,
                "manifest_path": str(self.manifest_path),
                "hdf5_path": str(self.hdf5_path),
                "runs": [run.to_manifest() for run in self.runs],
            }
        )
        return payload


@dataclass(frozen=True)
class GVLaunchSchedulerScript:
    platform: Platform
    renderer: str
    scheduler: str
    script_path: Path
    runtime_script: str
    runtime_environment_path: str
    gpu_resource_directives: tuple[str, ...]
    operator_checks: tuple[str, ...]
    job_name: str
    array_size: int
    gpu_count: int
    walltime: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "platform": self.platform.value,
            "renderer": self.renderer,
            "scheduler": self.scheduler,
            "script_path": str(self.script_path),
            "runtime_script": self.runtime_script,
            "runtime_environment_path": self.runtime_environment_path,
            "gpu_resource_directives": list(self.gpu_resource_directives),
            "operator_checks": list(self.operator_checks),
            "job_name": self.job_name,
            "array_size": self.array_size,
            "gpu_count": self.gpu_count,
            "walltime": self.walltime,
            "submits_jobs": False,
        }


@dataclass(frozen=True)
class GVLaunchRenderedCampaign:
    request: GVLaunchRequest
    campaign_manifest: GVLaunchCampaignManifest
    campaign_dir: Path
    manifest_path: Path
    scheduler_scripts: tuple[GVLaunchSchedulerScript, ...]
    git_state: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        payload = self.campaign_manifest.to_manifest()
        scripts = [script.to_manifest() for script in self.scheduler_scripts]
        payload.update(
            {
                "render_manifest_schema_version": GV_LAUNCH_RENDER_MANIFEST_SCHEMA_VERSION,
                "campaign_dir": str(self.campaign_dir),
                "campaign_manifest_path": str(self.manifest_path),
                "platform_renderers": [
                    {
                        "platform": script.platform.value,
                        "renderer": script.renderer,
                        "scheduler": script.scheduler,
                    }
                    for script in self.scheduler_scripts
                ],
                "generated_script_paths": [script["script_path"] for script in scripts],
                "generated_files": [str(self.manifest_path), *[script["script_path"] for script in scripts]],
                "scheduler_scripts": scripts,
                "output_paths": {
                    "campaign_dir": str(self.campaign_dir),
                    "manifest_path": str(self.manifest_path),
                    "datasets_root": str(self.campaign_dir / GV_NUMERICAL_DATASET_DIRNAME),
                    "campaign_dataset_manifest": str(self.campaign_manifest.manifest_path),
                    "campaign_hdf5": str(self.campaign_manifest.hdf5_path),
                },
                "expected_hdf5_datasets": {
                    "campaign": {
                        "dataset_id": self.campaign_manifest.dataset_id,
                        "hdf5_path": str(self.campaign_manifest.hdf5_path),
                    },
                    "runs": [
                        {
                            "dataset_id": run.dataset_id,
                            "hdf5_path": str(run.hdf5_path),
                            "manifest_path": str(run.manifest_path),
                        }
                        for run in self.campaign_manifest.runs
                    ],
                },
                "provenance": {
                    "tags": dict(self.request.provenance_tags),
                    "renderer": "meso_uq.structures.gv.launch.render_gv_launch_campaign",
                    "git": dict(self.git_state),
                },
                "submission": {
                    "submitted": False,
                    "submission_commands": [],
                },
            }
        )
        return payload


def validate_gv_launch_request(
    *,
    experiment: object,
    material_parameters: Mapping[str, object],
    geometry: GVMaterialGeometry | Mapping[str, object] | None = None,
    controls: Mapping[str, object] | None,
    platform: Platform | str,
    output_root: str | Path,
    walltime: str,
    gpu_count: int,
    provenance_tags: Mapping[str, object],
    radGV: float | None = None,
    height: float | None = None,
) -> GVLaunchRequest:
    (
        validated_experiment,
        validated_materials,
        validated_geometry,
        fixed_controls,
        sweep,
        _runtime_options,
    ) = validate_sample_gv_request(
        experiment=experiment,
        material_parameters=material_parameters,
        geometry=geometry,
        controls=controls,
        runtime_options=None,
        radGV=radGV,
        height=height,
    )
    _validate_launch_material_parameters(validated_materials)
    _validate_unique_sweep_values(sweep)
    geometry_id = build_geometry(
        radius=validated_geometry.radGV,
        height=validated_geometry.height,
        source="gv_launch_request",
    ).id
    normalized_platform = _normalize_launch_platform(platform)
    normalized_output_root = _normalize_output_root(output_root=output_root, platform=normalized_platform)
    campaign_id = _derive_campaign_id(normalized_output_root)
    normalized_walltime, walltime_seconds = _normalize_walltime(walltime)
    normalized_gpu_count = _normalize_gpu_count(gpu_count)
    normalized_tags = _normalize_provenance_tags(provenance_tags)

    return GVLaunchRequest(
        experiment=validated_experiment,
        geometry=validated_geometry,
        geometry_id=geometry_id,
        material_parameters=validated_materials,
        fixed_controls=fixed_controls,
        sweep=sweep,
        platform=normalized_platform,
        output_root=normalized_output_root,
        campaign_id=campaign_id,
        walltime=normalized_walltime,
        walltime_seconds=walltime_seconds,
        gpu_count=normalized_gpu_count,
        provenance_tags=normalized_tags,
    )


def build_gv_launch_campaign_manifest(request: GVLaunchRequest) -> GVLaunchCampaignManifest:
    if not isinstance(request, GVLaunchRequest):
        raise ValueError("request must be a GVLaunchRequest.")

    runs = tuple(_build_run_manifest(request=request, sweep_value=value) for value in request.sweep.values)
    artifact_scope = "control_point" if len(runs) == 1 else "sweep"
    output_id = _campaign_output_id(fixed_controls=request.fixed_controls, sweep=request.sweep)
    dataset_id = canonical_dataset_id("gv", request.experiment, request.geometry_id, output_id)
    manifest_path = _launch_dataset_manifest_path(output_root=request.output_root, dataset_id=dataset_id)
    hdf5_path = _launch_dataset_hdf5_path(output_root=request.output_root, dataset_id=dataset_id)

    return GVLaunchCampaignManifest(
        request=request,
        artifact_scope=artifact_scope,
        output_id=output_id,
        dataset_id=dataset_id,
        manifest_path=manifest_path,
        hdf5_path=hdf5_path,
        runs=runs,
    )


def render_gv_launch_campaign(
    request: GVLaunchRequest,
    *,
    platforms: Platform | str | Iterable[Platform | str] | None = None,
    overwrite: bool = False,
) -> GVLaunchRenderedCampaign:
    """Materialize one validated GV launch request without submitting jobs."""

    return render_gv_launch_campaigns(request, platforms=platforms, overwrite=overwrite)[0]


def render_gv_launch_campaigns(
    requests: GVLaunchRequest | Iterable[GVLaunchRequest],
    *,
    platforms: Platform | str | Iterable[Platform | str] | None = None,
    overwrite: bool = False,
) -> tuple[GVLaunchRenderedCampaign, ...]:
    """Materialize one or more GV launch requests as manifests and scheduler scripts."""

    normalized_requests = _normalize_render_requests(requests)
    _validate_unique_campaign_dirs(normalized_requests)
    return tuple(
        _render_one_gv_launch_campaign(request=request, platforms=platforms, overwrite=overwrite)
        for request in normalized_requests
    )


def _render_one_gv_launch_campaign(
    *,
    request: GVLaunchRequest,
    platforms: Platform | str | Iterable[Platform | str] | None,
    overwrite: bool,
) -> GVLaunchRenderedCampaign:
    target_platforms = _normalize_render_platforms(platforms=platforms, default_platform=request.platform)
    for platform in target_platforms:
        _validate_render_output_root_for_platform(output_root=request.output_root, platform=platform)

    campaign_manifest = build_gv_launch_campaign_manifest(request)
    campaign_dir = request.output_root
    _prepare_campaign_dir(campaign_dir=campaign_dir, overwrite=overwrite)

    (campaign_dir / "logs").mkdir(parents=True, exist_ok=True)
    scripts = tuple(
        _write_scheduler_script(
            request=request,
            campaign_manifest=campaign_manifest,
            platform=platform,
            campaign_dir=campaign_dir,
        )
        for platform in target_platforms
    )

    manifest_path = campaign_dir / GV_LAUNCH_RENDER_MANIFEST_FILENAME
    rendered = GVLaunchRenderedCampaign(
        request=request,
        campaign_manifest=campaign_manifest,
        campaign_dir=campaign_dir,
        manifest_path=manifest_path,
        scheduler_scripts=scripts,
        git_state=_repo_git_state(),
    )
    manifest_path.write_text(
        json.dumps(rendered.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _validate_rendered_campaign_files(rendered)
    return rendered


def _build_run_manifest(*, request: GVLaunchRequest, sweep_value: float) -> GVLaunchRunManifest:
    controls = dict(request.fixed_controls)
    controls[request.sweep.axis] = float(sweep_value)
    output_id = control_identifier(controls)
    dataset_id = canonical_dataset_id("gv", request.experiment, request.geometry_id, output_id)
    return GVLaunchRunManifest(
        output_id=output_id,
        controls=controls,
        dataset_id=dataset_id,
        manifest_path=_launch_dataset_manifest_path(output_root=request.output_root, dataset_id=dataset_id),
        hdf5_path=_launch_dataset_hdf5_path(output_root=request.output_root, dataset_id=dataset_id),
    )


def _campaign_output_id(*, fixed_controls: Mapping[str, float], sweep: GVSweep) -> str:
    if len(sweep.values) == 1:
        controls = dict(fixed_controls)
        controls[sweep.axis] = float(sweep.values[0])
        return control_identifier(controls)

    parts: list[str] = []
    fixed_control_id = control_identifier(fixed_controls)
    if fixed_control_id != "default":
        parts.append(fixed_control_id)
    parts.append(f"sweep_{sweep.axis}")
    value_token = "__".join(format_float(value) for value in sweep.values)
    parts.append(f"values_{value_token}")
    return "__".join(parts)


def _normalize_render_requests(
    requests: GVLaunchRequest | Iterable[GVLaunchRequest],
) -> tuple[GVLaunchRequest, ...]:
    if isinstance(requests, GVLaunchRequest):
        return (requests,)
    normalized = tuple(requests)
    if not normalized:
        raise ValueError("At least one GVLaunchRequest is required.")
    for request in normalized:
        if not isinstance(request, GVLaunchRequest):
            raise ValueError("All GV launch render inputs must be GVLaunchRequest instances.")
    return normalized


def _validate_unique_campaign_dirs(requests: Sequence[GVLaunchRequest]) -> None:
    seen: dict[Path, str] = {}
    for request in requests:
        key = request.output_root
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "GV launch batch contains duplicate campaign output roots: "
                f"{key} ({previous!r} and {request.campaign_id!r})."
            )
        seen[key] = request.campaign_id


def _normalize_render_platforms(
    *,
    platforms: Platform | str | Iterable[Platform | str] | None,
    default_platform: Platform,
) -> tuple[Platform, ...]:
    if platforms is None:
        raw_platforms: tuple[Platform | str, ...] = (default_platform,)
    elif isinstance(platforms, (Platform, str)):
        raw_platforms = (platforms,)
    else:
        raw_platforms = tuple(platforms)
    if not raw_platforms:
        raise ValueError("At least one GV launch render platform is required.")

    normalized: list[Platform] = []
    seen: set[Platform] = set()
    for raw_platform in raw_platforms:
        platform = _normalize_launch_platform(raw_platform)
        if platform not in _GV_LAUNCH_RENDER_PLATFORMS:
            supported = ", ".join(sorted(item.value for item in _GV_LAUNCH_RENDER_PLATFORMS))
            raise ValueError(f"GV launch render platform must be one of: {supported}.")
        if platform not in seen:
            normalized.append(platform)
            seen.add(platform)
    return tuple(normalized)


def _validate_render_output_root_for_platform(*, output_root: Path, platform: Platform) -> None:
    errors = validate_platform_path_policy(
        platform,
        {"output_root": output_root.as_posix()},
        label="gv_launch_renderer",
    )
    if errors:
        raise ValueError(errors[0])


def _prepare_campaign_dir(*, campaign_dir: Path, overwrite: bool) -> None:
    if campaign_dir.exists():
        if not campaign_dir.is_dir():
            raise ValueError(f"GV launch campaign path exists and is not a directory: {campaign_dir}.")
        if not overwrite:
            raise ValueError(
                "GV launch campaign directory already exists; choose a new campaign id "
                f"or pass overwrite=True: {campaign_dir}."
            )
    campaign_dir.mkdir(parents=True, exist_ok=True)


def _write_scheduler_script(
    *,
    request: GVLaunchRequest,
    campaign_manifest: GVLaunchCampaignManifest,
    platform: Platform,
    campaign_dir: Path,
) -> GVLaunchSchedulerScript:
    renderer = _renderer_name(platform)
    job_name = _slurm_job_name(request.campaign_id)
    script_dir = campaign_dir / "scripts" / platform.value
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{_path_token(request.campaign_id)}.sbatch"
    runtime_script = _platform_runtime_script(platform)
    script_path.write_text(
        _render_slurm_script(
            request=request,
            campaign_manifest=campaign_manifest,
            platform=platform,
            job_name=job_name,
            renderer=renderer,
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return GVLaunchSchedulerScript(
        platform=platform,
        renderer=renderer,
        scheduler="slurm",
        script_path=script_path,
        runtime_script=runtime_script,
        runtime_environment_path=_platform_runtime_environment_path(platform),
        gpu_resource_directives=_gpu_resource_directives(platform=platform, gpu_count=request.gpu_count),
        operator_checks=_platform_operator_checks(platform),
        job_name=job_name,
        array_size=len(campaign_manifest.runs),
        gpu_count=request.gpu_count,
        walltime=request.walltime,
    )


def _validate_rendered_campaign_files(rendered: GVLaunchRenderedCampaign) -> None:
    expected = [rendered.manifest_path, *(script.script_path for script in rendered.scheduler_scripts)]
    seen: set[Path] = set()
    missing: list[str] = []
    duplicates: list[str] = []
    for path in expected:
        if path in seen:
            duplicates.append(path.as_posix())
        seen.add(path)
        if not path.is_file():
            missing.append(path.as_posix())
    if duplicates:
        raise RuntimeError("GV launch render produced duplicate generated file paths: " + ", ".join(duplicates))
    if missing:
        raise RuntimeError("GV launch render did not materialize expected files: " + ", ".join(missing))
    for script in rendered.scheduler_scripts:
        if script.array_size != len(rendered.campaign_manifest.runs):
            raise RuntimeError(
                "GV launch render produced inconsistent scheduler array size "
                f"for {script.platform.value}: {script.array_size} != {len(rendered.campaign_manifest.runs)}."
            )


def _render_slurm_script(
    *,
    request: GVLaunchRequest,
    campaign_manifest: GVLaunchCampaignManifest,
    platform: Platform,
    job_name: str,
    renderer: str,
) -> str:
    header = _slurm_header(request=request, platform=platform, job_name=job_name)
    setup = _platform_setup(platform)
    run_arrays = _run_arrays(campaign_manifest)
    material_args = _shell_arg_array(_material_cli_args(request.material_parameters))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Generated by meso_uq.structures.gv.launch.render_gv_launch_campaign.",
            "# This script is submission-free; submit it explicitly from the scheduler if needed.",
            f"# GV launch renderer: {renderer}",
            *header,
            "",
            "set -euo pipefail",
            "",
            'REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"',
            'if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then',
            '  echo "Set REPO_ROOT explicitly or submit the job from the repo root." >&2',
            "  exit 2",
            "fi",
            'cd "${REPO_ROOT}"',
            "",
            f"CAMPAIGN_DIR={_shell_quote(request.output_root.as_posix())}",
            'mkdir -p "${CAMPAIGN_DIR}/logs"',
            'PYTHON_BIN="${PYTHON_BIN:-python}"',
            'export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"',
            "",
            *setup,
            "",
            f"EXPERIMENT={_shell_quote(request.experiment)}",
            f"GEOMETRY_RADIUS={_shell_quote(str(request.geometry.radGV))}",
            f"GEOMETRY_HEIGHT={_shell_quote(str(request.geometry.height))}",
            f"CAMPAIGN_DATASET_ID={_shell_quote(campaign_manifest.dataset_id)}",
            f"CAMPAIGN_HDF5_PATH={_shell_quote(campaign_manifest.hdf5_path.as_posix())}",
            *run_arrays,
            f"MATERIAL_ARGV=({material_args})",
            "",
            'RUN_INDEX="${SLURM_ARRAY_TASK_ID:-0}"',
            'if (( RUN_INDEX < 0 || RUN_INDEX >= ${#RUN_DATASET_IDS[@]} )); then',
            '  echo "Invalid GV launch run index: ${RUN_INDEX}" >&2',
            "  exit 2",
            "fi",
            "",
            'IFS=" " read -r -a CONTROL_ARGV <<< "${RUN_CONTROL_ARGS[$RUN_INDEX]}"',
            'RUN_OUTPUT_ROOT="${CAMPAIGN_DIR}/runtime/${RUN_OUTPUT_IDS[$RUN_INDEX]}"',
            "",
            "command=(",
            '  "${PYTHON_BIN}"',
            f"  {_platform_runtime_script(platform)}",
            f"  --site {_shell_quote(platform.value)}",
            '  --experiment "${EXPERIMENT}"',
            '  --radius "${GEOMETRY_RADIUS}"',
            '  --height "${GEOMETRY_HEIGHT}"',
            '  --output-root "${RUN_OUTPUT_ROOT}"',
            ")",
            'command+=("${CONTROL_ARGV[@]}" "${MATERIAL_ARGV[@]}")',
            "",
            'echo "[gv-launch] campaign_dataset_id=${CAMPAIGN_DATASET_ID}"',
            'echo "[gv-launch] campaign_hdf5_path=${CAMPAIGN_HDF5_PATH}"',
            'echo "[gv-launch] run_dataset_id=${RUN_DATASET_IDS[$RUN_INDEX]}"',
            'echo "[gv-launch] run_manifest_path=${RUN_MANIFEST_PATHS[$RUN_INDEX]}"',
            'echo "[gv-launch] run_hdf5_path=${RUN_HDF5_PATHS[$RUN_INDEX]}"',
            '"${command[@]}"',
            "",
        ]
    )


def _slurm_header(*, request: GVLaunchRequest, platform: Platform, job_name: str) -> list[str]:
    output_path = (request.output_root / "logs" / "%x-%A_%a.out").as_posix()
    error_path = (request.output_root / "logs" / "%x-%A_%a.err").as_posix()
    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={request.walltime}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
    ]
    if platform == Platform.KAROLINA:
        lines.extend(
            [
                "#SBATCH --account=eu-26-17",
                "#SBATCH --partition=qgpu",
                "#SBATCH --cpus-per-task=8",
                f"#SBATCH --gpus={request.gpu_count}",
            ]
        )
    else:
        lines.extend(
            [
                "#SBATCH --partition=gpu",
                "#SBATCH --cpus-per-task=4",
                f"#SBATCH --gres=gpu:{request.gpu_count}",
            ]
        )
    if len(request.sweep.values) > 1:
        lines.append(f"#SBATCH --array=0-{len(request.sweep.values) - 1}")
    lines.extend(
        [
            f"#SBATCH --output={output_path}",
            f"#SBATCH --error={error_path}",
        ]
    )
    return lines


def _platform_runtime_script(platform: Platform) -> str:
    return "scripts/platforms/hpc/run_gv_runtime.py"


def _platform_runtime_environment_path(platform: Platform) -> str:
    if platform == Platform.KAROLINA:
        return "${MESOUQ_GV_ENV_SCRIPT:-${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh}"
    return "${MESOUQ_GV_ENV_SCRIPT:-${REPO_ROOT}/_vega/gv_venv/env.sh}"


def _gpu_resource_directives(*, platform: Platform, gpu_count: int) -> tuple[str, ...]:
    if platform == Platform.KAROLINA:
        return (f"#SBATCH --gpus={gpu_count}",)
    return (f"#SBATCH --gres=gpu:{gpu_count}",)


def _platform_operator_checks(platform: Platform) -> tuple[str, ...]:
    if platform == Platform.VEGA:
        return (
            "Vega maintenance state and partition availability must be checked by the operator before submission.",
            "Verify that the Vega module stack and _vega/gv_venv/env.sh are current for the checkout.",
        )
    return (
        "Verify Karolina project allocation and qgpu availability before submission.",
        "Verify MESOUQ_SITE_RUNTIME_ROOT contains the GV runtime environment for this checkout.",
    )


def _platform_setup(platform: Platform) -> list[str]:
    if platform == Platform.KAROLINA:
        return [
            'source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"',
            'GV_ENV_SCRIPT="${MESOUQ_GV_ENV_SCRIPT:-${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh}"',
            'if [[ ! -f "${GV_ENV_SCRIPT}" ]]; then',
            '  echo "Missing required GV runtime environment: ${GV_ENV_SCRIPT}" >&2',
            "  exit 1",
            "fi",
            'source "${GV_ENV_SCRIPT}"',
            'export MESOUQ_GV_MPI_RANKS="${MESOUQ_GV_MPI_RANKS:-2}"',
            'export MESOUQ_GV_EIGENMODES_MPI_RANKS="${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}"',
            'export MESOUQ_GV_EIGENMODES_DOMAIN_RANKS="${MESOUQ_GV_EIGENMODES_DOMAIN_RANKS:-1,1,1}"',
            'export MESOUQ_SITE="karolina"',
        ]
    return [
        "module purge",
        "module load \\",
        "  Python/3.10.8-GCCcore-12.2.0 \\",
        "  OpenMPI/4.1.4-GCC-12.2.0 \\",
        "  CUDA/12.2.2 \\",
        "  GSL/2.7-GCC-12.2.0 \\",
        "  Eigen/3.4.0-GCCcore-12.2.0 \\",
        "  HDF5/1.14.0-gompi-2022b \\",
        "  MPFR/4.2.0-GCCcore-12.2.0 \\",
        "  GMP/6.2.1-GCCcore-12.2.0",
        'GV_ENV_SCRIPT="${MESOUQ_GV_ENV_SCRIPT:-${REPO_ROOT}/_vega/gv_venv/env.sh}"',
        'if [[ ! -f "${GV_ENV_SCRIPT}" ]]; then',
        '  echo "Missing required GV runtime environment: ${GV_ENV_SCRIPT}" >&2',
        "  exit 1",
        "fi",
        'source "${GV_ENV_SCRIPT}"',
        'export MESOUQ_GV_MPI_RANKS="${MESOUQ_GV_MPI_RANKS:-2}"',
        'export MESOUQ_GV_EIGENMODES_MPI_RANKS="${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}"',
        'export MESOUQ_GV_EIGENMODES_DOMAIN_RANKS="${MESOUQ_GV_EIGENMODES_DOMAIN_RANKS:-1,1,1}"',
        'export MESOUQ_SITE="vega"',
    ]


def _run_arrays(campaign_manifest: GVLaunchCampaignManifest) -> list[str]:
    dataset_ids = _shell_array(run.dataset_id for run in campaign_manifest.runs)
    output_ids = _shell_array(run.output_id for run in campaign_manifest.runs)
    manifest_paths = _shell_array(run.manifest_path.as_posix() for run in campaign_manifest.runs)
    hdf5_paths = _shell_array(run.hdf5_path.as_posix() for run in campaign_manifest.runs)
    control_args = _shell_array(_control_cli_arg_string(run.controls) for run in campaign_manifest.runs)
    return [
        f"RUN_DATASET_IDS=({dataset_ids})",
        f"RUN_OUTPUT_IDS=({output_ids})",
        f"RUN_MANIFEST_PATHS=({manifest_paths})",
        f"RUN_HDF5_PATHS=({hdf5_paths})",
        f"RUN_CONTROL_ARGS=({control_args})",
    ]


def _control_cli_arg_string(controls: Mapping[str, float]) -> str:
    return " ".join(_control_cli_args(controls))


def _control_cli_args(controls: Mapping[str, float]) -> list[str]:
    args: list[str] = []
    for name in sorted(controls):
        args.extend(["--control", f"{name}={controls[name]}"])
    return args


def _material_cli_args(material_parameters: Mapping[str, float]) -> list[str]:
    args: list[str] = []
    for name in GV_MATERIAL_PARAMETER_NAMES:
        args.extend(["--material", f"{name}={material_parameters[name]}"])
    return args


def _shell_array(values: Iterable[str]) -> str:
    return " ".join(_shell_quote(value) for value in values)


def _shell_arg_array(values: Iterable[str]) -> str:
    return " ".join(_shell_quote(value) for value in values)


def _shell_quote(value: str) -> str:
    return shlex.quote(str(value))


def _renderer_name(platform: Platform) -> str:
    return f"gv_launch_renderer:{platform.value}_slurm"


def _slurm_job_name(campaign_id: str) -> str:
    return f"mesouq-gv-{_path_token(campaign_id)}"[:128]


def _path_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value.strip())
    return token or "campaign"


def _repo_git_state() -> dict[str, Any]:
    repo_root = _find_repo_root()
    git_pointer = repo_root / ".git"
    git_dir = _resolve_git_dir(repo_root=repo_root, git_pointer=git_pointer)
    if git_dir is None:
        return {
            "available": False,
            "head": None,
            "branch": None,
            "commit": None,
        }

    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {
            "available": False,
            "head": None,
            "branch": None,
            "commit": None,
        }

    branch = None
    commit = head
    if head.startswith("ref: "):
        ref = head.removeprefix("ref: ").strip()
        branch = ref.removeprefix("refs/heads/")
        commit = _read_git_ref(git_dir=git_dir, ref=ref)
    return {
        "available": commit is not None,
        "head": head,
        "branch": branch,
        "commit": commit,
    }


def _resolve_git_dir(*, repo_root: Path, git_pointer: Path) -> Path | None:
    if git_pointer.is_dir():
        return git_pointer
    try:
        text = git_pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not text.startswith(prefix):
        return None
    git_dir = Path(text.removeprefix(prefix).strip())
    if not git_dir.is_absolute():
        git_dir = (repo_root / git_dir).resolve()
    return git_dir


def _read_git_ref(*, git_dir: Path, ref: str) -> str | None:
    for candidate_dir in _candidate_git_ref_dirs(git_dir):
        ref_path = candidate_dir / ref
        try:
            return ref_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass

        packed_refs = candidate_dir / "packed-refs"
        try:
            lines = packed_refs.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        suffix = f" {ref}"
        for line in lines:
            if line.startswith("#") or not line.endswith(suffix):
                continue
            return line.split(" ", 1)[0]
    return None


def _candidate_git_ref_dirs(git_dir: Path) -> tuple[Path, ...]:
    candidates = [git_dir]
    common_dir_path = git_dir / "commondir"
    try:
        common_dir = Path(common_dir_path.read_text(encoding="utf-8").strip())
    except OSError:
        return tuple(candidates)
    if not common_dir.is_absolute():
        common_dir = (git_dir / common_dir).resolve()
    if common_dir not in candidates:
        candidates.append(common_dir)
    return tuple(candidates)


def _normalize_launch_platform(platform: Platform | str) -> Platform:
    normalized = coerce_platform(platform)
    if normalized == Platform.GENERIC:
        normalized = Platform.GENERIC_SLURM
    if normalized not in _GV_LAUNCH_SUPPORTED_PLATFORMS:
        supported = ", ".join(sorted(item.value for item in _GV_LAUNCH_SUPPORTED_PLATFORMS))
        raise ValueError(f"GV launch platform must be one of: {supported}.")
    return normalized


def _normalize_output_root(*, output_root: str | Path, platform: Platform) -> Path:
    raw = str(output_root).strip()
    if not raw:
        raise ValueError("GV launch output_root must be a non-empty path.")
    path = Path(raw).expanduser()
    if any(part == ".." for part in path.parts):
        raise ValueError("GV launch output_root must not contain path traversal segments.")
    _reject_unsafe_repo_output_root(path)
    errors = validate_platform_path_policy(platform, {"output_root": path.as_posix()}, label="gv_launch")
    if errors:
        raise ValueError(errors[0])
    return path


def _validate_launch_material_parameters(material_parameters: Mapping[str, float]) -> None:
    for name in GV_MATERIAL_PARAMETER_NAMES:
        value = float(material_parameters[name])
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"GV launch material parameter '{name}' must be finite and > 0.")


def _validate_unique_sweep_values(sweep: GVSweep) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in sweep.values:
        value_id = format_float(value)
        if value_id in seen:
            duplicates.append(value_id)
        seen.add(value_id)
    if duplicates:
        duplicate_list = ", ".join(duplicates)
        raise ValueError(
            "GV launch sweep values must produce unique formatted output ids; "
            f"duplicates: {duplicate_list}."
        )


def _reject_unsafe_repo_output_root(path: Path) -> None:
    repo_root = _find_repo_root()
    resolved_path = path.resolve()
    unsafe_roots = (
        repo_root / "gv_simulation_files",
        repo_root / "gv",
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "tests",
    )
    for unsafe_root in unsafe_roots:
        if resolved_path == unsafe_root or unsafe_root in resolved_path.parents:
            raise ValueError(f"GV launch output_root must not be inside repository source roots: {unsafe_root}.")
    if path.name in {"gv", "src", "scripts", "tests", "gv_simulation_files"}:
        raise ValueError("GV launch output_root must not be a repository source directory.")


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repository root for GV launch validation.")


def _derive_campaign_id(output_root: Path) -> str:
    campaign_id = output_root.name.strip()
    if campaign_id in {"", ".", ".."}:
        raise ValueError("GV launch output_root must end in a campaign directory name.")
    if "/" in campaign_id or "\\" in campaign_id or ".." in campaign_id:
        raise ValueError("GV launch campaign_id derived from output_root must be path-safe.")
    return campaign_id


def _launch_dataset_dir(*, output_root: Path, dataset_id: str) -> Path:
    structure, experiment, geometry, control_id = parse_gv_dataset_id(dataset_id)
    validate_gv_numerical_dataset_id(dataset_id, structure=structure)
    return output_root / GV_NUMERICAL_DATASET_DIRNAME / structure / experiment / geometry / control_id


def _launch_dataset_manifest_path(*, output_root: Path, dataset_id: str) -> Path:
    return _launch_dataset_dir(output_root=output_root, dataset_id=dataset_id) / GV_NUMERICAL_DATASET_MANIFEST_FILENAME


def _launch_dataset_hdf5_path(*, output_root: Path, dataset_id: str) -> Path:
    return _launch_dataset_dir(output_root=output_root, dataset_id=dataset_id) / GV_NUMERICAL_DATASET_HDF5_FILENAME


def _normalize_walltime(walltime: str) -> tuple[str, int]:
    text = str(walltime).strip()
    if not text:
        raise ValueError("GV launch walltime must be a non-empty string.")
    try:
        walltime_seconds = parse_slurm_time_limit(text)
    except ValueError as exc:
        raise ValueError(f"GV launch walltime must be a valid SLURM time limit, got {text!r}.") from exc
    return text, walltime_seconds


def _normalize_gpu_count(gpu_count: int) -> int:
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int):
        raise ValueError("GV launch gpu_count must be an integer.")
    if gpu_count <= 0:
        raise ValueError("GV launch gpu_count must be positive.")
    return gpu_count


def _normalize_provenance_tags(provenance_tags: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(provenance_tags, Mapping):
        raise ValueError("GV launch provenance_tags must be a mapping.")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in provenance_tags.items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key:
            raise ValueError("GV launch provenance_tags keys must be non-empty strings.")
        if not value:
            raise ValueError(f"GV launch provenance_tags[{key!r}] must be a non-empty string.")
        normalized[key] = value
    if not normalized:
        raise ValueError("GV launch provenance_tags must include at least one tag.")
    return normalized


__all__ = [
    "GV_LAUNCH_MANIFEST_SCHEMA_VERSION",
    "GV_LAUNCH_RENDER_MANIFEST_FILENAME",
    "GV_LAUNCH_RENDER_MANIFEST_SCHEMA_VERSION",
    "GVLaunchCampaignManifest",
    "GVLaunchRenderedCampaign",
    "GVLaunchRequest",
    "GVLaunchRunManifest",
    "GVLaunchSchedulerScript",
    "build_gv_launch_campaign_manifest",
    "render_gv_launch_campaign",
    "render_gv_launch_campaigns",
    "validate_gv_launch_request",
]
