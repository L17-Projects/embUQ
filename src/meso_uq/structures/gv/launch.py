from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

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
_GV_LAUNCH_SUPPORTED_PLATFORMS = frozenset(
    {
        Platform.VEGA,
        Platform.KAROLINA,
        Platform.GENERIC_SLURM,
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
    "GVLaunchCampaignManifest",
    "GVLaunchRequest",
    "GVLaunchRunManifest",
    "build_gv_launch_campaign_manifest",
    "validate_gv_launch_request",
]
