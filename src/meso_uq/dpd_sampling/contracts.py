from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


DPD_SAMPLING_BATCH_SCHEMA_VERSION = "meso_uq.dpd_sampling.batch_request.v1"
DPD_SAMPLING_RENDER_SCHEMA_VERSION = "meso_uq.dpd_sampling.render_result.v1"
DPD_SAMPLING_VALIDATION_SCHEMA_VERSION = "meso_uq.dpd_sampling.validation_report.v1"


def _coerce_path(value: str | Path, *, field_name: str) -> Path:
    path = Path(value)
    if not str(path).strip():
        raise ValueError(f"{field_name} must be a non-empty path.")
    return path


def _coerce_jsonable_payload(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    return {str(key): val for key, val in value.items()}


def _normalise_counts(payload: Mapping[str, int], *, field_name: str) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in payload.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} values must be non-negative integers.")
        normalized[key_text] = value
    return normalized


@dataclass(frozen=True)
class DPDDataRef:
    "Reference to an expected DPD HDF5/manifest pair."

    family: str
    dataset_id: str
    hdf5_path: Path
    manifest_path: Path

    def __post_init__(self) -> None:
        family = str(self.family).strip().lower()
        if family not in {"gv", "emb"}:
            raise ValueError("DPD dataset family must be either 'gv' or 'emb'.")
        object.__setattr__(self, "family", family)

        dataset_id = str(self.dataset_id).strip()
        if not dataset_id:
            raise ValueError("DPD dataset_id must be a non-empty string.")
        object.__setattr__(self, "dataset_id", dataset_id)

        object.__setattr__(self, "hdf5_path", _coerce_path(self.hdf5_path, field_name="hdf5_path"))
        object.__setattr__(self, "manifest_path", _coerce_path(self.manifest_path, field_name="manifest_path"))

    def as_manifest(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "dataset_id": self.dataset_id,
            "hdf5_path": str(self.hdf5_path),
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True)
class DPDCandidateManifest:
    "Contract describing one selected DPD candidate in a batch."

    candidate_id: str
    family: str
    platform: str
    output_root: Path
    campaign_root: Path
    normalized_payload: dict[str, Any]
    expected_hdf5_datasets: tuple[DPDDataRef, ...]
    active_learning_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())
        if not self.candidate_id:
            raise ValueError("DPD candidate_id must be a non-empty string.")

        family = str(self.family).strip().lower()
        if family not in {"gv", "emb"}:
            raise ValueError("DPD candidate family must be either 'gv' or 'emb'.")
        object.__setattr__(self, "family", family)

        object.__setattr__(self, "platform", str(self.platform).strip().lower())
        if self.platform not in {"karolina", "vega"}:
            raise ValueError("DPD candidate platform must be 'karolina' or 'vega'.")

        object.__setattr__(self, "output_root", _coerce_path(self.output_root, field_name="output_root"))
        object.__setattr__(self, "campaign_root", _coerce_path(self.campaign_root, field_name="campaign_root"))

        object.__setattr__(self, "normalized_payload", _coerce_jsonable_payload(self.normalized_payload, field_name="normalized_payload"))
        object.__setattr__(
            self,
            "active_learning_metadata",
            _coerce_jsonable_payload(self.active_learning_metadata, field_name="active_learning_metadata"),
        )
        object.__setattr__(self, "expected_hdf5_datasets", tuple(self.expected_hdf5_datasets))
        if not self.expected_hdf5_datasets:
            raise ValueError("DPD candidate manifest must include at least one expected_hdf5 dataset.")

    @property
    def output_dir(self) -> Path:
        return self.output_root

    def as_manifest(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "platform": self.platform,
            "campaign_root": str(self.campaign_root),
            "output_root": str(self.output_root),
            "normalized_payload": self.normalized_payload,
            "active_learning_metadata": dict(self.active_learning_metadata),
            "expected_hdf5_datasets": [item.as_manifest() for item in self.expected_hdf5_datasets],
        }


@dataclass(frozen=True)
class DPDSamplingBatchRequest:
    "Contract representing a validated batch request prior to rendering."

    batch_id: str
    run_id: str
    iteration: str
    family: str
    platform: str
    walltime: str
    walltime_seconds: int
    gpu_count: int
    provenance_tags: dict[str, Any]
    campaign_root: Path
    candidate_manifests: tuple[DPDCandidateManifest, ...]
    defaults: dict[str, Any]
    _gv_handoff: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_id", str(self.batch_id).strip())
        if not self.batch_id:
            raise ValueError("DPD batch_id must be a non-empty string.")

        object.__setattr__(self, "run_id", str(self.run_id).strip())
        if not self.run_id:
            raise ValueError("DPD run_id must be a non-empty string.")

        object.__setattr__(self, "iteration", str(self.iteration).strip())
        if not self.iteration:
            raise ValueError("DPD iteration must be a non-empty string.")

        family = str(self.family).strip().lower()
        if family not in {"gv", "emb"}:
            raise ValueError("DPD batch family must be either 'gv' or 'emb'.")
        object.__setattr__(self, "family", family)

        platform = str(self.platform).strip().lower()
        if platform not in {"karolina", "vega"}:
            raise ValueError("DPD sampling platform must be one of: karolina, vega")
        object.__setattr__(self, "platform", platform)

        object.__setattr__(self, "walltime", str(self.walltime).strip())
        if not self.walltime:
            raise ValueError("DPD walltime must be a non-empty string.")
        if not isinstance(self.walltime_seconds, int) or self.walltime_seconds <= 0:
            raise ValueError("DPD walltime_seconds must be a positive integer.")
        object.__setattr__(self, "walltime_seconds", int(self.walltime_seconds))

        if not isinstance(self.gpu_count, int) or self.gpu_count <= 0:
            raise ValueError("DPD gpu_count must be a positive integer.")
        object.__setattr__(self, "gpu_count", int(self.gpu_count))

        object.__setattr__(self, "provenance_tags", _coerce_jsonable_payload(self.provenance_tags, field_name="provenance_tags"))
        object.__setattr__(self, "campaign_root", _coerce_path(self.campaign_root, field_name="campaign_root"))
        object.__setattr__(self, "candidate_manifests", tuple(self.candidate_manifests))
        if not self.candidate_manifests:
            raise ValueError("DPD batch must contain at least one candidate manifest.")

        object.__setattr__(
            self,
            "defaults",
            _coerce_jsonable_payload(self.defaults, field_name="defaults") if self.defaults else {},
        )

    @property
    def manifest_path(self) -> Path:
        return self.campaign_root / "dpd_sampling_batch_manifest.json"

    @property
    def validation_report_path(self) -> Path:
        return self.campaign_root / "dpd_sampling_validation_report.json"

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": DPD_SAMPLING_BATCH_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "family": self.family,
            "platform": self.platform,
            "walltime": self.walltime,
            "walltime_seconds": self.walltime_seconds,
            "gpu_count": self.gpu_count,
            "provenance_tags": dict(self.provenance_tags),
            "campaign_root": str(self.campaign_root),
            "defaults": dict(self.defaults),
            "candidate_manifests": [item.as_manifest() for item in self.candidate_manifests],
            "submission": {
                "submitted": False,
                "submission_commands": [],
            },
        }


@dataclass(frozen=True)
class DPDValidationReport:
    "Contract capturing deterministic validation statistics for a batch."

    batch_id: str
    run_id: str
    iteration: str
    family: str
    candidates: int
    family_distribution: dict[str, int]
    platform_distribution: dict[str, int]
    runtime_distribution: dict[str, int]
    mixed_family_rejections: tuple[str, ...]
    scheduler_owned_field_rejections: tuple[str, ...]
    rejected_candidates: tuple[str, ...]
    expected_hdf5_refs: tuple[DPDDataRef, ...]
    submission: dict[str, Any]
    candidate_lineage: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_distribution",
            _normalise_counts(self.family_distribution, field_name="family_distribution"),
        )
        object.__setattr__(
            self,
            "platform_distribution",
            _normalise_counts(self.platform_distribution, field_name="platform_distribution"),
        )
        object.__setattr__(
            self,
            "runtime_distribution",
            _normalise_counts(self.runtime_distribution, field_name="runtime_distribution"),
        )
        object.__setattr__(self, "mixed_family_rejections", tuple(self.mixed_family_rejections))
        object.__setattr__(
            self,
            "scheduler_owned_field_rejections",
            tuple(self.scheduler_owned_field_rejections),
        )
        object.__setattr__(self, "rejected_candidates", tuple(self.rejected_candidates))
        object.__setattr__(
            self,
            "candidate_lineage",
            tuple(_coerce_jsonable_payload(item, field_name="candidate_lineage item") for item in self.candidate_lineage),
        )
        object.__setattr__(self, "expected_hdf5_refs", tuple(self.expected_hdf5_refs))
        object.__setattr__(self, "batch_id", str(self.batch_id).strip())
        object.__setattr__(self, "run_id", str(self.run_id).strip())
        object.__setattr__(self, "iteration", str(self.iteration).strip())
        if not self.batch_id:
            raise ValueError("DPD validation report batch_id must be a non-empty string.")
        if not self.run_id:
            raise ValueError("DPD validation report run_id must be a non-empty string.")
        if not self.iteration:
            raise ValueError("DPD validation report iteration must be a non-empty string.")
        family = str(self.family).strip().lower()
        if family not in {"gv", "emb"}:
            raise ValueError("DPD validation report family must be either 'gv' or 'emb'.")
        object.__setattr__(self, "family", family)
        if not isinstance(self.candidates, int) or self.candidates < 0:
            raise ValueError("DPD validation report candidate count must be a non-negative integer.")
        object.__setattr__(self, "submission", _coerce_jsonable_payload(self.submission, field_name="submission"))

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": DPD_SAMPLING_VALIDATION_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "family": self.family,
            "candidates": self.candidates,
            "family_distribution": dict(self.family_distribution),
            "platform_distribution": dict(self.platform_distribution),
            "runtime_distribution": dict(self.runtime_distribution),
            "mixed_family_rejections": list(self.mixed_family_rejections),
            "scheduler_owned_field_rejections": list(self.scheduler_owned_field_rejections),
            "rejected_candidates": list(self.rejected_candidates),
            "candidate_lineage": [dict(item) for item in self.candidate_lineage],
            "expected_hdf5_refs": [item.as_manifest() for item in self.expected_hdf5_refs],
            "submission": dict(self.submission),
        }


@dataclass(frozen=True)
class DPDSamplingRenderResult:
    "Artifacts produced by rendering a DPD sampling batch."

    batch_request: DPDSamplingBatchRequest
    campaign_root: Path
    rendered_manifest_paths: tuple[Path, ...]
    plot_paths: tuple[Path, ...]
    validation_report: DPDValidationReport
    plot_sidecar_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_root", _coerce_path(self.campaign_root, field_name="campaign_root"))
        object.__setattr__(self, "rendered_manifest_paths", tuple(self.rendered_manifest_paths))
        object.__setattr__(self, "plot_paths", tuple(self.plot_paths))
        object.__setattr__(self, "plot_sidecar_paths", tuple(self.plot_sidecar_paths))

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": DPD_SAMPLING_RENDER_SCHEMA_VERSION,
            "batch_request": self.batch_request.as_manifest(),
            "campaign_root": str(self.campaign_root),
            "rendered_manifests": [str(path) for path in self.rendered_manifest_paths],
            "plots": [str(path) for path in self.plot_paths],
            "plot_sidecars": [str(path) for path in self.plot_sidecar_paths],
            "validation_report": self.validation_report.as_manifest(),
            "submission": {
                "submitted": False,
                "submission_commands": [],
            },
        }


__all__ = [
    "DPDDataRef",
    "DPDCandidateManifest",
    "DPDSamplingBatchRequest",
    "DPDValidationReport",
    "DPDSamplingRenderResult",
    "DPD_SAMPLING_BATCH_SCHEMA_VERSION",
    "DPD_SAMPLING_RENDER_SCHEMA_VERSION",
    "DPD_SAMPLING_VALIDATION_SCHEMA_VERSION",
]
