from __future__ import annotations

"""Deterministic simulation manifest scaffold for Active Learning batches."""

import hashlib
import json
import shutil
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


ACTIVE_LEARNING_SIMULATION_MANIFEST_SCHEMA_VERSION = "meso_uq.active_learning.simulation_manifest.v1"
ACTIVE_LEARNING_SIMULATION_RECORD_SCHEMA_VERSION = "meso_uq.active_learning.simulation_record.v1"
ACTIVE_LEARNING_SUBMISSION_BOUNDARY_SCHEMA_VERSION = "meso_uq.active_learning.submission_boundary.v1"
ACTIVE_LEARNING_QUARANTINE_MANIFEST_SCHEMA_VERSION = "meso_uq.active_learning.failed_quarantine_manifest.v1"
ACTIVE_LEARNING_QUARANTINE_REPORT_SCHEMA_VERSION = "meso_uq.active_learning.failed_quarantine_report.v1"

FAILED_SIMULATION_QUARANTINE_SUBDIR = "quarantine"
FAILED_SIMULATION_MANIFEST_FILENAME = "failed_simulation_manifest.json"
FAILED_SIMULATION_VALIDATION_PNG_FILENAME = "failed_simulation_validation.png"
FAILED_SIMULATION_VALIDATION_PNG_SIDECAR_FILENAME = "failed_simulation_validation.png.json"

_SUPPORTED_FAMILIES = frozenset({"gv", "emb"})
_FAILED_STATUSES = frozenset({"failed", "submission_failed", "cancelled"})
_TERMINAL_STATUSES = frozenset({"success", "failed", "submission_failed", "cancelled"})


def _coerce_text(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            normalized[str(key)] = _normalize_json(nested)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise ValueError(f"Unsupported JSON value type: {type(value).__name__}.")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_normalize_json(dict(payload)), sort_keys=True, separators=(",", ":"))


def _sha256_hex(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _coerce_text_tuple(values: Sequence[object], *, field_name: str) -> tuple[str, ...]:
    items = tuple(_coerce_text(value, field_name=field_name) for value in values)
    return items


def _coerce_provenance_hashes(values: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError("provenance_hashes must be a mapping.")
    coerced: dict[str, str] = {}
    for key, value in values.items():
        coerced[_coerce_text(key, field_name="provenance hash key")] = _coerce_text(
            value,
            field_name="provenance hash value",
        )
    return coerced


@dataclass(frozen=True)
class SubmissionBoundaryState:
    submitted: bool = False
    state: str = "not_submitted"
    submission_commands: tuple[str, ...] | Sequence[object] = field(default_factory=tuple)
    schema_version: str = ACTIVE_LEARNING_SUBMISSION_BOUNDARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "submitted", bool(self.submitted))
        object.__setattr__(self, "state", _coerce_text(self.state, field_name="submission state"))
        object.__setattr__(
            self,
            "submission_commands",
            _coerce_text_tuple(self.submission_commands, field_name="submission command"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "state": self.state,
            "submission_commands": list(self.submission_commands),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubmissionBoundaryState":
        return cls(
            submitted=bool(payload.get("submitted", False)),
            state=str(payload.get("state", "not_submitted")),
            submission_commands=tuple(payload.get("submission_commands", ())),
            schema_version=str(
                payload.get("schema_version", ACTIVE_LEARNING_SUBMISSION_BOUNDARY_SCHEMA_VERSION)
            ),
        )


@dataclass(frozen=True)
class SimulationManifestRecord:
    candidate_id: str
    family: str
    experiment: str
    expected_dataset_refs: tuple[str, ...] | Sequence[object] = field(default_factory=tuple)
    scheduler_artifact_refs: tuple[str, ...] | Sequence[object] = field(default_factory=tuple)
    render_artifact_refs: tuple[str, ...] | Sequence[object] = field(default_factory=tuple)
    submission: SubmissionBoundaryState | Mapping[str, Any] = field(default_factory=SubmissionBoundaryState)
    stage: str = "simulation"
    run_status: str = "pending"
    error_message: str | None = None
    retry_count: int = 0
    provenance_hashes: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_SIMULATION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _coerce_text(self.candidate_id, field_name="candidate_id"))
        family = _coerce_text(self.family, field_name="family").lower()
        if family not in _SUPPORTED_FAMILIES:
            raise ValueError(f"family must be one of: {', '.join(sorted(_SUPPORTED_FAMILIES))}.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "experiment", _coerce_text(self.experiment, field_name="experiment"))
        object.__setattr__(
            self,
            "expected_dataset_refs",
            _coerce_text_tuple(self.expected_dataset_refs, field_name="dataset ref"),
        )
        object.__setattr__(
            self,
            "scheduler_artifact_refs",
            _coerce_text_tuple(self.scheduler_artifact_refs, field_name="scheduler artifact ref"),
        )
        object.__setattr__(
            self,
            "render_artifact_refs",
            _coerce_text_tuple(self.render_artifact_refs, field_name="render artifact ref"),
        )
        if not isinstance(self.submission, SubmissionBoundaryState):
            object.__setattr__(self, "submission", SubmissionBoundaryState.from_dict(self.submission))
        object.__setattr__(self, "stage", _coerce_text(self.stage, field_name="stage"))
        run_status = _coerce_text(self.run_status, field_name="run_status")
        if run_status not in _TERMINAL_STATUSES and run_status not in {"pending", "running", "submitted"}:
            raise ValueError("run_status must be pending, submitted, running, success, failed, submission_failed, or cancelled.")
        object.__setattr__(self, "run_status", run_status)
        if self.error_message is not None:
            object.__setattr__(self, "error_message", _coerce_text(self.error_message, field_name="error_message"))
        if not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer.")
        object.__setattr__(self, "provenance_hashes", _coerce_provenance_hashes(self.provenance_hashes))
        object.__setattr__(self, "metadata", _normalize_json(dict(self.metadata)))

    @property
    def is_failed(self) -> bool:
        return self.run_status in _FAILED_STATUSES or bool(self.error_message)

    @property
    def reason(self) -> str:
        if self.error_message:
            return self.error_message
        return self.run_status

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "experiment": self.experiment,
            "expected_dataset_refs": list(self.expected_dataset_refs),
            "scheduler_artifact_refs": list(self.scheduler_artifact_refs),
            "render_artifact_refs": list(self.render_artifact_refs),
            "submission": self.submission.as_dict(),
            "stage": self.stage,
            "run_status": self.run_status,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "provenance_hashes": dict(self.provenance_hashes),
            "metadata": _normalize_json(dict(self.metadata)),
            "schema_version": self.schema_version,
        }

    def stable_hash(self) -> str:
        return _sha256_hex(self.as_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationManifestRecord":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            family=str(payload["family"]),
            experiment=str(payload["experiment"]),
            expected_dataset_refs=tuple(payload.get("expected_dataset_refs", ())),
            scheduler_artifact_refs=tuple(payload.get("scheduler_artifact_refs", ())),
            render_artifact_refs=tuple(payload.get("render_artifact_refs", ())),
            submission=payload.get("submission", {}),
            stage=str(payload.get("stage", "simulation")),
            run_status=str(payload.get("run_status", "pending")),
            error_message=payload.get("error_message"),
            retry_count=int(payload.get("retry_count", 0)),
            provenance_hashes=payload.get("provenance_hashes", {}),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_SIMULATION_RECORD_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class SimulationBatchManifest:
    run_id: str
    iteration: int
    records: tuple[SimulationManifestRecord, ...] | Sequence[SimulationManifestRecord | Mapping[str, Any]]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_SIMULATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _coerce_text(self.run_id, field_name="run_id"))
        if not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
        records = tuple(
            record if isinstance(record, SimulationManifestRecord) else SimulationManifestRecord.from_dict(record)
            for record in self.records
        )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", _normalize_json(dict(self.metadata)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "record_count": len(self.records),
            "records": [record.as_dict() for record in self.records],
            "metadata": _normalize_json(dict(self.metadata)),
            "schema_version": self.schema_version,
        }

    def stable_hash(self) -> str:
        return _sha256_hex(self.as_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationBatchManifest":
        return cls(
            run_id=str(payload["run_id"]),
            iteration=int(payload["iteration"]),
            records=tuple(payload.get("records", ())),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_SIMULATION_MANIFEST_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class FailedSimulationQuarantineArtifacts:
    quarantine_dir: Path
    manifest_path: Path
    validation_png_path: Path
    validation_sidecar_path: Path
    failed_manifest: SimulationBatchManifest
    summary: Mapping[str, Any]


def _count_by_key(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _quarantine_root(output_root: str | Path, run_id: str, iteration: int) -> Path:
    return Path(output_root) / run_id / "iterations" / f"iter_{iteration:04d}" / FAILED_SIMULATION_QUARANTINE_SUBDIR


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_json(dict(payload)), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _clear_quarantine_dir(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _png_chunk(type_code: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(type_code + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + type_code + data + struct.pack(">I", checksum)


def _build_validation_png(*, failed_count: int) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    width = max(1, min(32, failed_count))
    height = 8
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)

    red = min(255, 64 + failed_count * 8)
    pixel = bytes((red, 32, 32, 255))
    row = b"\x00" + pixel * width
    raw = row * height
    idat = zlib.compress(raw, level=9)
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def write_failed_simulation_quarantine(
    *,
    output_root: str | Path,
    manifest: SimulationBatchManifest | Mapping[str, Any],
) -> FailedSimulationQuarantineArtifacts | None:
    manifest_obj = manifest if isinstance(manifest, SimulationBatchManifest) else SimulationBatchManifest.from_dict(manifest)
    failed_records = tuple(record for record in manifest_obj.records if record.is_failed)
    quarantine_dir = _quarantine_root(output_root, manifest_obj.run_id, manifest_obj.iteration)
    if not failed_records:
        _clear_quarantine_dir(quarantine_dir)
        return None

    quarantine_manifest = SimulationBatchManifest(
        run_id=manifest_obj.run_id,
        iteration=manifest_obj.iteration,
        records=failed_records,
        metadata={
            **dict(manifest_obj.metadata),
            "quarantine": True,
            "source_manifest_hash": manifest_obj.stable_hash(),
        },
        schema_version=ACTIVE_LEARNING_QUARANTINE_MANIFEST_SCHEMA_VERSION,
    )

    reasons = [record.reason for record in failed_records]
    summary = {
        "schema_version": ACTIVE_LEARNING_QUARANTINE_REPORT_SCHEMA_VERSION,
        "run_id": manifest_obj.run_id,
        "iteration": manifest_obj.iteration,
        "failed_count": len(failed_records),
        "family_distribution": _count_by_key([record.family for record in failed_records]),
        "stage_distribution": _count_by_key([record.stage for record in failed_records]),
        "reason_distribution": _count_by_key(reasons),
        "manifest_hash": quarantine_manifest.stable_hash(),
    }

    manifest_path = quarantine_dir / FAILED_SIMULATION_MANIFEST_FILENAME
    validation_png_path = quarantine_dir / FAILED_SIMULATION_VALIDATION_PNG_FILENAME
    validation_sidecar_path = quarantine_dir / FAILED_SIMULATION_VALIDATION_PNG_SIDECAR_FILENAME

    _write_json(manifest_path, quarantine_manifest.as_dict())
    validation_png_path.write_bytes(_build_validation_png(failed_count=len(failed_records)))
    _write_json(validation_sidecar_path, summary)

    return FailedSimulationQuarantineArtifacts(
        quarantine_dir=quarantine_dir,
        manifest_path=manifest_path,
        validation_png_path=validation_png_path,
        validation_sidecar_path=validation_sidecar_path,
        failed_manifest=quarantine_manifest,
        summary=summary,
    )


__all__ = [
    "ACTIVE_LEARNING_QUARANTINE_MANIFEST_SCHEMA_VERSION",
    "ACTIVE_LEARNING_QUARANTINE_REPORT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SIMULATION_MANIFEST_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SIMULATION_RECORD_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SUBMISSION_BOUNDARY_SCHEMA_VERSION",
    "FAILED_SIMULATION_MANIFEST_FILENAME",
    "FAILED_SIMULATION_QUARANTINE_SUBDIR",
    "FAILED_SIMULATION_VALIDATION_PNG_FILENAME",
    "FAILED_SIMULATION_VALIDATION_PNG_SIDECAR_FILENAME",
    "FailedSimulationQuarantineArtifacts",
    "SimulationBatchManifest",
    "SimulationManifestRecord",
    "SubmissionBoundaryState",
    "write_failed_simulation_quarantine",
]
