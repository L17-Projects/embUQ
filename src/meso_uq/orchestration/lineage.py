from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.core import ArtifactClass, Platform, coerce_artifact_class, coerce_platform


class RunStageStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REUSED = "reused"
    SKIPPED = "skipped"


def _coerce_run_stage_status(value: RunStageStatus | str) -> RunStageStatus:
    if isinstance(value, RunStageStatus):
        return value
    try:
        return RunStageStatus(str(value))
    except ValueError as exc:
        expected = ", ".join(item.value for item in RunStageStatus)
        raise ValueError(f"Unsupported run stage status '{value}'. Expected one of: {expected}.") from exc


def _metadata_dict(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): value for key, value in (metadata or {}).items()}


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def config_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_run_id(prefix: str, payload: Mapping[str, Any], *, length: int = 12) -> str:
    digest = config_digest(payload)[:length]
    return f"{prefix.strip().replace(' ', '-')}-{digest}"


@dataclass(frozen=True)
class LineageArtifact:
    artifact_id: str
    artifact_class: ArtifactClass
    path: str
    role: str
    checksum: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_class", coerce_artifact_class(self.artifact_class))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if Path(self.path).is_absolute():
            raise ValueError("Lineage artifact paths must be relative or placeholder-based.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_class": self.artifact_class.value,
            "path": self.path,
            "role": self.role,
            "checksum": self.checksum,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LineageArtifact":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_class=payload["artifact_class"],
            path=str(payload["path"]),
            role=str(payload["role"]),
            checksum=payload.get("checksum"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class RunStageRecord:
    stage_id: str
    status: RunStageStatus
    command: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    reused_from: str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_run_stage_status(self.status))
        object.__setattr__(self, "command", tuple(str(item) for item in self.command))
        object.__setattr__(self, "inputs", tuple(str(item) for item in self.inputs))
        object.__setattr__(self, "outputs", tuple(str(item) for item in self.outputs))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if self.status is RunStageStatus.FAILED and not self.error:
            raise ValueError(f"Failed stage '{self.stage_id}' must record an error.")
        if self.status is RunStageStatus.REUSED and not self.reused_from:
            raise ValueError(f"Reused stage '{self.stage_id}' must record reused_from.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "status": self.status.value,
            "command": list(self.command),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "reused_from": self.reused_from,
            "error": self.error,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunStageRecord":
        return cls(
            stage_id=str(payload["stage_id"]),
            status=payload["status"],
            command=tuple(payload.get("command", ())),
            inputs=tuple(payload.get("inputs", ())),
            outputs=tuple(payload.get("outputs", ())),
            reused_from=payload.get("reused_from"),
            error=payload.get("error"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class LineageValidationEvidence:
    validation_id: str
    matrix_report_path: str
    status: str
    platform: Platform
    slurm_job_id: str | None = None
    hard_failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", coerce_platform(self.platform))
        object.__setattr__(self, "hard_failures", tuple(str(item) for item in self.hard_failures))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if Path(self.matrix_report_path).is_absolute() and "${" not in self.matrix_report_path:
            # Operational paths are acceptable in evidence, but callers should
            # make the storage policy explicit through metadata.
            object.__setattr__(
                self,
                "metadata",
                {**_metadata_dict(self.metadata), "path_policy": self.metadata.get("path_policy", "external")},
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "matrix_report_path": self.matrix_report_path,
            "status": self.status,
            "platform": self.platform.value,
            "slurm_job_id": self.slurm_job_id,
            "hard_failures": list(self.hard_failures),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LineageValidationEvidence":
        return cls(
            validation_id=str(payload["validation_id"]),
            matrix_report_path=str(payload["matrix_report_path"]),
            status=str(payload["status"]),
            platform=payload["platform"],
            slurm_job_id=payload.get("slurm_job_id"),
            hard_failures=tuple(payload.get("hard_failures", ())),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class RunLineageRecord:
    run_id: str
    config_digest: str
    platform: Platform
    code_version: str
    stages: tuple[RunStageRecord, ...] = ()
    artifacts: tuple[LineageArtifact, ...] = ()
    validations: tuple[LineageValidationEvidence, ...] = ()
    parent_run_ids: tuple[str, ...] = ()
    schema_version: str = "meso_uq.run_lineage.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", coerce_platform(self.platform))
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "validations", tuple(self.validations))
        object.__setattr__(self, "parent_run_ids", tuple(str(item) for item in self.parent_run_ids))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def with_stage(self, stage: RunStageRecord) -> "RunLineageRecord":
        replaced = [item for item in self.stages if item.stage_id != stage.stage_id]
        return replace(self, stages=tuple((*replaced, stage)))

    def with_artifact(self, artifact: LineageArtifact) -> "RunLineageRecord":
        replaced = [item for item in self.artifacts if item.artifact_id != artifact.artifact_id]
        return replace(self, artifacts=tuple((*replaced, artifact)))

    def with_validation(self, evidence: LineageValidationEvidence) -> "RunLineageRecord":
        replaced = [item for item in self.validations if item.validation_id != evidence.validation_id]
        return replace(self, validations=tuple((*replaced, evidence)))

    def failed_stages(self) -> tuple[RunStageRecord, ...]:
        return tuple(stage for stage in self.stages if stage.status is RunStageStatus.FAILED)

    def resumable_stage_ids(self) -> tuple[str, ...]:
        return tuple(
            stage.stage_id
            for stage in self.stages
            if stage.status in {RunStageStatus.FAILED, RunStageStatus.PLANNED, RunStageStatus.SKIPPED}
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "config_digest": self.config_digest,
            "platform": self.platform.value,
            "code_version": self.code_version,
            "stages": [stage.as_dict() for stage in self.stages],
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "validations": [evidence.as_dict() for evidence in self.validations],
            "parent_run_ids": list(self.parent_run_ids),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunLineageRecord":
        return cls(
            run_id=str(payload["run_id"]),
            config_digest=str(payload["config_digest"]),
            platform=payload["platform"],
            code_version=str(payload["code_version"]),
            stages=tuple(RunStageRecord.from_dict(item) for item in payload.get("stages", ())),
            artifacts=tuple(LineageArtifact.from_dict(item) for item in payload.get("artifacts", ())),
            validations=tuple(
                LineageValidationEvidence.from_dict(item) for item in payload.get("validations", ())
            ),
            parent_run_ids=tuple(payload.get("parent_run_ids", ())),
            schema_version=str(payload.get("schema_version", "meso_uq.run_lineage.v1")),
            metadata=payload.get("metadata", {}),
        )


def new_run_lineage(
    *,
    prefix: str,
    config: Mapping[str, Any],
    platform: Platform | str,
    code_version: str,
    parent_run_ids: Sequence[str] = (),
) -> RunLineageRecord:
    digest = config_digest(config)
    return RunLineageRecord(
        run_id=stable_run_id(prefix, config),
        config_digest=digest,
        platform=platform,
        code_version=code_version,
        parent_run_ids=tuple(parent_run_ids),
    )
