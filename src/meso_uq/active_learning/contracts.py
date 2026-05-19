from __future__ import annotations

"""Contracts for active-learning loop state and contracts."""

import json
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.core import Platform, coerce_platform


ACTIVE_LEARNING_CANDIDATE_SCHEMA_VERSION = "meso_uq.active_learning.candidate.v1"
ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION = "meso_uq.active_learning.acquisition_score.v1"
ACTIVE_LEARNING_SIMULATION_REQUEST_SCHEMA_VERSION = "meso_uq.active_learning.simulation_request.v1"
ACTIVE_LEARNING_SIMULATION_RESULT_SCHEMA_VERSION = "meso_uq.active_learning.simulation_result.v1"
ACTIVE_LEARNING_RETRAIN_REQUEST_SCHEMA_VERSION = "meso_uq.active_learning.retrain_request.v1"
ACTIVE_LEARNING_RETRAIN_RESULT_SCHEMA_VERSION = "meso_uq.active_learning.retrain_result.v1"
ACTIVE_LEARNING_LOOP_STATE_SCHEMA_VERSION = "meso_uq.active_learning.loop_state.v1"
ACTIVE_LEARNING_FAILURE_RECORD_SCHEMA_VERSION = "meso_uq.active_learning.failure_record.v1"
ACTIVE_LEARNING_STOPPING_CRITERIA_SCHEMA_VERSION = "meso_uq.active_learning.stopping_criteria.v1"
ACTIVE_LEARNING_BUDGET_CONFIG_SCHEMA_VERSION = "meso_uq.active_learning.budget_config.v1"
ACTIVE_LEARNING_PLATFORM_CONFIG_SCHEMA_VERSION = "meso_uq.active_learning.platform_config.v1"
ACTIVE_LEARNING_RUNTIME_CONFIG_SCHEMA_VERSION = "meso_uq.active_learning.runtime_config.v1"
ACTIVE_LEARNING_OUTPUT_CONFIG_SCHEMA_VERSION = "meso_uq.active_learning.output_config.v1"
ACTIVE_LEARNING_CONFIG_SCHEMA_VERSION = "meso_uq.active_learning.config.v1"
ACTIVE_LEARNING_ITERATION_LINEAGE_SCHEMA_VERSION = "meso_uq.active_learning.iteration_lineage.v1"


def _coerce_walltime(payload: Any) -> str:
    if not isinstance(payload, str):
        raise ValueError("walltime must be a string.")
    text = payload.strip()
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("walltime must use HH:MM:SS.")
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("walltime must be a string in HH:MM:SS format.") from exc
    if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
        raise ValueError("walltime minutes and seconds must be in range [0, 59].")
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_normalize_json(payload, "payload"), sort_keys=True, separators=(",", ":"), default=_json_default)


def _sha256_short(payload: Mapping[str, Any], *, length: int = 16) -> str:
    if not isinstance(length, int) or length < 4 or length > 64:
        raise ValueError("hash length must be in range [4, 64].")
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def candidate_hash(candidate: "Candidate", *, length: int = 16) -> str:
    return _sha256_short(
        {
            "candidate_id": str(candidate.candidate_id),
            "parameters": dict(candidate.parameters),
            "metadata": dict(candidate.metadata),
        },
        length=length,
    )


@dataclass(frozen=True)
class ActiveLearningBudgetConfig:
    max_iterations: int = 1
    max_evaluations: int | None = None
    max_failures: int | None = None
    schema_version: str = ACTIVE_LEARNING_BUDGET_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if self.max_evaluations is not None:
            if not isinstance(self.max_evaluations, int) or self.max_evaluations < 1:
                raise ValueError("max_evaluations must be a positive integer when specified.")
        if self.max_failures is not None:
            if not isinstance(self.max_failures, int) or self.max_failures < 1:
                raise ValueError("max_failures must be a positive integer when specified.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "max_failures": self.max_failures,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveLearningBudgetConfig":
        return cls(
            max_iterations=int(payload.get("max_iterations", 1)),
            max_evaluations=payload.get("max_evaluations"),
            max_failures=payload.get("max_failures"),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_BUDGET_CONFIG_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class ActiveLearningPlatformConfig:
    platform: Platform | str = Platform.WORKSTATION
    walltime: str = "00:30:00"
    gpu_count: int = 1
    partition: str | None = None
    schema_version: str = ACTIVE_LEARNING_PLATFORM_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", coerce_platform(self.platform))
        object.__setattr__(self, "walltime", _coerce_walltime(self.walltime))
        if not isinstance(self.gpu_count, int) or self.gpu_count < 1:
            raise ValueError("gpu_count must be a positive integer.")
        if self.partition is not None:
            object.__setattr__(self, "partition", str(self.partition).strip() or None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform.value,
            "walltime": self.walltime,
            "gpu_count": self.gpu_count,
            "partition": self.partition,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveLearningPlatformConfig":
        return cls(
            platform=payload.get("platform", Platform.WORKSTATION),
            walltime=payload.get("walltime", "00:30:00"),
            gpu_count=int(payload.get("gpu_count", 1)),
            partition=payload.get("partition"),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_PLATFORM_CONFIG_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class ActiveLearningRuntimeConfig:
    batch_size: int = 1
    schema_version: str = ACTIVE_LEARNING_RUNTIME_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveLearningRuntimeConfig":
        return cls(
            batch_size=int(payload.get("batch_size", 1)),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_RUNTIME_CONFIG_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class ActiveLearningOutputConfig:
    output_root: str = "_runs/active_learning"
    emit_artifacts: bool = True
    emit_plots: bool = True
    schema_version: str = ACTIVE_LEARNING_OUTPUT_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        output_root = str(self.output_root).strip()
        if not output_root:
            raise ValueError("output_root must be a non-empty string.")
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "emit_artifacts", bool(self.emit_artifacts))
        object.__setattr__(self, "emit_plots", bool(self.emit_plots))

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "emit_artifacts": self.emit_artifacts,
            "emit_plots": self.emit_plots,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveLearningOutputConfig":
        return cls(
            output_root=str(payload.get("output_root", "_runs/active_learning")),
            emit_artifacts=bool(payload.get("emit_artifacts", True)),
            emit_plots=bool(payload.get("emit_plots", True)),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_OUTPUT_CONFIG_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class ActiveLearningConfig:
    budget: ActiveLearningBudgetConfig | Mapping[str, Any] = field(default_factory=ActiveLearningBudgetConfig)
    platform: ActiveLearningPlatformConfig | Mapping[str, Any] = field(
        default_factory=ActiveLearningPlatformConfig
    )
    runtime: ActiveLearningRuntimeConfig | Mapping[str, Any] = field(default_factory=ActiveLearningRuntimeConfig)
    output: ActiveLearningOutputConfig | Mapping[str, Any] = field(default_factory=ActiveLearningOutputConfig)
    loop_id: str = "active_learning"
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "loop_id", str(self.loop_id).strip())
        if not self.loop_id:
            raise ValueError("loop_id must be a non-empty string.")
        object.__setattr__(self, "budget", _coerce_budget_config(self.budget))
        object.__setattr__(self, "platform", _coerce_platform_config(self.platform))
        object.__setattr__(self, "runtime", _coerce_runtime_config(self.runtime))
        object.__setattr__(self, "output", _coerce_output_config(self.output))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))
        computed_run_id = str(self.run_id).strip() if self.run_id else ""
        if not computed_run_id:
            computed_run_id = _sha256_short(
                {
                    "loop_id": self.loop_id,
                    "budget": _coerce_as_dict(self.budget),
                    "platform": _coerce_as_dict(self.platform),
                    "runtime": _coerce_as_dict(self.runtime),
                    "output": _coerce_as_dict(self.output),
                },
                length=12,
            )
            computed_run_id = f"{self.loop_id}-{computed_run_id}"
        object.__setattr__(self, "run_id", computed_run_id)

    @property
    def output_root(self) -> str:
        return self.output.output_root

    def to_stopping_criteria(self) -> "StoppingCriteria":
        return StoppingCriteria(
            max_iterations=self.budget.max_iterations,
            max_evaluations=self.budget.max_evaluations,
            max_failures=self.budget.max_failures,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "run_id": self.run_id,
            "budget": _coerce_as_dict(self.budget),
            "platform": _coerce_as_dict(self.platform),
            "runtime": _coerce_as_dict(self.runtime),
            "output": _coerce_as_dict(self.output),
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveLearningConfig":
        return cls(
            budget=ActiveLearningBudgetConfig.from_dict(payload.get("budget", {})),
            platform=ActiveLearningPlatformConfig.from_dict(payload.get("platform", {})),
            runtime=ActiveLearningRuntimeConfig.from_dict(payload.get("runtime", {})),
            output=ActiveLearningOutputConfig.from_dict(payload.get("output", {})),
            loop_id=str(payload.get("loop_id", "active_learning")),
            run_id=payload.get("run_id"),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_CONFIG_SCHEMA_VERSION)),
        )

    def to_json(self, **kwargs: Any) -> str:
        params = {"sort_keys": True}
        params.update(kwargs)
        return json.dumps(self.as_dict(), **params)

    @classmethod
    def from_json(cls, payload: str) -> "ActiveLearningConfig":
        return cls.from_dict(json.loads(payload))


def _metadata_dict(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("metadata must be a mapping.")
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized[str(key)] = _normalize_json(value, f"metadata[{key!r}]")
    return normalized


def _normalize_json(value: Any, context: str) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_normalize_json(item, f"{context}[]") for item in value]
    if isinstance(value, list):
        return [_normalize_json(item, f"{context}[]") for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item, f"{context}[%r]" % key) for key, item in value.items()}
    raise ValueError(f"{context} must be JSON-compatible, got {type(value)!r}.")


def _normalize_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return _metadata_dict(payload)


def _coerce_as_dict(value: ActiveLearningBudgetConfig | ActiveLearningPlatformConfig | ActiveLearningRuntimeConfig | ActiveLearningOutputConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, ActiveLearningBudgetConfig):
        return value.as_dict()
    if isinstance(value, ActiveLearningPlatformConfig):
        return value.as_dict()
    if isinstance(value, ActiveLearningRuntimeConfig):
        return value.as_dict()
    if isinstance(value, ActiveLearningOutputConfig):
        return value.as_dict()
    raise ValueError("Expected a known active-learning configuration object or mapping.")


def _coerce_budget_config(payload: ActiveLearningBudgetConfig | Mapping[str, Any]) -> ActiveLearningBudgetConfig:
    if isinstance(payload, ActiveLearningBudgetConfig):
        return payload
    if isinstance(payload, Mapping):
        return ActiveLearningBudgetConfig.from_dict(payload)
    raise ValueError("budget must be an ActiveLearningBudgetConfig or mapping.")


def _coerce_platform_config(payload: ActiveLearningPlatformConfig | Mapping[str, Any]) -> ActiveLearningPlatformConfig:
    if isinstance(payload, ActiveLearningPlatformConfig):
        return payload
    if isinstance(payload, Mapping):
        return ActiveLearningPlatformConfig.from_dict(payload)
    raise ValueError("platform must be an ActiveLearningPlatformConfig or mapping.")


def _coerce_runtime_config(payload: ActiveLearningRuntimeConfig | Mapping[str, Any]) -> ActiveLearningRuntimeConfig:
    if isinstance(payload, ActiveLearningRuntimeConfig):
        return payload
    if isinstance(payload, Mapping):
        return ActiveLearningRuntimeConfig.from_dict(payload)
    raise ValueError("runtime must be an ActiveLearningRuntimeConfig or mapping.")


def _coerce_output_config(payload: ActiveLearningOutputConfig | Mapping[str, Any]) -> ActiveLearningOutputConfig:
    if isinstance(payload, ActiveLearningOutputConfig):
        return payload
    if isinstance(payload, Mapping):
        return ActiveLearningOutputConfig.from_dict(payload)
    raise ValueError("output must be an ActiveLearningOutputConfig or mapping.")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        if not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string.")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping.")
        if not self.parameters:
            raise ValueError("parameters must contain at least one value.")
        object.__setattr__(self, "parameters", _normalize_json(dict(self.parameters), "parameters"))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parameters": dict(self.parameters),
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    def stable_hash(self, *, length: int = 16) -> str:
        return candidate_hash(self, length=length)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Candidate":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            parameters=payload["parameters"],
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_CANDIDATE_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class AcquisitionScore:
    candidate_id: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        if not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string.")
        if not isinstance(self.score, int | float):
            raise ValueError("score must be numeric.")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be a finite number.")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AcquisitionScore":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            score=float(payload["score"]),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class SimulationRequest:
    request_id: str
    candidate: Candidate
    iteration: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_SIMULATION_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", str(self.request_id))
        if not self.request_id:
            raise ValueError("request_id must be a non-empty string.")
        if not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
        if not isinstance(self.candidate, Candidate):
            object.__setattr__(self, "candidate", Candidate.from_dict(self.candidate))  # type: ignore[arg-type]
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate": self.candidate.as_dict(),
            "iteration": self.iteration,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationRequest":
        return cls(
            request_id=str(payload["request_id"]),
            candidate=Candidate.from_dict(payload["candidate"]),
            iteration=int(payload["iteration"]),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_SIMULATION_REQUEST_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class SimulationResult:
    request_id: str
    candidate_id: str
    status: str
    metrics: Mapping[str, Any]
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_SIMULATION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", str(self.request_id))
        if not self.request_id:
            raise ValueError("request_id must be a non-empty string.")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        if not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string.")
        if self.status not in {"success", "failed"}:
            raise ValueError("status must be 'success' or 'failed'.")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("metrics must be a mapping.")
        object.__setattr__(self, "metrics", _normalize_json(dict(self.metrics), "metrics"))
        if self.error is not None:
            object.__setattr__(self, "error", str(self.error))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "metrics": self.metrics,
            "error": self.error,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationResult":
        return cls(
            request_id=str(payload["request_id"]),
            candidate_id=str(payload["candidate_id"]),
            status=str(payload["status"]),
            metrics=payload["metrics"],
            error=payload.get("error"),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_SIMULATION_RESULT_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class RetrainingRequest:
    iteration: int
    successful_results: tuple[SimulationResult, ...]
    candidate_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_RETRAIN_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
        object.__setattr__(
            self,
            "successful_results",
            tuple(
                result if isinstance(result, SimulationResult) else SimulationResult.from_dict(result)
                for result in self.successful_results
            ),
        )
        object.__setattr__(self, "candidate_ids", tuple(str(item) for item in self.candidate_ids))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "successful_results": [item.as_dict() for item in self.successful_results],
            "candidate_ids": list(self.candidate_ids),
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetrainingRequest":
        return cls(
            iteration=int(payload["iteration"]),
            successful_results=tuple(
                SimulationResult.from_dict(item) for item in payload.get("successful_results", ())
            ),
            candidate_ids=tuple(payload.get("candidate_ids", ())),
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_RETRAIN_REQUEST_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class RetrainingResult:
    iteration: int
    status: str
    metrics: Mapping[str, Any]
    model_token: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_RETRAIN_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
        object.__setattr__(self, "status", str(self.status))
        if self.status not in {"completed", "failed", "skipped"}:
            raise ValueError("status must be one of: completed, failed, skipped.")
        object.__setattr__(self, "model_token", None if self.model_token is None else str(self.model_token))
        if self.status in {"completed", "failed"} and not self.model_token:
            raise ValueError("model_token must be a non-empty string for completed/failed retraining.")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("metrics must be a mapping.")
        object.__setattr__(self, "metrics", _normalize_json(dict(self.metrics), "metrics"))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "status": self.status,
            "model_token": self.model_token,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetrainingResult":
        return cls(
            iteration=int(payload["iteration"]),
            status=str(payload["status"]),
            model_token=payload.get("model_token"),
            metrics=payload["metrics"],
            metadata=payload.get("metadata", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_RETRAIN_RESULT_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class FailureRecord:
    stage: str
    message: str
    iteration: int
    candidate_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ACTIVE_LEARNING_FAILURE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", str(self.stage))
        if self.stage not in {"candidate_generation", "validation", "acquisition", "simulation", "retraining"}:
            raise ValueError("stage must be one of: candidate_generation, validation, acquisition, simulation, retraining.")
        object.__setattr__(self, "message", str(self.message))
        if not self.message:
            raise ValueError("message must be a non-empty string.")
        if not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
        if self.candidate_id is not None:
            object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "details", _normalize_metadata(self.details))

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "iteration": self.iteration,
            "candidate_id": self.candidate_id,
            "details": self.details,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureRecord":
        return cls(
            stage=str(payload["stage"]),
            message=str(payload["message"]),
            iteration=int(payload["iteration"]),
            candidate_id=payload.get("candidate_id"),
            details=payload.get("details", {}),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_FAILURE_RECORD_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class StoppingCriteria:
    max_iterations: int = 1
    max_evaluations: int | None = None
    max_failures: int | None = None
    schema_version: str = ACTIVE_LEARNING_STOPPING_CRITERIA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if self.max_evaluations is not None:
            if not isinstance(self.max_evaluations, int) or self.max_evaluations < 1:
                raise ValueError("max_evaluations must be a positive integer when specified.")
        if self.max_failures is not None:
            if not isinstance(self.max_failures, int) or self.max_failures < 1:
                raise ValueError("max_failures must be a positive integer when specified.")

    def should_stop(self, state: "LoopState") -> str | None:
        if state.iteration >= self.max_iterations:
            return "max_iterations_reached"
        if self.max_evaluations is not None and state.evaluated_candidate_count >= self.max_evaluations:
            return "max_evaluations_reached"
        if self.max_failures is not None and state.failure_count >= self.max_failures:
            return "max_failures_reached"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "max_failures": self.max_failures,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StoppingCriteria":
        return cls(
            max_iterations=int(payload["max_iterations"]),
            max_evaluations=payload.get("max_evaluations"),
            max_failures=payload.get("max_failures"),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_STOPPING_CRITERIA_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class ActiveLearningIterationLineage:
    iteration: int
    iteration_id: str
    run_id: str
    selected_candidate_hashes: tuple[str, ...]
    request_count: int
    success_count: int
    failure_count: int
    parent_iteration_id: str | None = None
    completion_reason: str | None = None
    schema_version: str = ACTIVE_LEARNING_ITERATION_LINEAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
        object.__setattr__(self, "iteration_id", str(self.iteration_id))
        if not self.iteration_id:
            raise ValueError("iteration_id must be a non-empty string.")
        object.__setattr__(self, "run_id", str(self.run_id))
        if not self.run_id:
            raise ValueError("run_id must be a non-empty string.")
        if self.parent_iteration_id is not None:
            object.__setattr__(self, "parent_iteration_id", str(self.parent_iteration_id))
        object.__setattr__(self, "selected_candidate_hashes", tuple(str(item) for item in self.selected_candidate_hashes))
        if not isinstance(self.request_count, int) or self.request_count < 0:
            raise ValueError("request_count must be a non-negative integer.")
        if not isinstance(self.success_count, int) or self.success_count < 0:
            raise ValueError("success_count must be a non-negative integer.")
        if not isinstance(self.failure_count, int) or self.failure_count < 0:
            raise ValueError("failure_count must be a non-negative integer.")

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "iteration_id": self.iteration_id,
            "run_id": self.run_id,
            "selected_candidate_hashes": list(self.selected_candidate_hashes),
            "request_count": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "parent_iteration_id": self.parent_iteration_id,
            "completion_reason": self.completion_reason,
            "schema_version": self.schema_version,
        }

    def as_dict(self) -> dict[str, Any]:
        return self.payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveLearningIterationLineage":
        return cls(
            iteration=int(payload["iteration"]),
            iteration_id=str(payload["iteration_id"]),
            run_id=str(payload["run_id"]),
            selected_candidate_hashes=tuple(payload.get("selected_candidate_hashes", ())),
            request_count=int(payload.get("request_count", 0)),
            success_count=int(payload.get("success_count", 0)),
            failure_count=int(payload.get("failure_count", 0)),
            parent_iteration_id=payload.get("parent_iteration_id"),
            completion_reason=payload.get("completion_reason"),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_ITERATION_LINEAGE_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class LoopState:
    loop_id: str
    iteration: int = 0
    run_id: str | None = None
    candidates: tuple[Candidate, ...] = ()
    scored_candidates: tuple[AcquisitionScore, ...] = ()
    selected_candidate_ids: tuple[str, ...] = ()
    simulation_requests: tuple[SimulationRequest, ...] = ()
    simulation_results: tuple[SimulationResult, ...] = ()
    retraining_results: tuple[RetrainingResult, ...] = ()
    failures: tuple[FailureRecord, ...] = ()
    iteration_lineage: tuple[ActiveLearningIterationLineage, ...] = ()
    completion_reason: str | None = None
    schema_version: str = ACTIVE_LEARNING_LOOP_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "loop_id", str(self.loop_id))
        if not self.loop_id:
            raise ValueError("loop_id must be a non-empty string.")
        if not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
        if self.run_id is not None:
            object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(
            self,
            "candidates",
            tuple(item if isinstance(item, Candidate) else Candidate.from_dict(item) for item in self.candidates),  # type: ignore[arg-type]
        )
        object.__setattr__(
            self,
            "scored_candidates",
            tuple(
                item if isinstance(item, AcquisitionScore) else AcquisitionScore.from_dict(item)
                for item in self.scored_candidates
            ),  # type: ignore[arg-type]
        )
        object.__setattr__(self, "selected_candidate_ids", tuple(str(item) for item in self.selected_candidate_ids))
        object.__setattr__(
            self,
            "simulation_requests",
            tuple(
                item if isinstance(item, SimulationRequest) else SimulationRequest.from_dict(item)
                for item in self.simulation_requests
            ),
        )
        object.__setattr__(
            self,
            "simulation_results",
            tuple(
                item if isinstance(item, SimulationResult) else SimulationResult.from_dict(item)
                for item in self.simulation_results
            ),
        )
        object.__setattr__(
            self,
            "retraining_results",
            tuple(
                item if isinstance(item, RetrainingResult) else RetrainingResult.from_dict(item)
                for item in self.retraining_results
            ),
        )
        object.__setattr__(
            self,
            "failures",
            tuple(item if isinstance(item, FailureRecord) else FailureRecord.from_dict(item) for item in self.failures),
        )
        object.__setattr__(
            self,
            "iteration_lineage",
            tuple(
                item if isinstance(item, ActiveLearningIterationLineage) else ActiveLearningIterationLineage.from_dict(item)
                for item in self.iteration_lineage
            ),
        )
        if self.completion_reason is not None:
            object.__setattr__(self, "completion_reason", str(self.completion_reason))

    @property
    def is_complete(self) -> bool:
        return self.completion_reason is not None

    @property
    def latest_iteration_lineage(self) -> ActiveLearningIterationLineage | None:
        return self.iteration_lineage[-1] if self.iteration_lineage else None

    @property
    def evaluated_candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate.candidate_id for item in self.simulation_requests)

    @property
    def evaluated_candidate_count(self) -> int:
        return len(self.evaluated_candidate_ids)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def successful_results(self) -> tuple[SimulationResult, ...]:
        return tuple(result for result in self.simulation_results if result.is_success)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "iteration": self.iteration,
            "run_id": self.run_id,
            "candidates": [item.as_dict() for item in self.candidates],
            "scored_candidates": [item.as_dict() for item in self.scored_candidates],
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "simulation_requests": [item.as_dict() for item in self.simulation_requests],
            "simulation_results": [item.as_dict() for item in self.simulation_results],
            "retraining_results": [item.as_dict() for item in self.retraining_results],
            "failures": [item.as_dict() for item in self.failures],
            "iteration_lineage": [item.as_dict() for item in self.iteration_lineage],
            "completion_reason": self.completion_reason,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoopState":
        return cls(
            loop_id=str(payload["loop_id"]),
            iteration=int(payload.get("iteration", 0)),
            run_id=payload.get("run_id"),
            candidates=tuple(Candidate.from_dict(item) for item in payload.get("candidates", ())),
            scored_candidates=tuple(
                AcquisitionScore.from_dict(item) for item in payload.get("scored_candidates", ())
            ),
            selected_candidate_ids=tuple(payload.get("selected_candidate_ids", ())),
            simulation_requests=tuple(SimulationRequest.from_dict(item) for item in payload.get("simulation_requests", ())),
            simulation_results=tuple(SimulationResult.from_dict(item) for item in payload.get("simulation_results", ())),
            retraining_results=tuple(RetrainingResult.from_dict(item) for item in payload.get("retraining_results", ())),
            failures=tuple(FailureRecord.from_dict(item) for item in payload.get("failures", ())),
            iteration_lineage=tuple(ActiveLearningIterationLineage.from_dict(item) for item in payload.get("iteration_lineage", ())),
            completion_reason=payload.get("completion_reason"),
            schema_version=str(payload.get("schema_version", ACTIVE_LEARNING_LOOP_STATE_SCHEMA_VERSION)),
        )

    def to_json(self, **json_kwargs: Any) -> str:
        params = {"sort_keys": True}
        params.update(json_kwargs)
        return json.dumps(self.as_dict(), **params)

    @classmethod
    def from_json(cls, payload: str) -> "LoopState":
        return cls.from_dict(json.loads(payload))


def as_candidate(payload: Any) -> Candidate:
    return payload if isinstance(payload, Candidate) else Candidate.from_dict(payload)


def as_scores(payload: Sequence[Any]) -> tuple[AcquisitionScore, ...]:
    return tuple(item if isinstance(item, AcquisitionScore) else AcquisitionScore.from_dict(item) for item in payload)


def as_simulation_results(payload: Sequence[Any]) -> tuple[SimulationResult, ...]:
    return tuple(
        item if isinstance(item, SimulationResult) else SimulationResult.from_dict(item) for item in payload
    )


def as_failure(payload: Sequence[Any]) -> tuple[FailureRecord, ...]:
    return tuple(item if isinstance(item, FailureRecord) else FailureRecord.from_dict(item) for item in payload)


def as_retraining_result(payload: Any) -> RetrainingResult:
    return payload if isinstance(payload, RetrainingResult) else RetrainingResult.from_dict(payload)


__all__ = [
    "ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION",
    "ACTIVE_LEARNING_BUDGET_CONFIG_SCHEMA_VERSION",
    "ACTIVE_LEARNING_CANDIDATE_SCHEMA_VERSION",
    "ACTIVE_LEARNING_FAILURE_RECORD_SCHEMA_VERSION",
    "ACTIVE_LEARNING_ITERATION_LINEAGE_SCHEMA_VERSION",
    "ACTIVE_LEARNING_OUTPUT_CONFIG_SCHEMA_VERSION",
    "ACTIVE_LEARNING_LOOP_STATE_SCHEMA_VERSION",
    "ACTIVE_LEARNING_PLATFORM_CONFIG_SCHEMA_VERSION",
    "ACTIVE_LEARNING_RUNTIME_CONFIG_SCHEMA_VERSION",
    "ACTIVE_LEARNING_RETRAIN_REQUEST_SCHEMA_VERSION",
    "ACTIVE_LEARNING_RETRAIN_RESULT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SIMULATION_REQUEST_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SIMULATION_RESULT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_STOPPING_CRITERIA_SCHEMA_VERSION",
    "ACTIVE_LEARNING_CONFIG_SCHEMA_VERSION",
    "ActiveLearningBudgetConfig",
    "ActiveLearningConfig",
    "ActiveLearningIterationLineage",
    "ActiveLearningOutputConfig",
    "ActiveLearningPlatformConfig",
    "ActiveLearningRuntimeConfig",
    "candidate_hash",
    "AcquisitionScore",
    "Candidate",
    "FailureRecord",
    "LoopState",
    "RetrainingRequest",
    "RetrainingResult",
    "SimulationRequest",
    "SimulationResult",
    "StoppingCriteria",
    "as_candidate",
    "as_failure",
    "as_retraining_result",
    "as_scores",
    "as_simulation_results",
]
