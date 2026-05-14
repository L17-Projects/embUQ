from __future__ import annotations

"""Contracts for active-learning loop state and contracts."""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


ACTIVE_LEARNING_CANDIDATE_SCHEMA_VERSION = "meso_uq.active_learning.candidate.v1"
ACTIVE_LEARNING_ACQUISITION_SCORE_SCHEMA_VERSION = "meso_uq.active_learning.acquisition_score.v1"
ACTIVE_LEARNING_SIMULATION_REQUEST_SCHEMA_VERSION = "meso_uq.active_learning.simulation_request.v1"
ACTIVE_LEARNING_SIMULATION_RESULT_SCHEMA_VERSION = "meso_uq.active_learning.simulation_result.v1"
ACTIVE_LEARNING_RETRAIN_REQUEST_SCHEMA_VERSION = "meso_uq.active_learning.retrain_request.v1"
ACTIVE_LEARNING_RETRAIN_RESULT_SCHEMA_VERSION = "meso_uq.active_learning.retrain_result.v1"
ACTIVE_LEARNING_LOOP_STATE_SCHEMA_VERSION = "meso_uq.active_learning.loop_state.v1"
ACTIVE_LEARNING_FAILURE_RECORD_SCHEMA_VERSION = "meso_uq.active_learning.failure_record.v1"
ACTIVE_LEARNING_STOPPING_CRITERIA_SCHEMA_VERSION = "meso_uq.active_learning.stopping_criteria.v1"


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
class LoopState:
    loop_id: str
    iteration: int = 0
    candidates: tuple[Candidate, ...] = ()
    scored_candidates: tuple[AcquisitionScore, ...] = ()
    selected_candidate_ids: tuple[str, ...] = ()
    simulation_requests: tuple[SimulationRequest, ...] = ()
    simulation_results: tuple[SimulationResult, ...] = ()
    retraining_results: tuple[RetrainingResult, ...] = ()
    failures: tuple[FailureRecord, ...] = ()
    completion_reason: str | None = None
    schema_version: str = ACTIVE_LEARNING_LOOP_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "loop_id", str(self.loop_id))
        if not self.loop_id:
            raise ValueError("loop_id must be a non-empty string.")
        if not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer.")
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
        if self.completion_reason is not None:
            object.__setattr__(self, "completion_reason", str(self.completion_reason))

    @property
    def is_complete(self) -> bool:
        return self.completion_reason is not None

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
            "candidates": [item.as_dict() for item in self.candidates],
            "scored_candidates": [item.as_dict() for item in self.scored_candidates],
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "simulation_requests": [item.as_dict() for item in self.simulation_requests],
            "simulation_results": [item.as_dict() for item in self.simulation_results],
            "retraining_results": [item.as_dict() for item in self.retraining_results],
            "failures": [item.as_dict() for item in self.failures],
            "completion_reason": self.completion_reason,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LoopState":
        return cls(
            loop_id=str(payload["loop_id"]),
            iteration=int(payload.get("iteration", 0)),
            candidates=tuple(Candidate.from_dict(item) for item in payload.get("candidates", ())),
            scored_candidates=tuple(
                AcquisitionScore.from_dict(item) for item in payload.get("scored_candidates", ())
            ),
            selected_candidate_ids=tuple(payload.get("selected_candidate_ids", ())),
            simulation_requests=tuple(SimulationRequest.from_dict(item) for item in payload.get("simulation_requests", ())),
            simulation_results=tuple(SimulationResult.from_dict(item) for item in payload.get("simulation_results", ())),
            retraining_results=tuple(RetrainingResult.from_dict(item) for item in payload.get("retraining_results", ())),
            failures=tuple(FailureRecord.from_dict(item) for item in payload.get("failures", ())),
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
    "ACTIVE_LEARNING_CANDIDATE_SCHEMA_VERSION",
    "ACTIVE_LEARNING_FAILURE_RECORD_SCHEMA_VERSION",
    "ACTIVE_LEARNING_LOOP_STATE_SCHEMA_VERSION",
    "ACTIVE_LEARNING_RETRAIN_REQUEST_SCHEMA_VERSION",
    "ACTIVE_LEARNING_RETRAIN_RESULT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SIMULATION_REQUEST_SCHEMA_VERSION",
    "ACTIVE_LEARNING_SIMULATION_RESULT_SCHEMA_VERSION",
    "ACTIVE_LEARNING_STOPPING_CRITERIA_SCHEMA_VERSION",
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
