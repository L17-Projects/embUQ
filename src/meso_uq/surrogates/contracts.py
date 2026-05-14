"""Backend-neutral surrogate contracts for train/load/predict implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, runtime_checkable, Protocol

from meso_uq.core import (
    AgentFamily,
    Modality,
    ModelBackend,
    RuntimeRequirement,
    coerce_agent_family,
    coerce_modality,
    coerce_model_backend,
)


def _coerce_float_tuple(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _metadata_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items()}


@dataclass(frozen=True)
class SurrogatePrediction:
    """Prediction output exposed by every backend."""

    mean: tuple[float, ...]
    epistemic: tuple[float, ...] | None = None
    aleatoric: tuple[float, ...] | None = None
    surrogate_error: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _coerce_float_tuple(self.mean))
        object.__setattr__(
            self,
            "epistemic",
            _coerce_float_tuple(self.epistemic) if self.epistemic is not None else None,
        )
        object.__setattr__(
            self,
            "aleatoric",
            _coerce_float_tuple(self.aleatoric) if self.aleatoric is not None else None,
        )
        object.__setattr__(
            self,
            "surrogate_error",
            _coerce_float_tuple(self.surrogate_error) if self.surrogate_error is not None else None,
        )

        size = len(self.mean)
        for field_name, field_value in {
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "surrogate_error": self.surrogate_error,
        }.items():
            if field_value is not None and len(field_value) != size:
                raise ValueError(f"{field_name} must match mean dimensionality ({size}).")

    @property
    def is_deterministic(self) -> bool:
        return self.epistemic is None and self.aleatoric is None and self.surrogate_error is None


@dataclass(frozen=True)
class SurrogatePredictionRequest:
    """Input contract passed into predict-time inference."""

    features: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _coerce_float_tuple(self.features))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


@dataclass(frozen=True)
class SurrogateTrainingRequest:
    """Input contract for building a backend checkpoint."""

    backend: ModelBackend
    agent_family: AgentFamily
    modality: Modality
    training_data_uri: str
    checkpoint_uri: str
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", coerce_model_backend(self.backend))
        object.__setattr__(self, "agent_family", coerce_agent_family(self.agent_family))
        object.__setattr__(self, "modality", coerce_modality(self.modality))
        object.__setattr__(self, "training_data_uri", str(self.training_data_uri))
        object.__setattr__(self, "checkpoint_uri", str(self.checkpoint_uri))
        object.__setattr__(self, "hyperparameters", _metadata_dict(self.hyperparameters))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


@dataclass(frozen=True)
class SurrogateLoadRequest:
    """Input contract for loading a trained backend artifact."""

    backend: ModelBackend
    agent_family: AgentFamily
    modality: Modality
    checkpoint_uri: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", coerce_model_backend(self.backend))
        object.__setattr__(self, "agent_family", coerce_agent_family(self.agent_family))
        object.__setattr__(self, "modality", coerce_modality(self.modality))
        object.__setattr__(self, "checkpoint_uri", str(self.checkpoint_uri))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


@dataclass(frozen=True)
class SurrogateBackendMetadata:
    """Declarative metadata required by the backend registry."""

    backend: ModelBackend
    label: str
    supported_agent_families: tuple[AgentFamily, ...]
    supported_modalities: tuple[Modality, ...]
    runtime_requirements: tuple[RuntimeRequirement, ...] = ()
    supports_epistemic: bool = False
    supports_aleatoric: bool = False
    supports_surrogate_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", coerce_model_backend(self.backend))
        object.__setattr__(
            self,
            "supported_agent_families",
            tuple(coerce_agent_family(item) for item in self.supported_agent_families),
        )
        object.__setattr__(
            self,
            "supported_modalities",
            tuple(coerce_modality(item) for item in self.supported_modalities),
        )
        object.__setattr__(
            self,
            "runtime_requirements",
            tuple(
                req if isinstance(req, RuntimeRequirement) else RuntimeRequirement.from_dict(req)
                for req in self.runtime_requirements
            ),
        )
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def supports(self, *, agent_family: AgentFamily | str, modality: Modality | str) -> bool:
        selected_family = coerce_agent_family(agent_family)
        selected_modality = coerce_modality(modality)
        return (
            selected_family in self.supported_agent_families
            and selected_modality in self.supported_modalities
        )


@dataclass(frozen=True)
class SurrogateCheckpointMetadata:
    """Checkpoint-level metadata persisted by a backend implementation."""

    checkpoint_id: str
    backend: ModelBackend
    agent_family: AgentFamily
    modality: Modality
    schema_version: str = "meso_uq.surrogate_checkpoint.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", coerce_model_backend(self.backend))
        object.__setattr__(self, "agent_family", coerce_agent_family(self.agent_family))
        object.__setattr__(self, "modality", coerce_modality(self.modality))
        object.__setattr__(self, "checkpoint_id", str(self.checkpoint_id))
        object.__setattr__(self, "schema_version", str(self.schema_version))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))


@dataclass(frozen=True)
class SurrogateCheckpoint:
    """Small, portable artifact record returned by a train call."""

    uri: str
    metadata: SurrogateCheckpointMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", str(self.uri))


@runtime_checkable
class SurrogateModel(Protocol):
    """Protocol for loaded inference objects returned by backend `load`."""

    metadata: SurrogateBackendMetadata

    def predict(self, request: SurrogatePredictionRequest) -> SurrogatePrediction:
        ...


@runtime_checkable
class SurrogateBackend(Protocol):
    """Backend contract shared by all future implementations."""

    metadata: SurrogateBackendMetadata

    def train(self, request: SurrogateTrainingRequest) -> SurrogateCheckpoint:
        ...

    def load(self, request: SurrogateLoadRequest) -> SurrogateModel:
        ...


__all__ = [
    "SurrogateBackend",
    "SurrogateBackendMetadata",
    "SurrogateCheckpoint",
    "SurrogateCheckpointMetadata",
    "SurrogateLoadRequest",
    "SurrogateModel",
    "SurrogatePrediction",
    "SurrogatePredictionRequest",
    "SurrogateTrainingRequest",
]
