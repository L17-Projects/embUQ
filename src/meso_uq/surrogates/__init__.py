"""Surrogate backend contract layer."""

from .contracts import (
    SurrogateBackend,
    SurrogateBackendMetadata,
    SurrogateCheckpoint,
    SurrogateCheckpointMetadata,
    SurrogateLoadRequest,
    SurrogateModel,
    SurrogatePrediction,
    SurrogatePredictionRequest,
    SurrogateTrainingRequest,
)
from .registry import (
    SurrogateDependencyError,
    clear_surrogate_backend_registry,
    list_surrogate_backends,
    register_surrogate_backend,
    resolve_surrogate_backend,
)

__all__ = [
    "SurrogateBackend",
    "SurrogateBackendMetadata",
    "SurrogateCheckpoint",
    "SurrogateCheckpointMetadata",
    "SurrogateDependencyError",
    "SurrogateLoadRequest",
    "SurrogateModel",
    "SurrogatePrediction",
    "SurrogatePredictionRequest",
    "SurrogateTrainingRequest",
    "clear_surrogate_backend_registry",
    "list_surrogate_backends",
    "register_surrogate_backend",
    "resolve_surrogate_backend",
]
