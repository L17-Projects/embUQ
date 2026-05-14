"""In-memory registry for surrogate-backend contracts."""

from __future__ import annotations

import importlib.util
from typing import Mapping

from meso_uq.core import (
    AgentFamily,
    Modality,
    ModelBackend,
    RuntimeRequirement,
    RuntimeRequirementKind,
    RequirementState,
    coerce_agent_family,
    coerce_modality,
    coerce_model_backend,
)

from .contracts import SurrogateBackend, SurrogateBackendMetadata


class SurrogateDependencyError(RuntimeError):
    """Raised when a backend cannot be resolved because required dependencies are missing."""

    def __init__(self, backend: ModelBackend, missing_requirements: tuple[RuntimeRequirement, ...]) -> None:
        self.backend = backend
        self.missing_requirements = missing_requirements
        names = ", ".join(
            requirement.package or requirement.name for requirement in missing_requirements
        )
        super().__init__(
            f"Backend '{backend.value}' is registered but unavailable: missing required dependencies: {names}."
        )


_BACKEND_REGISTRY: dict[ModelBackend, SurrogateBackend] = {}


def clear_surrogate_backend_registry() -> None:
    """Clear all registered backends (mainly for tests)."""
    _BACKEND_REGISTRY.clear()


def list_surrogate_backends() -> tuple[ModelBackend, ...]:
    """Return backend identifiers in registration order."""
    return tuple(_BACKEND_REGISTRY.keys())


def register_surrogate_backend(backend: SurrogateBackend) -> None:
    """Register a backend implementation."""
    backend_id = backend.metadata.backend
    if backend_id in _BACKEND_REGISTRY:
        raise ValueError(f"Surrogate backend '{backend_id.value}' is already registered.")
    _BACKEND_REGISTRY[backend_id] = backend


def _resolve_available_state(
    requirement: RuntimeRequirement,
    availability: Mapping[str, bool] | None = None,
) -> bool:
    if requirement.kind is not RuntimeRequirementKind.PYTHON_PACKAGE:
        return True

    package = requirement.package or requirement.name
    if availability is not None:
        return bool(availability.get(package, False))

    return importlib.util.find_spec(package) is not None


def _required_unavailable_dependencies(
    backend_metadata: SurrogateBackendMetadata,
    availability: Mapping[str, bool] | None,
) -> tuple[RuntimeRequirement, ...]:
    missing: list[RuntimeRequirement] = []
    for requirement in backend_metadata.runtime_requirements:
        if requirement.state not in {RequirementState.REQUIRED, RequirementState.EXTERNAL}:
            continue
        if not _resolve_available_state(requirement, availability):
            missing.append(requirement)
    return tuple(missing)


def resolve_surrogate_backend(
    backend: ModelBackend | str,
    *,
    agent_family: AgentFamily | str | None = None,
    modality: Modality | str | None = None,
    availability: Mapping[str, bool] | None = None,
) -> SurrogateBackend:
    """Resolve a backend by id and enforce family/modality dependency constraints."""
    backend_id = coerce_model_backend(backend)
    try:
        resolved = _BACKEND_REGISTRY[backend_id]
    except KeyError as exc:
        supported = ", ".join(item.value for item in _BACKEND_REGISTRY)
        raise ValueError(
            f"Unsupported surrogate backend '{backend_id.value}'. Registered: {supported or '<none>'}."
        ) from exc

    if agent_family is not None:
        selected_family = coerce_agent_family(agent_family)
        if selected_family not in resolved.metadata.supported_agent_families:
            supported = ", ".join(item.value for item in resolved.metadata.supported_agent_families)
            raise ValueError(
                f"Backend '{backend_id.value}' does not support agent family '{selected_family.value}'. "
                f"Supported: {supported}."
            )

    if modality is not None:
        selected_modality = coerce_modality(modality)
        if selected_modality not in resolved.metadata.supported_modalities:
            supported = ", ".join(item.value for item in resolved.metadata.supported_modalities)
            raise ValueError(
                f"Backend '{backend_id.value}' does not support modality '{selected_modality.value}'. "
                f"Supported: {supported}."
            )

    missing = _required_unavailable_dependencies(resolved.metadata, availability)
    if missing:
        raise SurrogateDependencyError(backend_id, missing)

    return resolved


__all__ = [
    "SurrogateDependencyError",
    "clear_surrogate_backend_registry",
    "list_surrogate_backends",
    "register_surrogate_backend",
    "resolve_surrogate_backend",
]
