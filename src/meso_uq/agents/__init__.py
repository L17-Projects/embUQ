"""Metadata-only agent registry for the package migration spine."""

from .registry import (
    AGENT_REGISTRY,
    classify_agent_modality_support,
    get_agent_definition,
    list_agent_definitions,
    list_agent_modalities,
    missing_dependency_requirements_for_agent,
    resolve_agent_modality,
    resolve_agent_family_identifier,
    runtime_requirements_for_agent,
    supported_modalities_for_agent,
)

__all__ = [
    "AGENT_REGISTRY",
    "classify_agent_modality_support",
    "get_agent_definition",
    "list_agent_definitions",
    "list_agent_modalities",
    "missing_dependency_requirements_for_agent",
    "resolve_agent_modality",
    "resolve_agent_family_identifier",
    "runtime_requirements_for_agent",
    "supported_modalities_for_agent",
]
