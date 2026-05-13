"""Metadata-only agent registry for the package migration spine."""

from .registry import (
    AGENT_REGISTRY,
    get_agent_definition,
    list_agent_definitions,
    list_agent_modalities,
    resolve_agent_modality,
    supported_modalities_for_agent,
)

__all__ = [
    "AGENT_REGISTRY",
    "get_agent_definition",
    "list_agent_definitions",
    "list_agent_modalities",
    "resolve_agent_modality",
    "supported_modalities_for_agent",
]
