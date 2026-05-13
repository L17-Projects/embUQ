"""Narrow public API surface for dependency-light MesoUQ package contracts."""

from meso_uq.agents import (
    get_agent_definition,
    list_agent_definitions,
    list_agent_modalities,
    resolve_agent_modality,
    supported_modalities_for_agent,
)
from meso_uq.core import (
    AgentDefinition,
    AgentFamily,
    ArtifactClass,
    ManifestMetadata,
    Modality,
    ModalityDescriptor,
    ModelBackend,
    Platform,
    RunMetadata,
)
from meso_uq.modalities import get_modality_descriptor, list_modality_descriptors

__all__ = [
    "AgentDefinition",
    "AgentFamily",
    "ArtifactClass",
    "ManifestMetadata",
    "Modality",
    "ModalityDescriptor",
    "ModelBackend",
    "Platform",
    "RunMetadata",
    "get_agent_definition",
    "get_modality_descriptor",
    "list_agent_definitions",
    "list_agent_modalities",
    "list_modality_descriptors",
    "resolve_agent_modality",
    "supported_modalities_for_agent",
]
