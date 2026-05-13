"""Dependency-light contracts for the package migration spine."""

from .contracts import (
    AgentDefinition,
    AgentFamily,
    ArtifactClass,
    ManifestMetadata,
    Modality,
    ModalityDescriptor,
    ModelBackend,
    Platform,
    RunMetadata,
    coerce_agent_family,
    coerce_artifact_class,
    coerce_modality,
    coerce_model_backend,
    coerce_platform,
    enum_values,
)

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
    "coerce_agent_family",
    "coerce_artifact_class",
    "coerce_modality",
    "coerce_model_backend",
    "coerce_platform",
    "enum_values",
]
