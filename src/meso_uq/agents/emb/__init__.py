"""Metadata boundary for encapsulated microbubble agent contracts."""

from __future__ import annotations

from meso_uq.core import AgentFamily, Modality
from .workflows import (
    EMB_GENERATION_WORKFLOWS,
    EmbGenerationWorkflow,
    EmbParameterFileContract,
    emb_config_candidate_paths,
    get_emb_generation_workflow,
    list_emb_generation_workflows,
    resolve_emb_config_file,
)


FAMILY = AgentFamily.EMB
ALIASES = ("elastic_microbubble", "elastic-microbubble", "microbubble", "uqdpd", "emb")
LEGACY_ROOTS = ("compression", "indentation")
SUPPORTED_MODALITIES = (Modality.COMPRESSION, Modality.INDENTATION)


__all__ = [
    "ALIASES",
    "EMB_GENERATION_WORKFLOWS",
    "EmbGenerationWorkflow",
    "EmbParameterFileContract",
    "FAMILY",
    "LEGACY_ROOTS",
    "SUPPORTED_MODALITIES",
    "emb_config_candidate_paths",
    "get_emb_generation_workflow",
    "list_emb_generation_workflows",
    "resolve_emb_config_file",
]
