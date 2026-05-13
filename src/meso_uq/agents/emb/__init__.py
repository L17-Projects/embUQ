"""Metadata boundary for encapsulated microbubble agent contracts."""

from __future__ import annotations

from meso_uq.core import AgentFamily, Modality


FAMILY = AgentFamily.EMB
ALIASES = ("elastic_microbubble", "elastic-microbubble", "microbubble", "uqdpd", "emb")
LEGACY_ROOTS = ("compression", "indentation")
SUPPORTED_MODALITIES = (Modality.COMPRESSION, Modality.INDENTATION)


__all__ = [
    "ALIASES",
    "FAMILY",
    "LEGACY_ROOTS",
    "SUPPORTED_MODALITIES",
]
