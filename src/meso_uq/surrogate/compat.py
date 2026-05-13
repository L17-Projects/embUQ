"""Compatibility contracts for serialized surrogate artifacts.

This module is intentionally dependency-light.  It records pickle-era module
aliases without importing Torch-backed model implementations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class SerializedSurrogateAlias:
    """A legacy serialized class path and its canonical replacement."""

    legacy_module: str
    legacy_attribute: str
    replacement_module: str
    replacement_attribute: str
    artifact_format: str
    reason: str
    retirement: str


SURROGATE_SERIALIZATION_ALIASES: tuple[SerializedSurrogateAlias, ...] = (
    SerializedSurrogateAlias(
        legacy_module="learning.model",
        legacy_attribute="MLP",
        replacement_module="meso_uq.surrogate.model",
        replacement_attribute="MLP",
        artifact_format="pickle",
        reason="Older deterministic EMB surrogate pickles can reference learning.model.MLP.",
        retirement="Phase 6, after all release-critical pickles are migrated or regenerated.",
    ),
)


def list_serialized_surrogate_aliases() -> tuple[SerializedSurrogateAlias, ...]:
    """Return the immutable compatibility manifest for surrogate artifacts."""

    return SURROGATE_SERIALIZATION_ALIASES


def install_legacy_surrogate_pickle_aliases(
    *,
    surrogate_package: ModuleType | None,
    model_module: ModuleType,
) -> None:
    """Install narrow module aliases needed by legacy surrogate pickles."""

    if surrogate_package is not None:
        sys.modules.setdefault("learning", surrogate_package)
    sys.modules.setdefault("learning.model", model_module)


__all__ = [
    "SURROGATE_SERIALIZATION_ALIASES",
    "SerializedSurrogateAlias",
    "install_legacy_surrogate_pickle_aliases",
    "list_serialized_surrogate_aliases",
]
