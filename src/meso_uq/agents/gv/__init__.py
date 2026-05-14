"""Metadata boundary for gas-vesicle agent contracts."""

from __future__ import annotations

from meso_uq.core import AgentFamily, Modality


FAMILY = AgentFamily.GV
ALIASES = ("gas_vesicle", "gas-vesicle", "vesicle", "gv")
LEGACY_ROOTS = ("gv/stretching", "gv/buckling", "gv/torsion", "gv/eigenmodes", "gv/shear_flow")
SUPPORTED_MODALITIES = (
    Modality.STRETCHING,
    Modality.BUCKLING,
    Modality.TORSION,
    Modality.EIGENMODES,
    Modality.SHEAR_FLOW,
)


__all__ = [
    "ALIASES",
    "FAMILY",
    "LEGACY_ROOTS",
    "SUPPORTED_MODALITIES",
]
