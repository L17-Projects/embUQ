from __future__ import annotations

from ..registry import ObservableSpec


EXTENSION_CURVE = ObservableSpec("extension_curve", "Stretching force-extension response.", "curve")
BUCKLING_RESPONSE = ObservableSpec("buckling_response", "Buckling response under compression sweep.", "curve")
TORSION_RESPONSE = ObservableSpec("torsion_response", "Twist response under applied end rotation.", "curve")
EIGENMODE_SPECTRUM = ObservableSpec("eigenmode_spectrum", "Dominant eigenmode or resonance spectrum.", "spectrum")
SHEAR_FLOW_RESPONSE = ObservableSpec("shear_flow_response", "Response under experimental shear-flow forcing.", "curve")

GV_OBSERVABLE_SCHEMAS = (
    {
        "name": EXTENSION_CURVE.name,
        "description": EXTENSION_CURVE.description,
        "units": EXTENSION_CURVE.units,
    },
    {
        "name": BUCKLING_RESPONSE.name,
        "description": BUCKLING_RESPONSE.description,
        "units": BUCKLING_RESPONSE.units,
    },
    {
        "name": TORSION_RESPONSE.name,
        "description": TORSION_RESPONSE.description,
        "units": TORSION_RESPONSE.units,
    },
    {
        "name": EIGENMODE_SPECTRUM.name,
        "description": EIGENMODE_SPECTRUM.description,
        "units": EIGENMODE_SPECTRUM.units,
    },
    {
        "name": SHEAR_FLOW_RESPONSE.name,
        "description": SHEAR_FLOW_RESPONSE.description,
        "units": SHEAR_FLOW_RESPONSE.units,
    },
)
