from __future__ import annotations

from ..registry import ObservableSpec


EXTENSION_CURVE = ObservableSpec("extension_curve", "Stretching force-extension response.", "curve")
BUCKLING_RESPONSE = ObservableSpec("buckling_response", "Buckling response under compression sweep.", "curve")
TORSION_RESPONSE = ObservableSpec("torsion_response", "Twist response under applied end rotation.", "curve")
EIGENMODE_SPECTRUM = ObservableSpec("eigenmode_spectrum", "Dominant eigenmode or resonance spectrum.", "spectrum")
SHEAR_FLOW_RESPONSE = ObservableSpec("shear_flow_response", "Response under experimental shear-flow forcing.", "curve")
