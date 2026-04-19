"""Training baseline validation for MAP Mirheo evaluation scripts.

These constants and the validator are shared between compression and indentation
MAP evaluation scripts. Centralising them here avoids duplication and makes
them testable without Mirheo or korali installed.
"""
from __future__ import annotations

import numpy as np

EXPECTED_TRAINING_FSCALE: float = 0.0074
EXPECTED_TRAINING_SHELL_TH: float = 5.0e-9
EXPECTED_TRAINING_NUMSTEPS: int = 5000
EXPECTED_TRAINING_NUMSTEPS_EQ: int = 10000


def validate_training_baseline(params: dict, source_label: str) -> None:
    """Raise ValueError if params deviate from the validated training baseline.

    Parameters
    ----------
    params:
        Dict containing at minimum: fscale, shell_th, numsteps, numsteps_eq.
    source_label:
        Human-readable label for error messages (e.g. the YAML file path).
    """
    fscale = float(params.get("fscale", float("nan")))
    shell_th = float(params.get("shell_th", float("nan")))
    numsteps = int(float(params.get("numsteps", float("nan"))))
    numsteps_eq = int(float(params.get("numsteps_eq", float("nan"))))

    if not np.isclose(fscale, EXPECTED_TRAINING_FSCALE, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{source_label}: expected fscale={EXPECTED_TRAINING_FSCALE}, got {fscale}"
        )
    if not np.isclose(shell_th, EXPECTED_TRAINING_SHELL_TH, rtol=0.0, atol=1e-15):
        raise ValueError(
            f"{source_label}: expected shell_th={EXPECTED_TRAINING_SHELL_TH}, got {shell_th}"
        )
    if numsteps != EXPECTED_TRAINING_NUMSTEPS:
        raise ValueError(
            f"{source_label}: expected numsteps={EXPECTED_TRAINING_NUMSTEPS}, got {numsteps}"
        )
    if numsteps_eq != EXPECTED_TRAINING_NUMSTEPS_EQ:
        raise ValueError(
            f"{source_label}: expected numsteps_eq={EXPECTED_TRAINING_NUMSTEPS_EQ}, "
            f"got {numsteps_eq}"
        )
