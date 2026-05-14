"""Shared utilities for paper figure generation.

Provides physical unit conversion (DPD → nm / nN) and matplotlib
configuration used by the postprocess figure scripts.  This module has
no Mirheo or simulation dependencies and is fully testable in CI.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Colour palette (consistent across all paper figures)
# ---------------------------------------------------------------------------

COLORS: dict[str, dict[str, str]] = {
    "indentation": {"3.2": "#c07b00", "3.4": "#e09a00", "5.8": "#f5c842"},
    "compression": {"2.1": "#1a5a8a", "2.9": "#2a7fc0", "3.0": "#5aaee0"},
}
LINESTYLES = ["solid", "dashed", "dashdot"]


# ---------------------------------------------------------------------------
# Physical unit conversion
# ---------------------------------------------------------------------------

def load_scaling(emb_yaml: Path) -> Tuple[float, float]:
    """Return (length_factor, force_factor) from a parameters-default.emb.yaml file.

    length_factor: multiply DPD length units → nm
    force_factor:  multiply DPD force units  → nN
    """
    with open(emb_yaml, "rb") as f:
        p = yaml.load(f, Loader=yaml.CLoader)

    ul = float(p["ul"])
    rho_water = float(p["rho_water"])
    rhow = float(p["rhow"])
    energy_factor = float(p["energyFactor"])
    kbol = float(p["kbol"])
    t0 = float(p["t0"])
    fscale = float(p.get("fscale", 1.0))

    ue = energy_factor * kbol * t0
    um = rho_water * ul ** 3 / rhow
    ut = math.sqrt(um * ul ** 2 / ue)

    length_factor = ul * 1e9                          # DPD length → nm
    force_factor = (um * ul / ut ** 2) / fscale * 1e9  # DPD force  → nN
    return length_factor, force_factor


def convert_to_physical(
    modality: str,
    x_dpd: np.ndarray,
    y_dpd: np.ndarray,
    length_factor: float,
    force_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (x_dpd, y_dpd) to physical units based on modality axis convention.

    Compression: x = displacement [DPD→nm], y = force [DPD→nN]
    Indentation: x = force [DPD→nN],        y = displacement [DPD→nm]
    """
    if modality == "compression":
        return x_dpd * length_factor, y_dpd * force_factor
    if modality == "indentation":
        return x_dpd * force_factor, y_dpd * length_factor
    raise ValueError(f"Unknown modality: {modality!r}. Expected 'compression' or 'indentation'.")


def emb_yaml_path(modality: str, repo_root: Path) -> Path:
    """Return the parameters-default.emb.yaml path for a given modality."""
    if modality not in ("compression", "indentation"):
        raise ValueError(f"Unknown modality: {modality!r}")
    return repo_root / "emb" / modality / "src" / "parameters-default.emb.yaml"


# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------

def configure_matplotlib() -> None:
    """Apply paper-standard matplotlib rcParams (no LaTeX — safe for CI)."""
    import matplotlib.pyplot as plt  # imported lazily to avoid CI issues

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
    })
