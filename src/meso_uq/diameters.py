"""
Diameter Registry for hierarchical UQ workflows.

This module provides a centralized registry of diameters and their
associated configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DiameterConfig:
    value_um: float
    data_file: str
    surrogate_dir: str
    color: str
    label: str
    marker: str = "o"


DIAMETER_REGISTRY: Dict[float, DiameterConfig] = {
    2.1: DiameterConfig(
        value_um=2.1,
        data_file="data_1.csv",
        surrogate_dir="2.1um",
        color="#1f77b4",
        label="2.1 μm",
        marker="o",
    ),
    2.9: DiameterConfig(
        value_um=2.9,
        data_file="data_2.csv",
        surrogate_dir="2.9um",
        color="#ff7f0e",
        label="2.9 μm",
        marker="s",
    ),
    3.0: DiameterConfig(
        value_um=3.0,
        data_file="data_3.csv",
        surrogate_dir="3.0um",
        color="#2ca02c",
        label="3.0 μm",
        marker="^",
    ),
}


def get_available_diameters() -> List[float]:
    return sorted(DIAMETER_REGISTRY.keys())


def get_diameter_config(diameter_um: float) -> DiameterConfig:
    if diameter_um not in DIAMETER_REGISTRY:
        available = get_available_diameters()
        raise ValueError(
            f"Diameter {diameter_um} μm not registered. Available diameters: {available}"
        )
    return DIAMETER_REGISTRY[diameter_um]


def get_data_file_path(diameter_um: float, project_root: Optional[Path] = None) -> Path:
    config = get_diameter_config(diameter_um)
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return project_root / "emb" / "compression" / "evalkit" / "data" / config.data_file


def get_surrogate_path(diameter_um: float, project_root: Optional[Path] = None) -> Path:
    config = get_diameter_config(diameter_um)
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return project_root / "emb" / "compression" / "surrogate" / "diameters" / config.surrogate_dir / "trained"


def validate_diameter(diameter_um: float) -> bool:
    return diameter_um in DIAMETER_REGISTRY


def get_plotting_style(diameter_um: float) -> Dict[str, str]:
    config = get_diameter_config(diameter_um)
    return {"color": config.color, "marker": config.marker, "label": config.label}


PARAM_NAMES_6 = ["Yt", "kb", "b1", "b2", "a3", "a4"]
PARAM_NAMES_8 = ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]
