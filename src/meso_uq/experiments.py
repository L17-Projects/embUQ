"""
Experiment registry and helpers for multi-experiment inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    diameters: List[float]
    data_dir: Path
    data_prefix: str
    surrogate_dir: Path
    enabled: bool
    prior_d0: List[float]
    prior_sigma: List[float]
    data_files: Optional[Dict[Any, str]] = None

    def dataset_name(self, diameter_um: float) -> str:
        return f"{self.name}_{diameter_um}um"

    def data_file(self, diameter_um: float) -> Path:
        if self.data_files:
            if diameter_um in self.data_files:
                return self.data_dir / self.data_files[diameter_um]
            diameter_key = str(diameter_um)
            if diameter_key in self.data_files:
                return self.data_dir / self.data_files[diameter_key]
        filename = f"{self.data_prefix}{diameter_um}um.dat"
        return self.data_dir / filename

    def _resolve_reference_file(self, diameter_um: float) -> Path:
        data_file = self.data_file(diameter_um)
        if data_file.suffix == ".csv" and self.name == "indentation":
            dpd_file = self.data_dir / f"{self.data_prefix}{diameter_um}um.dat"
            if not dpd_file.exists():
                raise FileNotFoundError(
                    f"Indentation DPD data missing: {dpd_file}. "
                    "Run the workflow preparation step to regenerate it."
                )
            return dpd_file
        if data_file.suffix == ".csv" and self.name == "compression":
            dpd_file = self.data_dir / f"{self.data_prefix}{diameter_um}um.dat"
            if not dpd_file.exists():
                raise FileNotFoundError(
                    f"Compression DPD data missing: {dpd_file}. "
                    "Run the workflow preparation step to regenerate it."
                )
            return dpd_file
        return data_file

    def get_reference_points(self, diameter_um: float) -> List[float]:
        data_file = self._resolve_reference_file(diameter_um)
        if not data_file.exists():
            raise FileNotFoundError(f"Reference data not found: {data_file}")
        data = np.loadtxt(data_file, skiprows=1, ndmin=2)
        return list(data[:, 0])

    def get_reference_data(self, diameter_um: float) -> List[float]:
        data_file = self._resolve_reference_file(diameter_um)
        if not data_file.exists():
            raise FileNotFoundError(f"Reference data not found: {data_file}")
        data = np.loadtxt(data_file, skiprows=1, ndmin=2)
        return list(data[:, 1])


def _default_experiment_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    name = config.get("experiment", "compression")
    return [
        {
            "name": name,
            "enabled": True,
            "diameters": config["emb_diameters"],
            "data_dir": config.get("data_dir", f"{name}/evalkit/data"),
            "data_prefix": config.get("data_prefix", f"{name}_data_"),
            "data_files": config.get("data_files"),
            "surrogate_dir": config.get("surrogate_dir", f"{name}/surrogate/diameters"),
            "prior_d0": config.get("prior_d0", [0.0, 0.5]),
            "prior_sigma": config.get("prior_sigma", [0.0, 1.0]),
        }
    ]


def load_experiments(config: Dict[str, Any], project_root: Path) -> List[ExperimentSpec]:
    raw_experiments = config.get("experiments")
    if not raw_experiments:
        raw_experiments = _default_experiment_config(config)

    experiments: List[ExperimentSpec] = []
    for exp in raw_experiments:
        name = exp["name"]
        enabled = exp.get("enabled", True)
        diameters = exp.get("diameters", config.get("emb_diameters", []))
        if not diameters:
            raise ValueError(f"Experiment '{name}' must define diameters")
        data_dir = Path(project_root, exp.get("data_dir", f"{name}/evalkit/data"))
        surrogate_dir = Path(project_root, exp.get("surrogate_dir", f"{name}/surrogate/diameters"))
        data_prefix = exp.get("data_prefix", f"{name}_data_")
        data_files = exp.get("data_files")
        prior_d0 = exp.get("prior_d0", config.get("prior_d0", [0.0, 0.5]))
        prior_sigma = exp.get("prior_sigma", config.get("prior_sigma", [0.0, 1.0]))

        experiments.append(
            ExperimentSpec(
                name=name,
                diameters=sorted(diameters),
                data_dir=data_dir,
                data_prefix=data_prefix,
                surrogate_dir=surrogate_dir,
                enabled=enabled,
                prior_d0=prior_d0,
                prior_sigma=prior_sigma,
                data_files=data_files,
            )
        )

    return experiments
