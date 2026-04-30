"""
Experiment registry and helpers for multi-experiment inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from meso_uq.config.models import emb_geometry_id, format_emb_diameter, infer_structure


def canonical_experiment_id(structure: str, experiment: str) -> str:
    return f"{structure}:{experiment}"


def canonical_dataset_id(structure: str, experiment: str, geometry: str, controls: str = "default") -> str:
    return f"{structure}:{experiment}:{geometry}:{controls}"


def diameter_from_geometry_id(geometry: str) -> float:
    prefix = "diameter_"
    suffix = "um"
    if not geometry.startswith(prefix) or not geometry.endswith(suffix):
        raise ValueError(f"Geometry '{geometry}' is not a normalized EMB diameter geometry")
    return float(geometry[len(prefix) : -len(suffix)])


@dataclass(frozen=True)
class ExperimentSpec:
    structure: str
    name: str
    geometries: List[str]
    data_dir: Path
    data_prefix: str
    surrogate_dir: Path
    enabled: bool
    prior_d0: List[float]
    prior_sigma: List[float]
    controls: List[str]
    data_files: Optional[Dict[Any, str]] = None

    @property
    def diameters(self) -> List[float]:
        if self.structure != "emb":
            return []
        return [diameter_from_geometry_id(geometry) for geometry in self.geometries]

    @property
    def experiment_id(self) -> str:
        return canonical_experiment_id(self.structure, self.name)

    def default_control(self) -> str:
        if len(self.controls) != 1:
            raise ValueError(
                f"Experiment '{self.experiment_id}' requires an explicit control selection; "
                f"available controls: {self.controls}"
            )
        return self.controls[0]

    def canonical_dataset_id(self, geometry: str, control: Optional[str] = None) -> str:
        selected_control = control or self.default_control()
        if selected_control not in self.controls:
            raise ValueError(
                f"Control '{selected_control}' is not configured for experiment '{self.experiment_id}'"
            )
        if geometry not in self.geometries:
            raise ValueError(f"Geometry '{geometry}' is not configured for experiment '{self.experiment_id}'")
        return canonical_dataset_id(self.structure, self.name, geometry, selected_control)

    def dataset_name(self, geometry_or_diameter: Any, control: Optional[str] = None) -> str:
        if self.structure == "emb":
            diameter_um = (
                float(geometry_or_diameter)
                if isinstance(geometry_or_diameter, (float, int))
                else diameter_from_geometry_id(str(geometry_or_diameter))
            )
            geometry = emb_geometry_id(diameter_um)
            if geometry not in self.geometries:
                raise ValueError(f"Geometry '{geometry}' is not configured for experiment '{self.experiment_id}'")
            return f"{self.name}_{format_emb_diameter(diameter_um)}um"
        geometry = str(geometry_or_diameter)
        return self.canonical_dataset_id(geometry, control=control)

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
    structure = infer_structure(name, config.get("structure"))
    if structure is None:
        raise ValueError(f"Experiment '{name}' requires an explicit structure to avoid ambiguous references")
    return [
        {
            "structure": structure,
            "name": name,
            "enabled": True,
            "diameters": config.get("emb_diameters"),
            "geometries": config.get("geometries"),
            "data_dir": config.get("data_dir", f"{name}/evalkit/data"),
            "data_prefix": config.get("data_prefix", f"{name}_data_"),
            "data_files": config.get("data_files"),
            "surrogate_dir": config.get("surrogate_dir", f"{name}/surrogate/diameters"),
            "prior_d0": config.get("prior_d0", [0.0, 0.5]),
            "prior_sigma": config.get("prior_sigma", [0.0, 1.0]),
            "controls": ["default"],
        }
    ]


def load_experiments(config: Dict[str, Any], project_root: Path) -> List[ExperimentSpec]:
    raw_experiments = config.get("experiments")
    if not raw_experiments:
        raw_experiments = _default_experiment_config(config)

    experiments: List[ExperimentSpec] = []
    seen_experiments: set[str] = set()
    for exp in raw_experiments:
        name = exp["name"]
        structure = infer_structure(name, exp.get("structure", config.get("structure")))
        if structure is None:
            raise ValueError(f"Experiment '{name}' requires an explicit structure to avoid ambiguous references")
        enabled = exp.get("enabled", True)
        geometries = exp.get("geometries")
        diameters = exp.get("diameters", config.get("emb_diameters"))
        if structure == "emb":
            if geometries is None and diameters:
                geometries = [emb_geometry_id(diameter) for diameter in diameters]
            if not geometries:
                raise ValueError(f"Experiment '{name}' must define diameters")
        elif not geometries:
            raise ValueError(f"Experiment '{name}' must define geometries")
        geometries = sorted(set(geometries))
        experiment_id = canonical_experiment_id(structure, name)
        if experiment_id in seen_experiments:
            raise ValueError(f"Duplicate experiment reference '{experiment_id}'")
        seen_experiments.add(experiment_id)
        data_dir = Path(project_root, exp.get("data_dir", f"{name}/evalkit/data"))
        surrogate_dir = Path(project_root, exp.get("surrogate_dir", f"{name}/surrogate/diameters"))
        data_prefix = exp.get("data_prefix", f"{name}_data_")
        data_files = exp.get("data_files")
        prior_d0 = exp.get("prior_d0", config.get("prior_d0", [0.0, 0.5]))
        prior_sigma = exp.get("prior_sigma", config.get("prior_sigma", [0.0, 1.0]))
        controls = sorted(set(exp.get("controls", ["default"])))

        experiments.append(
            ExperimentSpec(
                structure=structure,
                name=name,
                geometries=geometries,
                data_dir=data_dir,
                data_prefix=data_prefix,
                surrogate_dir=surrogate_dir,
                enabled=enabled,
                prior_d0=prior_d0,
                prior_sigma=prior_sigma,
                controls=controls,
                data_files=data_files,
            )
        )

    return experiments
