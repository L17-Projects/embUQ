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
    lane: Optional[str] = None
    prior_ka: Optional[List[float]] = None
    prior_kb: Optional[List[float]] = None
    prior_ka_by_diameter_um: Optional[Dict[Any, List[float]]] = None
    prior_kb_by_diameter_um: Optional[Dict[Any, List[float]]] = None
    data_files: Optional[Dict[Any, str]] = None
    diameter_labels: Optional[Dict[Any, str]] = None
    surrogate_parameterization: str = "legacy_yt_kb"
    grouped_reference_data: bool = False
    reference_diameters: Optional[List[float]] = None

    @property
    def diameters(self) -> List[float]:
        if self.structure != "emb":
            return []
        return [diameter_from_geometry_id(geometry) for geometry in self.geometries]

    @property
    def experiment_id(self) -> str:
        return canonical_experiment_id(self.structure, self.routing_name)

    @property
    def routing_name(self) -> str:
        if not self.lane:
            return self.name
        return f"{self.name}_{self.lane}"

    def _lookup_diameter_mapping(
        self,
        mapping: Optional[Dict[Any, str]],
        diameter_um: float,
    ) -> Optional[str]:
        if not mapping:
            return None
        keys = (
            diameter_um,
            str(diameter_um),
            format_emb_diameter(diameter_um),
            emb_geometry_id(diameter_um),
        )
        for key in keys:
            if key in mapping:
                return str(mapping[key])
        return None

    def _lookup_diameter_value(
        self,
        mapping: Optional[Dict[Any, Any]],
        diameter_um: float,
    ) -> Optional[Any]:
        if not mapping:
            return None
        keys = (
            diameter_um,
            str(diameter_um),
            format_emb_diameter(diameter_um),
            emb_geometry_id(diameter_um),
        )
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    def _prior_override(
        self,
        name: str,
        mapping: Optional[Dict[Any, List[float]]],
        diameter_um: float,
    ) -> Optional[List[float]]:
        value = self._lookup_diameter_value(mapping, diameter_um)
        if value is None:
            return None
        try:
            bounds = [float(item) for item in value]
        except TypeError as exc:
            raise ValueError(
                f"{name} prior override for {diameter_um:g} um must be a two-value sequence."
            ) from exc
        if len(bounds) != 2:
            raise ValueError(
                f"{name} prior override for {diameter_um:g} um must have exactly two values."
            )
        if bounds[0] >= bounds[1]:
            raise ValueError(
                f"{name} prior override for {diameter_um:g} um must satisfy min < max."
            )
        return bounds

    def _experiment_prior_override(
        self,
        name: str,
        value: Optional[List[float]],
    ) -> Optional[List[float]]:
        if value is None:
            return None
        try:
            bounds = [float(item) for item in value]
        except TypeError as exc:
            raise ValueError(
                f"{name} experiment prior override must be a two-value sequence."
            ) from exc
        if len(bounds) != 2:
            raise ValueError(f"{name} experiment prior override must have exactly two values.")
        if bounds[0] >= bounds[1]:
            raise ValueError(f"{name} experiment prior override must satisfy min < max.")
        return bounds

    def phase1_prior_overrides(self, diameter_um: float) -> Dict[str, List[float]]:
        overrides: Dict[str, List[float]] = {}
        ka = self._experiment_prior_override("ka", self.prior_ka)
        kb = self._experiment_prior_override("kb", self.prior_kb)
        ka_diameter = self._prior_override("ka", self.prior_ka_by_diameter_um, diameter_um)
        kb_diameter = self._prior_override("kb", self.prior_kb_by_diameter_um, diameter_um)
        if ka_diameter is not None:
            ka = ka_diameter
        if kb_diameter is not None:
            kb = kb_diameter
        if ka is not None:
            overrides["ka"] = ka
        if kb is not None:
            overrides["kb"] = kb
        return overrides

    def phase1_prior_override_sources(self, diameter_um: float) -> Dict[str, str]:
        sources: Dict[str, str] = {}
        if self._experiment_prior_override("ka", self.prior_ka) is not None:
            sources["ka"] = "experiment_override"
        if self._experiment_prior_override("kb", self.prior_kb) is not None:
            sources["kb"] = "experiment_override"
        if self._prior_override("ka", self.prior_ka_by_diameter_um, diameter_um) is not None:
            sources["ka"] = "diameter_override"
        if self._prior_override("kb", self.prior_kb_by_diameter_um, diameter_um) is not None:
            sources["kb"] = "diameter_override"
        return sources

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
        return canonical_dataset_id(self.structure, self.routing_name, geometry, selected_control)

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
            diameter_label = self._lookup_diameter_mapping(
                self.diameter_labels,
                diameter_um,
            ) or format_emb_diameter(diameter_um)
            suffix = "um"
            try:
                float(diameter_label)
            except ValueError:
                suffix = ""
            return f"{self.routing_name}_{diameter_label}{suffix}"
        geometry = str(geometry_or_diameter)
        return self.canonical_dataset_id(geometry, control=control)

    def data_file(self, diameter_um: float, *, use_diameter_labels: bool = True) -> Path:
        if self.data_files:
            data_file = self._lookup_diameter_mapping(self.data_files, diameter_um)
            if data_file is not None:
                return self.data_dir / data_file
        diameter_label = (
            self._lookup_diameter_mapping(self.diameter_labels, diameter_um)
            if use_diameter_labels
            else None
        ) or format_emb_diameter(diameter_um)
        filename = f"{self.data_prefix}{diameter_label}um.dat"
        return self.data_dir / filename

    def _resolve_reference_file(
        self,
        diameter_um: float,
        *,
        use_diameter_labels: bool = True,
    ) -> Path:
        data_file = self.data_file(diameter_um, use_diameter_labels=use_diameter_labels)
        if data_file.suffix == ".csv" and self.name == "indentation":
            diameter_label = self._lookup_diameter_mapping(
                self.diameter_labels,
                diameter_um,
            ) or format_emb_diameter(diameter_um)
            dpd_file = self.data_dir / f"{self.data_prefix}{diameter_label}um.dat"
            if not dpd_file.exists():
                raise FileNotFoundError(
                    f"Indentation DPD data missing: {dpd_file}. "
                    "Run the workflow preparation step to regenerate it."
                )
            return dpd_file
        if data_file.suffix == ".csv" and self.name == "compression":
            diameter_label = self._lookup_diameter_mapping(
                self.diameter_labels,
                diameter_um,
            ) or format_emb_diameter(diameter_um)
            dpd_file = self.data_dir / f"{self.data_prefix}{diameter_label}um.dat"
            if not dpd_file.exists():
                raise FileNotFoundError(
                    f"Compression DPD data missing: {dpd_file}. "
                    "Run the workflow preparation step to regenerate it."
                )
            return dpd_file
        return data_file

    def _reference_diameters_for(self, diameter_um: float) -> List[float]:
        if not self.grouped_reference_data:
            return [float(diameter_um)]
        if not self.reference_diameters:
            raise ValueError(
                f"Grouped reference experiment '{self.experiment_id}' requires reference_diameters."
            )
        return [float(value) for value in self.reference_diameters]

    def iter_reference_rows(self, diameter_um: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for reference_diameter in self._reference_diameters_for(diameter_um):
            data_file = self._resolve_reference_file(
                reference_diameter,
                use_diameter_labels=not self.grouped_reference_data,
            )
            if not data_file.exists():
                raise FileNotFoundError(f"Reference data not found: {data_file}")
            data = np.loadtxt(data_file, skiprows=1, ndmin=2)
            for point, value in data[:, :2]:
                rows.append(
                    {
                        "configured_diameter_um": float(diameter_um),
                        "source_diameter_um": float(reference_diameter),
                        "point": float(point),
                        "value": float(value),
                        "data_file": data_file,
                    }
                )
        return rows

    def get_reference_points(self, diameter_um: float) -> List[float]:
        return [row["point"] for row in self.iter_reference_rows(diameter_um)]

    def get_reference_data(self, diameter_um: float) -> List[float]:
        return [row["value"] for row in self.iter_reference_rows(diameter_um)]


def _default_experiment_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    name = config.get("experiment", "compression")
    structure = infer_structure(name, config.get("structure"))
    if structure is None:
        raise ValueError(f"Experiment '{name}' requires an explicit structure to avoid ambiguous references")
    return [
        {
            "structure": structure,
            "name": name,
            "lane": config.get("lane"),
            "enabled": True,
            "diameters": config.get("emb_diameters"),
            "geometries": config.get("geometries"),
            "data_dir": config.get("data_dir", f"{name}/evalkit/data"),
            "data_prefix": config.get("data_prefix", f"{name}_data_"),
            "data_files": config.get("data_files"),
            "surrogate_dir": config.get("surrogate_dir", f"{name}/surrogate/diameters"),
            "prior_ka": config.get("prior_ka"),
            "prior_kb": config.get("prior_kb"),
            "prior_d0": config.get("prior_d0", [0.0, 0.5]),
            "prior_sigma": config.get("prior_sigma", [0.0, 1.0]),
            "prior_ka_by_diameter_um": config.get("prior_ka_by_diameter_um"),
            "prior_kb_by_diameter_um": config.get("prior_kb_by_diameter_um"),
            "controls": ["default"],
            "diameter_labels": config.get("diameter_labels"),
            "surrogate_parameterization": config.get(
                "surrogate_parameterization",
                "legacy_yt_kb",
            ),
            "grouped_reference_data": config.get("grouped_reference_data", False),
            "reference_diameters": config.get("reference_diameters"),
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
        lane = exp.get("lane")
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
        routing_name = f"{name}_{lane}" if lane else name
        experiment_id = canonical_experiment_id(structure, routing_name)
        if experiment_id in seen_experiments:
            raise ValueError(f"Duplicate experiment reference '{experiment_id}'")
        seen_experiments.add(experiment_id)
        data_dir = Path(project_root, exp.get("data_dir", f"{name}/evalkit/data"))
        surrogate_dir = Path(project_root, exp.get("surrogate_dir", f"{name}/surrogate/diameters"))
        data_prefix = exp.get("data_prefix", f"{routing_name}_data_")
        data_files = exp.get("data_files")
        prior_ka = exp.get("prior_ka")
        prior_kb = exp.get("prior_kb")
        prior_d0 = exp.get("prior_d0", config.get("prior_d0", [0.0, 0.5]))
        prior_sigma = exp.get("prior_sigma", config.get("prior_sigma", [0.0, 1.0]))
        prior_ka_by_diameter_um = exp.get("prior_ka_by_diameter_um")
        prior_kb_by_diameter_um = exp.get("prior_kb_by_diameter_um")
        controls = sorted(set(exp.get("controls", ["default"])))
        diameter_labels = exp.get("diameter_labels")
        surrogate_parameterization = str(
            exp.get("surrogate_parameterization", "legacy_yt_kb")
        )
        grouped_reference_data = bool(exp.get("grouped_reference_data", False))
        reference_diameters = exp.get("reference_diameters")
        if reference_diameters is not None:
            reference_diameters = [float(value) for value in reference_diameters]

        experiments.append(
            ExperimentSpec(
                structure=structure,
                name=name,
                lane=lane,
                geometries=geometries,
                data_dir=data_dir,
                data_prefix=data_prefix,
                surrogate_dir=surrogate_dir,
                enabled=enabled,
                prior_ka=prior_ka,
                prior_kb=prior_kb,
                prior_d0=prior_d0,
                prior_sigma=prior_sigma,
                prior_ka_by_diameter_um=prior_ka_by_diameter_um,
                prior_kb_by_diameter_um=prior_kb_by_diameter_um,
                controls=controls,
                data_files=data_files,
                diameter_labels=diameter_labels,
                surrogate_parameterization=surrogate_parameterization,
                grouped_reference_data=grouped_reference_data,
                reference_diameters=reference_diameters,
            )
        )

    return experiments
