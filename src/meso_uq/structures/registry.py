from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    latex: str
    description: str
    units: str
    prior_min: float | None = None
    prior_max: float | None = None
    is_calibrated: bool = True
    is_nuisance: bool = False
    optional: bool = False


@dataclass(frozen=True)
class NoiseModelSpec:
    kind: str
    parameter: str
    description: str


@dataclass(frozen=True)
class ParameterContract:
    calibrated: tuple[ParameterSpec, ...]
    nuisance: tuple[ParameterSpec, ...] = ()
    noise_model: NoiseModelSpec | None = None

    @property
    def calibrated_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.calibrated)

    @property
    def nuisance_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.nuisance)

    def get_parameter(self, name: str) -> ParameterSpec:
        for parameter in self.calibrated + self.nuisance:
            if parameter.name == name:
                return parameter
        raise KeyError(f"Unknown parameter '{name}'.")


@dataclass(frozen=True)
class ControlSpec:
    name: str
    description: str
    units: str
    default: float | None = None


@dataclass(frozen=True)
class GeometrySpec:
    id: str
    label: str
    shape: str
    parameters: Mapping[str, float]
    source: str


@dataclass(frozen=True)
class ObservableSpec:
    name: str
    description: str
    units: str


@dataclass(frozen=True)
class ExperimentSpec:
    structure: str
    name: str
    description: str
    controls: tuple[ControlSpec, ...]
    observables: tuple[ObservableSpec, ...]
    experimental: bool = False
    requires_opt_in: bool = False
    opt_in_flag: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def control_names(self) -> tuple[str, ...]:
        return tuple(control.name for control in self.controls)


@dataclass(frozen=True)
class StructureSpec:
    name: str
    description: str
    parameter_contract: ParameterContract
    geometries: tuple[GeometrySpec, ...]
    experiments: tuple[ExperimentSpec, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def get_geometry(self, geometry_id: str) -> GeometrySpec:
        for geometry in self.geometries:
            if geometry.id == geometry_id:
                return geometry
        raise KeyError(f"Unknown geometry '{geometry_id}' for structure '{self.name}'.")

    def get_experiment(self, experiment_name: str, *, include_experimental: bool = False) -> ExperimentSpec:
        for experiment in self.experiments:
            if experiment.name != experiment_name:
                continue
            if experiment.requires_opt_in and not include_experimental:
                flag = experiment.opt_in_flag or "experimental opt-in"
                raise ValueError(f"Experiment '{experiment_name}' requires {flag}.")
            return experiment
        raise KeyError(f"Unknown experiment '{experiment_name}' for structure '{self.name}'.")


def _load_structures() -> dict[str, StructureSpec]:
    from .gv import GV_STRUCTURE

    return {GV_STRUCTURE.name: GV_STRUCTURE}


STRUCTURE_REGISTRY = _load_structures()


def get_structure(structure_name: str) -> StructureSpec:
    try:
        return STRUCTURE_REGISTRY[structure_name]
    except KeyError as exc:
        raise KeyError(f"Unknown structure '{structure_name}'.") from exc


def list_structures() -> tuple[StructureSpec, ...]:
    return tuple(STRUCTURE_REGISTRY.values())
