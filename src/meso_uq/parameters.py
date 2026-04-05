from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    latex: str
    description: str
    units: str
    prior_min: float
    prior_max: float
    is_material: bool = True
    physical_constraint: Optional[str] = None


PARAM_YT = ParameterInfo("Yt", r"$Y_t$", "Young's modulus-like tensile stiffness parameter", "DPD energy units", 1e7, 1e8)
PARAM_KB = ParameterInfo("kb", r"$k_b$", "Bending modulus of the membrane", "DPD energy units", 100.0, 800.0)
PARAM_B1 = ParameterInfo("b1", r"$b_1$", "First strain-limiting parameter", "dimensionless", 0.0, 3.0)
PARAM_B2 = ParameterInfo("b2", r"$b_2$", "Second strain-limiting parameter", "dimensionless", 0.0, 10.0)
PARAM_A3 = ParameterInfo("a3", r"$a_3$", "Third-order area strain coefficient", "dimensionless", -2.5, 3.0)
PARAM_A4 = ParameterInfo("a4", r"$a_4$", "Fourth-order area strain coefficient", "dimensionless", 0.0, 4.0)
PARAM_D0 = ParameterInfo("d0", r"$d_0$", "Displacement offset", "DPD length units", 0.0, 0.5, False)
PARAM_SIGMA = ParameterInfo("sigma", r"$\sigma$", "Measurement noise standard deviation", "DPD force units", 0.0, 1.0, False)


@dataclass
class ParameterSet:
    name: str
    description: str
    parameters: List[ParameterInfo] = field(default_factory=list)

    @property
    def n_params(self) -> int:
        return len(self.parameters)

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def get_info(self, name: str) -> ParameterInfo:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"Parameter '{name}' not found in {self.name}")

    def get_bounds(self, name: str) -> Tuple[float, float]:
        info = self.get_info(name)
        return (info.prior_min, info.prior_max)

    def validate_values(self, values: Dict[str, float]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for p in self.parameters:
            if p.name not in values:
                errors.append(f"Missing parameter: {p.name}")
                continue
            if not (p.prior_min <= values[p.name] <= p.prior_max):
                errors.append(f"{p.name}={values[p.name]} out of bounds [{p.prior_min}, {p.prior_max}]")
        return len(errors) == 0, errors


PARAM_6 = ParameterSet("6-parameter", "Material parameters only", [PARAM_YT, PARAM_KB, PARAM_B1, PARAM_B2, PARAM_A3, PARAM_A4])
PARAM_8 = ParameterSet("8-parameter", "Full inference model", [PARAM_YT, PARAM_KB, PARAM_B1, PARAM_B2, PARAM_A3, PARAM_A4, PARAM_D0, PARAM_SIGMA])
_PARAMETER_SETS = {6: PARAM_6, 8: PARAM_8}


def get_parameter_set(n_params: int) -> ParameterSet:
    if n_params not in _PARAMETER_SETS:
        raise ValueError(f"Unsupported parameter count: {n_params}. Use 6 or 8.")
    return _PARAMETER_SETS[n_params]


def create_custom_parameter_set(name: str, param_names: List[str], bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> ParameterSet:
    all_params = {"Yt": PARAM_YT, "kb": PARAM_KB, "b1": PARAM_B1, "b2": PARAM_B2, "a3": PARAM_A3, "a4": PARAM_A4, "d0": PARAM_D0, "sigma": PARAM_SIGMA}
    params = []
    for pname in param_names:
        if pname not in all_params:
            raise ValueError(f"Unknown parameter: {pname}")
        base = all_params[pname]
        if bounds and pname in bounds:
            params.append(ParameterInfo(base.name, base.latex, base.description, base.units, bounds[pname][0], bounds[pname][1], base.is_material, base.physical_constraint))
        else:
            params.append(base)
    return ParameterSet(name=name, description=f"Custom parameter set: {name}", parameters=params)
