from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from meso_uq.noise.covariance import CovarianceBuildResult


_GEOMETRY_TOTAL = "geometry_jacobian"


def _finite_float(value: float, label: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{label} must be finite; got {value}.")
    return value


def _nonnegative_float(value: float, label: str) -> float:
    value = _finite_float(value, label)
    if value < 0.0:
        raise ValueError(f"{label} must be >= 0.0; got {value}.")
    return value


def _positive_float(value: float, label: str) -> float:
    value = _finite_float(value, label)
    if value <= 0.0:
        raise ValueError(f"{label} must be positive; got {value}.")
    return value


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    if np.allclose(diagonal, 0.0):
        return np.eye(covariance.shape[0], dtype=float)
    scale = np.sqrt(np.maximum(diagonal, 0.0))
    denominator = np.outer(scale, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if isfinite(value) else None


def _coerce_covariance(value: Sequence[Sequence[float]], size: int) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (size, size):
        raise ValueError(f"geometry covariance shape {covariance.shape} does not match parameter count {size}.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("geometry covariance values must be finite.")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("geometry covariance must be symmetric.")
    symmetric = 0.5 * (covariance + covariance.T)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
    if min_eigenvalue < -1e-12:
        raise ValueError("geometry covariance must be positive semidefinite.")
    return symmetric


def _cholesky_with_optional_jitter(covariance: np.ndarray, jitter: float, max_jitter: float) -> tuple[np.ndarray | None, float]:
    if np.allclose(covariance, 0.0):
        return None, 0.0
    symmetric = 0.5 * (covariance + covariance.T)
    try:
        return np.linalg.cholesky(symmetric), 0.0
    except np.linalg.LinAlgError:
        pass
    if max_jitter <= 0.0:
        return None, 0.0
    candidate = jitter if jitter > 0.0 else min(max_jitter, 1e-12)
    eye = np.eye(symmetric.shape[0], dtype=float)
    while candidate <= max_jitter * (1.0 + 1e-12):
        try:
            return np.linalg.cholesky(symmetric + candidate * eye), candidate
        except np.linalg.LinAlgError:
            candidate *= 10.0
    return None, 0.0


@dataclass(frozen=True)
class GeometryParameterUncertainty:
    name: str
    sigma: float
    units: str
    nominal: float | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("geometry parameter name must be non-empty.")
        units = str(self.units).strip()
        if not units:
            raise ValueError(f"geometry parameter '{name}' requires units.")
        sigma = _nonnegative_float(self.sigma, f"geometry parameter '{name}' sigma")
        nominal = None if self.nominal is None else _positive_float(self.nominal, f"geometry parameter '{name}' nominal")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "nominal", nominal)


@dataclass(frozen=True)
class GeometryUncertaintyConfig:
    enabled: bool = True
    parameters: tuple[GeometryParameterUncertainty, ...] = ()
    covariance: tuple[tuple[float, ...], ...] | None = None
    jitter: float = 0.0
    max_jitter: float = 0.0

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        names = [parameter.name for parameter in parameters]
        if len(set(names)) != len(names):
            raise ValueError("geometry parameter names must be unique.")
        jitter = _nonnegative_float(self.jitter, "jitter")
        max_jitter = _nonnegative_float(self.max_jitter, "max_jitter")
        if max_jitter < jitter:
            raise ValueError("max_jitter must be >= jitter.")
        if self.enabled and not parameters:
            raise ValueError("enabled geometry uncertainty requires at least one parameter.")
        covariance = None
        if self.covariance is not None:
            covariance_array = _coerce_covariance(self.covariance, len(parameters))
            covariance = tuple(tuple(float(value) for value in row) for row in covariance_array.tolist())
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "max_jitter", max_jitter)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def active(self) -> bool:
        if not self.enabled:
            return False
        if self.covariance is not None:
            return bool(np.any(np.asarray(self.covariance, dtype=float) != 0.0))
        return any(parameter.sigma > 0.0 for parameter in self.parameters)

    def parameter_covariance(self) -> np.ndarray:
        if self.covariance is not None:
            return np.asarray(self.covariance, dtype=float)
        return np.diag([parameter.sigma * parameter.sigma for parameter in self.parameters])


@dataclass(frozen=True)
class GeometrySensitivityInputs:
    predictions: tuple[float, ...]
    sensitivities: Mapping[str, tuple[float, ...]]
    curve_id: str = "curve"

    def __post_init__(self) -> None:
        predictions = _finite_vector(self.predictions, "predictions")
        sensitivities: dict[str, tuple[float, ...]] = {}
        for name, values in self.sensitivities.items():
            key = str(name).strip()
            if not key:
                raise ValueError("sensitivity names must be non-empty.")
            vector = _finite_vector(values, f"geometry sensitivity '{key}'")
            if len(vector) != len(predictions):
                raise ValueError(
                    f"geometry sensitivity '{key}' count {len(vector)} does not match prediction count {len(predictions)}."
                )
            sensitivities[key] = vector
        curve_id = str(self.curve_id).strip()
        if not curve_id:
            raise ValueError("curve_id must be non-empty.")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "sensitivities", sensitivities)
        object.__setattr__(self, "curve_id", curve_id)


@dataclass(frozen=True)
class GeometryUncertaintyResult:
    covariance: CovarianceBuildResult
    covariance_components: Mapping[str, np.ndarray]
    variance_components: Mapping[str, tuple[float, ...]]
    standard_deviation: tuple[float, ...]
    summary: Mapping[str, Any]


def build_geometry_uncertainty_covariance(
    inputs: GeometrySensitivityInputs,
    config: GeometryUncertaintyConfig,
) -> GeometryUncertaintyResult:
    size = len(inputs.predictions)
    if not config.enabled:
        zero = np.zeros((size, size), dtype=float)
        return _build_result(inputs.curve_id, config, zero, {}, enabled=False, active=False)

    missing = [name for name in config.parameter_names if name not in inputs.sensitivities]
    if missing:
        raise ValueError("Missing geometry sensitivity for parameter(s): " + ", ".join(missing))
    extras = [name for name in inputs.sensitivities if name not in config.parameter_names]
    if extras:
        raise ValueError("Unknown geometry sensitivity parameter(s): " + ", ".join(extras))

    jacobian = np.asarray([inputs.sensitivities[name] for name in config.parameter_names], dtype=float).T
    parameter_covariance = config.parameter_covariance()
    total = jacobian @ parameter_covariance @ jacobian.T
    components: dict[str, np.ndarray] = {}
    for index, parameter in enumerate(config.parameters):
        variance = parameter_covariance[index, index]
        column = jacobian[:, index]
        components[f"geometry:{parameter.name}"] = variance * np.outer(column, column)
    return _build_result(inputs.curve_id, config, total, components, enabled=True, active=config.active)


def _build_result(
    curve_id: str,
    config: GeometryUncertaintyConfig,
    covariance: np.ndarray,
    components: Mapping[str, np.ndarray],
    *,
    enabled: bool,
    active: bool,
) -> GeometryUncertaintyResult:
    cholesky, jitter_added = _cholesky_with_optional_jitter(covariance, config.jitter, config.max_jitter)
    if cholesky is not None and jitter_added > 0.0:
        covariance = covariance + jitter_added * np.eye(covariance.shape[0], dtype=float)
    correlation = _correlation_from_covariance(covariance)
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    condition_number = np.linalg.cond(covariance) if covariance.size and not np.allclose(covariance, 0.0) else float("nan")
    covariance_components = {name: np.asarray(component, dtype=float) for name, component in components.items()}
    covariance_components[_GEOMETRY_TOTAL] = covariance
    variance_components = {name: tuple(float(value) for value in np.diag(component)) for name, component in covariance_components.items()}
    standard_deviation = tuple(float(sqrt(max(value, 0.0))) for value in np.diag(covariance))
    summary = {
        "schema_version": 1,
        "enabled": bool(enabled),
        "active": bool(active),
        "curve_id": curve_id,
        "model": "geometry_jacobian_covariance",
        "parameter_names": list(config.parameter_names),
        "parameter_units": {parameter.name: parameter.units for parameter in config.parameters},
        "shape": list(covariance.shape),
        "diagonal_min": _finite_or_none(np.min(np.diag(covariance))),
        "diagonal_max": _finite_or_none(np.max(np.diag(covariance))),
        "min_eigenvalue": _finite_or_none(np.min(eigenvalues)),
        "max_eigenvalue": _finite_or_none(np.max(eigenvalues)),
        "condition_number": _finite_or_none(condition_number),
        "jitter_added": float(jitter_added),
        "cholesky_success": cholesky is not None,
    }
    covariance_result = CovarianceBuildResult(
        covariance=covariance,
        correlation=correlation,
        cholesky=cholesky,
        jitter_added=jitter_added,
        summary=summary,
    )
    return GeometryUncertaintyResult(
        covariance=covariance_result,
        covariance_components=covariance_components,
        variance_components=variance_components,
        standard_deviation=standard_deviation,
        summary=summary,
    )
