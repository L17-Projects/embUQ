"""Portable, bounded DPD breathing-frequency surface artifacts.

This module intentionally has no Korali, Mirheo, or YAML-runtime dependency.
It makes the DPD-derived resonance forward map testable before physical DPD
measurements are available.  The callback wiring may load this artifact, but
the artifact itself refuses to extrapolate in either ``ka`` or physical radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FREQUENCY_SURFACE_SCHEMA = "meso_uq.emb_dpd_frequency_surface.v1"
FREQUENCY_UNIT = "MHz"
RADIUS_UNIT = "um"
RADIUS_COORDINATE = "physical_radius"
FREQUENCY_SQUARED_TARGET = "frequency_squared_mhz2"
COEFFICIENT_ORDER = "u_major_v_minor"


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number.")
    return result


def _positive_float(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return result


def _bounds(payload: Any, field_name: str, *, lower_may_be_zero: bool) -> tuple[float, float]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or len(payload) != 2:
        raise ValueError(f"{field_name} must contain exactly [minimum, maximum].")
    lower = _finite_float(payload[0], f"{field_name}[0]")
    upper = _finite_float(payload[1], f"{field_name}[1]")
    if upper <= lower or (not lower_may_be_zero and lower <= 0.0) or (lower_may_be_zero and lower < 0.0):
        raise ValueError(f"{field_name} must be an increasing supported range.")
    return lower, upper


def _agent_name(value: Any) -> str:
    result = str(value).strip().lower()
    if result not in {"definity", "sonovue"}:
        raise ValueError("agent must be either 'definity' or 'sonovue'.")
    return result


def _optional_mapping(payload: Any, field_name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return dict(payload)


def _coefficient_array(
    payload: Any,
    *,
    ka_degree: int,
    radius_degree: int,
) -> np.ndarray:
    values = np.asarray(payload, dtype=np.float64)
    expected_shape = (ka_degree + 1, radius_degree + 1)
    if values.shape != expected_shape:
        raise ValueError(
            "polynomial.coefficients must have shape "
            f"{expected_shape}, got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("polynomial.coefficients must contain only finite values.")
    return values


def _broadcast_inputs(ka_dpd: Any, radius_um: Any) -> tuple[np.ndarray, np.ndarray]:
    ka = np.asarray(ka_dpd, dtype=np.float64)
    radius = np.asarray(radius_um, dtype=np.float64)
    try:
        ka, radius = np.broadcast_arrays(ka, radius)
    except ValueError as error:
        raise ValueError("ka_dpd and radius_um must be broadcast-compatible.") from error
    if not np.all(np.isfinite(ka)):
        raise ValueError("ka_dpd must contain only finite values.")
    if not np.all(np.isfinite(radius)):
        raise ValueError("radius_um must contain only finite values.")
    return ka, radius


@dataclass(frozen=True)
class DpdFrequencySurface:
    """A strict-support tensor polynomial for DPD breathing frequency squared."""

    agent: str
    ka_bounds_dpd: tuple[float, float]
    radius_bounds_um: tuple[float, float]
    coefficients: np.ndarray
    ka_degree: int = 3
    radius_degree: int = 3
    conditions: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent", _agent_name(self.agent))
        object.__setattr__(
            self,
            "ka_bounds_dpd",
            _bounds(self.ka_bounds_dpd, "ka_bounds_dpd", lower_may_be_zero=True),
        )
        object.__setattr__(
            self,
            "radius_bounds_um",
            _bounds(self.radius_bounds_um, "radius_bounds_um", lower_may_be_zero=False),
        )
        if int(self.ka_degree) < 0 or int(self.radius_degree) < 0:
            raise ValueError("Polynomial degrees must be non-negative.")
        object.__setattr__(self, "ka_degree", int(self.ka_degree))
        object.__setattr__(self, "radius_degree", int(self.radius_degree))
        object.__setattr__(
            self,
            "coefficients",
            _coefficient_array(
                self.coefficients,
                ka_degree=self.ka_degree,
                radius_degree=self.radius_degree,
            ),
        )
        object.__setattr__(self, "conditions", dict(self.conditions))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "validation", dict(self.validation))

    @property
    def normalization(self) -> dict[str, float]:
        return {
            "ka_scale_dpd": self.ka_bounds_dpd[1],
            "radius_min_um": self.radius_bounds_um[0],
            "radius_max_um": self.radius_bounds_um[1],
        }

    def _normalized_coordinates(self, ka_dpd: Any, radius_um: Any) -> tuple[np.ndarray, np.ndarray]:
        ka, radius = _broadcast_inputs(ka_dpd, radius_um)
        ka_min, ka_max = self.ka_bounds_dpd
        radius_min, radius_max = self.radius_bounds_um
        if np.any(ka < ka_min) or np.any(ka > ka_max):
            raise ValueError(
                f"ka_dpd is outside the supported range [{ka_min:g}, {ka_max:g}]."
            )
        if np.any(radius < radius_min) or np.any(radius > radius_max):
            raise ValueError(
                "radius_um is outside the supported range "
                f"[{radius_min:g}, {radius_max:g}]."
            )
        return ka / ka_max, (radius - radius_min) / (radius_max - radius_min)

    def predict_frequency_squared_mhz2(self, ka_dpd: Any, radius_um: Any) -> np.ndarray:
        """Evaluate f^2 without allowing ka or radius extrapolation."""

        u, v = self._normalized_coordinates(ka_dpd, radius_um)
        result = np.zeros(u.shape, dtype=np.float64)
        for i in range(self.ka_degree + 1):
            for j in range(self.radius_degree + 1):
                result += self.coefficients[i, j] * (u**i) * (v**j)
        if np.any(~np.isfinite(result)) or np.any(result < 0.0):
            raise ValueError("Frequency surface produced negative or non-finite f_squared.")
        return result

    def predict_mhz(self, ka_dpd: Any, radius_um: Any) -> np.ndarray:
        """Evaluate physical breathing frequency in MHz."""

        return np.sqrt(self.predict_frequency_squared_mhz2(ka_dpd, radius_um))

    def predict_for_diameters_mhz(self, ka_dpd: Any, diameter_um: Any) -> np.ndarray:
        diameters = np.asarray(diameter_um, dtype=np.float64)
        if not np.all(np.isfinite(diameters)) or np.any(diameters <= 0.0):
            raise ValueError("diameter_um must contain finite positive values.")
        return self.predict_mhz(ka_dpd, diameters / 2.0)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": FREQUENCY_SURFACE_SCHEMA,
            "agent": self.agent,
            "response": {
                "frequency_unit": FREQUENCY_UNIT,
                "target": FREQUENCY_SQUARED_TARGET,
            },
            "coordinates": {
                "radius_coordinate": RADIUS_COORDINATE,
                "radius_unit": RADIUS_UNIT,
                "ka_bounds_dpd": list(self.ka_bounds_dpd),
                "radius_bounds_um": list(self.radius_bounds_um),
                "normalization": self.normalization,
            },
            "polynomial": {
                "ka_degree": self.ka_degree,
                "radius_degree": self.radius_degree,
                "coefficient_order": COEFFICIENT_ORDER,
                "coefficients": self.coefficients.tolist(),
            },
            "conditions": dict(self.conditions),
            "provenance": dict(self.provenance),
            "validation": dict(self.validation),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DpdFrequencySurface":
        if not isinstance(payload, Mapping):
            raise ValueError("Frequency surface artifact must be a mapping.")
        if payload.get("schema") != FREQUENCY_SURFACE_SCHEMA:
            raise ValueError(
                f"Unsupported frequency surface schema: {payload.get('schema')!r}."
            )
        response = payload.get("response")
        coordinates = payload.get("coordinates")
        polynomial = payload.get("polynomial")
        if not isinstance(response, Mapping) or not isinstance(coordinates, Mapping) or not isinstance(polynomial, Mapping):
            raise ValueError("Frequency surface artifact is missing response, coordinates, or polynomial mappings.")
        if response.get("frequency_unit") != FREQUENCY_UNIT:
            raise ValueError(f"Frequency surface must use {FREQUENCY_UNIT}.")
        if response.get("target") != FREQUENCY_SQUARED_TARGET:
            raise ValueError("Frequency surface must target frequency_squared_mhz2.")
        if coordinates.get("radius_coordinate") != RADIUS_COORDINATE:
            raise ValueError("Frequency surface radius_coordinate must be physical_radius.")
        if coordinates.get("radius_unit") != RADIUS_UNIT:
            raise ValueError("Frequency surface radius_unit must be um.")
        if polynomial.get("coefficient_order") != COEFFICIENT_ORDER:
            raise ValueError("Frequency surface coefficient_order must be u_major_v_minor.")
        try:
            ka_degree = int(polynomial["ka_degree"])
            radius_degree = int(polynomial["radius_degree"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Frequency surface polynomial degrees are invalid.") from error
        return cls(
            agent=payload.get("agent"),
            ka_bounds_dpd=_bounds(coordinates.get("ka_bounds_dpd"), "coordinates.ka_bounds_dpd", lower_may_be_zero=True),
            radius_bounds_um=_bounds(coordinates.get("radius_bounds_um"), "coordinates.radius_bounds_um", lower_may_be_zero=False),
            coefficients=_coefficient_array(
                polynomial.get("coefficients"),
                ka_degree=ka_degree,
                radius_degree=radius_degree,
            ),
            ka_degree=ka_degree,
            radius_degree=radius_degree,
            conditions=_optional_mapping(payload.get("conditions"), "conditions"),
            provenance=_optional_mapping(payload.get("provenance"), "provenance"),
            validation=_optional_mapping(payload.get("validation"), "validation"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DpdFrequencySurface":
        artifact_path = Path(path)
        with artifact_path.open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def write(self, path: str | Path) -> Path:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_mapping(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return artifact_path


@dataclass(frozen=True)
class RadiusHoldoutMetric:
    """One leave-one-full-radius-node-out surface validation result."""

    held_out_radius_um: float
    point_count: int
    rmse_mhz: float
    max_absolute_error_mhz: float

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "held_out_radius_um": self.held_out_radius_um,
            "point_count": self.point_count,
            "rmse_mhz": self.rmse_mhz,
            "max_absolute_error_mhz": self.max_absolute_error_mhz,
        }


@dataclass(frozen=True)
class KaHoldoutMetric:
    """One leave-one-full-ka-node-out surface validation result."""

    held_out_ka_dpd: float
    point_count: int
    rmse_mhz: float
    max_absolute_error_mhz: float

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "held_out_ka_dpd": self.held_out_ka_dpd,
            "point_count": self.point_count,
            "rmse_mhz": self.rmse_mhz,
            "max_absolute_error_mhz": self.max_absolute_error_mhz,
        }


def fit_tensor_cubic_frequency_surface(
    *,
    agent: str,
    ka_dpd: Any,
    radius_um: Any,
    frequency_mhz: Any,
    ka_bounds_dpd: tuple[float, float] = (0.0, 30000.0),
    radius_bounds_um: tuple[float, float] | None = None,
    ka_degree: int = 3,
    radius_degree: int = 3,
    conditions: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
) -> DpdFrequencySurface:
    """Fit the contract's tensor polynomial to measured physical frequencies."""

    ka, radius = _broadcast_inputs(ka_dpd, radius_um)
    frequency = np.asarray(frequency_mhz, dtype=np.float64)
    try:
        frequency = np.broadcast_to(frequency, ka.shape)
    except ValueError as error:
        raise ValueError("frequency_mhz must be broadcast-compatible with ka_dpd and radius_um.") from error
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_mhz must contain finite positive values.")
    if radius_bounds_um is None:
        radius_bounds_um = (float(np.min(radius)), float(np.max(radius)))
    candidate = DpdFrequencySurface(
        agent=agent,
        ka_bounds_dpd=ka_bounds_dpd,
        radius_bounds_um=radius_bounds_um,
        coefficients=np.zeros((int(ka_degree) + 1, int(radius_degree) + 1), dtype=np.float64),
        ka_degree=ka_degree,
        radius_degree=radius_degree,
        conditions=conditions or {},
        provenance=provenance or {},
        validation=validation or {},
    )
    u, v = candidate._normalized_coordinates(ka, radius)
    columns = [
        (u.reshape(-1) ** i) * (v.reshape(-1) ** j)
        for i in range(candidate.ka_degree + 1)
        for j in range(candidate.radius_degree + 1)
    ]
    design = np.column_stack(columns)
    if design.shape[0] < design.shape[1]:
        raise ValueError(
            "Not enough frequency measurements for the requested tensor polynomial degree."
        )
    solution, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        frequency.reshape(-1) ** 2,
        rcond=None,
    )
    if rank != design.shape[1]:
        raise ValueError("Frequency-surface design matrix is rank deficient.")
    coefficients = solution.reshape((candidate.ka_degree + 1, candidate.radius_degree + 1))
    return DpdFrequencySurface(
        agent=candidate.agent,
        ka_bounds_dpd=candidate.ka_bounds_dpd,
        radius_bounds_um=candidate.radius_bounds_um,
        coefficients=coefficients,
        ka_degree=candidate.ka_degree,
        radius_degree=candidate.radius_degree,
        conditions=candidate.conditions,
        provenance=candidate.provenance,
        validation=candidate.validation,
    )


def leave_one_radius_out_metrics(
    *,
    agent: str,
    ka_dpd: Any,
    radius_um: Any,
    frequency_mhz: Any,
    ka_bounds_dpd: tuple[float, float] = (0.0, 30000.0),
    radius_bounds_um: tuple[float, float] | None = None,
    ka_degree: int = 3,
    radius_degree: int = 3,
) -> list[RadiusHoldoutMetric]:
    """Validate interpolation by withholding every measurement at one radius."""

    ka, radius = _broadcast_inputs(ka_dpd, radius_um)
    frequency = np.asarray(frequency_mhz, dtype=np.float64)
    try:
        frequency = np.broadcast_to(frequency, ka.shape)
    except ValueError as error:
        raise ValueError("frequency_mhz must be broadcast-compatible with ka_dpd and radius_um.") from error
    flat_ka = ka.reshape(-1)
    flat_radius = radius.reshape(-1)
    flat_frequency = frequency.reshape(-1)
    unique_radii = np.unique(flat_radius)
    if unique_radii.size < 2:
        raise ValueError("Leave-one-radius-out validation requires at least two radius nodes.")
    validation_radius_bounds = radius_bounds_um or (
        float(np.min(flat_radius)),
        float(np.max(flat_radius)),
    )
    metrics: list[RadiusHoldoutMetric] = []
    for held_out in unique_radii:
        test_mask = flat_radius == held_out
        surface = fit_tensor_cubic_frequency_surface(
            agent=agent,
            ka_dpd=flat_ka[~test_mask],
            radius_um=flat_radius[~test_mask],
            frequency_mhz=flat_frequency[~test_mask],
            ka_bounds_dpd=ka_bounds_dpd,
            radius_bounds_um=validation_radius_bounds,
            ka_degree=ka_degree,
            radius_degree=radius_degree,
        )
        predicted = surface.predict_mhz(flat_ka[test_mask], flat_radius[test_mask])
        errors = predicted - flat_frequency[test_mask]
        metrics.append(
            RadiusHoldoutMetric(
                held_out_radius_um=float(held_out),
                point_count=int(np.count_nonzero(test_mask)),
                rmse_mhz=float(np.sqrt(np.mean(errors**2))),
                max_absolute_error_mhz=float(np.max(np.abs(errors))),
            )
        )
    return metrics


def leave_one_ka_out_metrics(
    *,
    agent: str,
    ka_dpd: Any,
    radius_um: Any,
    frequency_mhz: Any,
    ka_bounds_dpd: tuple[float, float] = (0.0, 30000.0),
    radius_bounds_um: tuple[float, float] | None = None,
    ka_degree: int = 3,
    radius_degree: int = 3,
) -> list[KaHoldoutMetric]:
    """Validate interpolation by withholding every measurement at one ka node."""

    ka, radius = _broadcast_inputs(ka_dpd, radius_um)
    frequency = np.asarray(frequency_mhz, dtype=np.float64)
    try:
        frequency = np.broadcast_to(frequency, ka.shape)
    except ValueError as error:
        raise ValueError("frequency_mhz must be broadcast-compatible with ka_dpd and radius_um.") from error
    flat_ka = ka.reshape(-1)
    flat_radius = radius.reshape(-1)
    flat_frequency = frequency.reshape(-1)
    unique_ka = np.unique(flat_ka)
    if unique_ka.size < 2:
        raise ValueError("Leave-one-ka-out validation requires at least two ka nodes.")
    metrics: list[KaHoldoutMetric] = []
    for held_out in unique_ka:
        test_mask = flat_ka == held_out
        surface = fit_tensor_cubic_frequency_surface(
            agent=agent,
            ka_dpd=flat_ka[~test_mask],
            radius_um=flat_radius[~test_mask],
            frequency_mhz=flat_frequency[~test_mask],
            ka_bounds_dpd=ka_bounds_dpd,
            radius_bounds_um=radius_bounds_um,
            ka_degree=ka_degree,
            radius_degree=radius_degree,
        )
        predicted = surface.predict_mhz(flat_ka[test_mask], flat_radius[test_mask])
        errors = predicted - flat_frequency[test_mask]
        metrics.append(
            KaHoldoutMetric(
                held_out_ka_dpd=float(held_out),
                point_count=int(np.count_nonzero(test_mask)),
                rmse_mhz=float(np.sqrt(np.mean(errors**2))),
                max_absolute_error_mhz=float(np.max(np.abs(errors))),
            )
        )
    return metrics


__all__ = [
    "COEFFICIENT_ORDER",
    "DpdFrequencySurface",
    "FREQUENCY_SURFACE_SCHEMA",
    "KaHoldoutMetric",
    "RadiusHoldoutMetric",
    "fit_tensor_cubic_frequency_surface",
    "leave_one_ka_out_metrics",
    "leave_one_radius_out_metrics",
]
