from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from meso_uq.noise.covariance import CovarianceBuildResult


_COMPONENT_NAMES = (
    "contact_offset",
    "alignment_tilt",
    "displacement_scale_calibration",
    "force_scale_calibration",
    "minimum_variance_floor",
)


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


def _summary(
    covariance: np.ndarray,
    correlation: np.ndarray,
    *,
    enabled: bool,
    active: bool,
    curve_id: str,
    jitter_added: float,
    cholesky_success: bool,
) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    condition_number = np.linalg.cond(covariance) if covariance.size and not np.allclose(covariance, 0.0) else float("nan")
    return {
        "schema_version": 1,
        "enabled": bool(enabled),
        "active": bool(active),
        "curve_id": curve_id,
        "component_names": list(_COMPONENT_NAMES) + ["contact_alignment_total"],
        "shape": list(covariance.shape),
        "diagonal_min": _finite_or_none(np.min(np.diag(covariance))),
        "diagonal_max": _finite_or_none(np.max(np.diag(covariance))),
        "correlation_min": _finite_or_none(np.min(correlation)),
        "correlation_max": _finite_or_none(np.max(correlation)),
        "min_eigenvalue": _finite_or_none(np.min(eigenvalues)),
        "max_eigenvalue": _finite_or_none(np.max(eigenvalues)),
        "condition_number": _finite_or_none(condition_number),
        "jitter_added": float(jitter_added),
        "cholesky_success": bool(cholesky_success),
    }


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


def _outer(vector: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return np.zeros((vector.size, vector.size), dtype=float)
    return (sigma * sigma) * np.outer(vector, vector)


@dataclass(frozen=True)
class ContactAlignmentUncertaintyConfig:
    enabled: bool = True
    contact_offset_sigma: float = 0.0
    alignment_slope_sigma: float = 0.0
    displacement_scale_sigma: float = 0.0
    force_scale_sigma: float = 0.0
    minimum_variance: float = 0.0
    jitter: float = 0.0
    max_jitter: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "contact_offset_sigma": self.contact_offset_sigma,
            "alignment_slope_sigma": self.alignment_slope_sigma,
            "displacement_scale_sigma": self.displacement_scale_sigma,
            "force_scale_sigma": self.force_scale_sigma,
            "minimum_variance": self.minimum_variance,
            "jitter": self.jitter,
            "max_jitter": self.max_jitter,
        }
        coerced = {name: _nonnegative_float(value, name) for name, value in values.items()}
        if coerced["max_jitter"] < coerced["jitter"]:
            raise ValueError("max_jitter must be >= jitter.")
        object.__setattr__(self, "enabled", bool(self.enabled))
        for name, value in coerced.items():
            object.__setattr__(self, name, value)

    @property
    def active(self) -> bool:
        return self.enabled and any(
            value > 0.0
            for value in (
                self.contact_offset_sigma,
                self.alignment_slope_sigma,
                self.displacement_scale_sigma,
                self.force_scale_sigma,
                self.minimum_variance,
            )
        )


@dataclass(frozen=True)
class ContactAlignmentInputs:
    controls: tuple[float, ...]
    predictions: tuple[float, ...]
    force_sensitivity: tuple[float, ...] | None = None
    curve_id: str = "curve"

    def __post_init__(self) -> None:
        controls = _finite_vector(self.controls, "controls")
        predictions = _finite_vector(self.predictions, "predictions")
        if len(controls) != len(predictions):
            raise ValueError(f"controls count {len(controls)} does not match predictions count {len(predictions)}.")
        force_sensitivity = None
        if self.force_sensitivity is not None:
            force_sensitivity = _finite_vector(self.force_sensitivity, "force_sensitivity")
            if len(force_sensitivity) != len(controls):
                raise ValueError(
                    f"force_sensitivity count {len(force_sensitivity)} does not match controls count {len(controls)}."
                )
        curve_id = str(self.curve_id).strip()
        if not curve_id:
            raise ValueError("curve_id must be non-empty.")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "force_sensitivity", force_sensitivity)
        object.__setattr__(self, "curve_id", curve_id)


@dataclass(frozen=True)
class ContactAlignmentUncertaintyResult:
    covariance: CovarianceBuildResult
    covariance_components: Mapping[str, np.ndarray]
    variance_components: Mapping[str, tuple[float, ...]]
    standard_deviation: tuple[float, ...]
    summary: Mapping[str, Any]


def build_contact_alignment_covariance(
    inputs: ContactAlignmentInputs,
    config: ContactAlignmentUncertaintyConfig,
) -> ContactAlignmentUncertaintyResult:
    controls = np.asarray(inputs.controls, dtype=float)
    predictions = np.asarray(inputs.predictions, dtype=float)
    size = controls.size
    zero = np.zeros((size, size), dtype=float)

    if not config.enabled:
        components = {name: zero.copy() for name in _COMPONENT_NAMES}
        return _build_result(inputs.curve_id, config, components, enabled=False, active=False)

    if config.force_scale_sigma > 0.0 and inputs.force_sensitivity is None:
        raise ValueError("force_scale_sigma requires force_sensitivity.")

    centered_controls = controls - float(np.mean(controls))
    force_vector = np.zeros(size, dtype=float)
    if inputs.force_sensitivity is not None:
        force_vector = controls * np.asarray(inputs.force_sensitivity, dtype=float)

    components = {
        "contact_offset": _outer(np.ones(size, dtype=float), config.contact_offset_sigma),
        "alignment_tilt": _outer(centered_controls, config.alignment_slope_sigma),
        "displacement_scale_calibration": _outer(predictions, config.displacement_scale_sigma),
        "force_scale_calibration": _outer(force_vector, config.force_scale_sigma),
        "minimum_variance_floor": np.diag(np.full(size, config.minimum_variance, dtype=float)),
    }
    return _build_result(inputs.curve_id, config, components, enabled=True, active=config.active)


def _build_result(
    curve_id: str,
    config: ContactAlignmentUncertaintyConfig,
    components: Mapping[str, np.ndarray],
    *,
    enabled: bool,
    active: bool,
) -> ContactAlignmentUncertaintyResult:
    covariance = sum((np.asarray(component, dtype=float) for component in components.values()), np.zeros_like(next(iter(components.values()))))
    cholesky, jitter_added = _cholesky_with_optional_jitter(covariance, config.jitter, config.max_jitter)
    if cholesky is not None and jitter_added > 0.0:
        covariance = covariance + jitter_added * np.eye(covariance.shape[0], dtype=float)
    correlation = _correlation_from_covariance(covariance)
    standard_deviation = tuple(float(sqrt(max(value, 0.0))) for value in np.diag(covariance))
    covariance_components = {name: np.asarray(component, dtype=float) for name, component in components.items()}
    covariance_components["contact_alignment_total"] = covariance
    variance_components = {name: tuple(float(value) for value in np.diag(component)) for name, component in covariance_components.items()}
    summary = _summary(
        covariance,
        correlation,
        enabled=enabled,
        active=active,
        curve_id=curve_id,
        jitter_added=jitter_added,
        cholesky_success=cholesky is not None,
    )
    covariance_result = CovarianceBuildResult(
        covariance=covariance,
        correlation=correlation,
        cholesky=cholesky,
        jitter_added=jitter_added,
        summary=summary,
    )
    return ContactAlignmentUncertaintyResult(
        covariance=covariance_result,
        covariance_components=covariance_components,
        variance_components=variance_components,
        standard_deviation=standard_deviation,
        summary=summary,
    )
