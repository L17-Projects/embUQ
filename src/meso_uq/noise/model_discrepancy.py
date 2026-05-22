from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from meso_uq.noise.covariance import CovarianceBuildResult


_TOTAL_COMPONENT = "model_discrepancy_total"
_FLOOR_COMPONENT = "model_discrepancy_minimum_variance"


def _finite_float(value: float, label: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be finite numeric scalars; got {value!r}.") from exc
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


def _coerce_basis(value: Sequence[Sequence[float]], point_count: int) -> np.ndarray:
    try:
        basis = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("basis values must be finite numeric scalars.") from exc
    if basis.ndim != 2:
        raise ValueError(f"basis must be a 2D [points, rank] array; got shape {basis.shape}.")
    if basis.shape[0] != point_count:
        raise ValueError(f"basis row count {basis.shape[0]} does not match prediction count {point_count}.")
    if basis.shape[1] < 1:
        raise ValueError("basis must contain at least one column.")
    if not np.all(np.isfinite(basis)):
        raise ValueError("basis values must be finite.")
    return basis


def _ensure_finite_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} values must be finite after covariance assembly.")
    return matrix


def _coerce_coefficient_covariance(value: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    try:
        covariance = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("coefficient_covariance values must be finite numeric scalars.") from exc
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"coefficient_covariance must be square; got shape {covariance.shape}.")
    if covariance.shape[0] < 1:
        raise ValueError("coefficient_covariance must contain at least one coefficient.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("coefficient_covariance values must be finite.")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise ValueError("coefficient_covariance must be symmetric.")
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(0.5 * covariance + 0.5 * covariance.T)))
    if min_eigenvalue < -1e-10:
        raise ValueError(f"coefficient_covariance must be positive semidefinite; min eigenvalue {min_eigenvalue}.")
    return tuple(tuple(float(item) for item in row) for row in covariance.tolist())


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


def _cholesky_with_optional_jitter(covariance: np.ndarray, jitter: float, max_jitter: float) -> tuple[np.ndarray | None, float]:
    if np.allclose(covariance, 0.0):
        return None, 0.0
    symmetric = 0.5 * covariance + 0.5 * covariance.T
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
class LowRankDiscrepancyConfig:
    enabled: bool = False
    coefficient_scale: float = 1.0
    coefficient_scales: tuple[float, ...] | None = None
    coefficient_covariance: tuple[tuple[float, ...], ...] | None = None
    shrinkage_strength: float = 1.0
    minimum_variance: float = 0.0
    jitter: float = 0.0
    max_jitter: float = 0.0

    def __post_init__(self) -> None:
        enabled = bool(self.enabled)
        coefficient_scale = _nonnegative_float(self.coefficient_scale, "coefficient_scale")
        coefficient_scales = None
        if self.coefficient_scales is not None:
            coefficient_scales = tuple(_nonnegative_float(value, "coefficient_scales") for value in self.coefficient_scales)
            if not coefficient_scales:
                raise ValueError("coefficient_scales must contain at least one value when provided.")
        coefficient_covariance = None
        if self.coefficient_covariance is not None:
            coefficient_covariance = _coerce_coefficient_covariance(self.coefficient_covariance)
        shrinkage_strength = _nonnegative_float(self.shrinkage_strength, "shrinkage_strength")
        minimum_variance = _nonnegative_float(self.minimum_variance, "minimum_variance")
        jitter = _nonnegative_float(self.jitter, "jitter")
        max_jitter = _nonnegative_float(self.max_jitter, "max_jitter")
        if max_jitter < jitter:
            raise ValueError("max_jitter must be >= jitter.")
        if enabled:
            if coefficient_covariance is None:
                if coefficient_scales is None and coefficient_scale <= 0.0:
                    raise ValueError("enabled low-rank discrepancy requires a positive coefficient_scale.")
                if coefficient_scales is not None and any(value <= 0.0 for value in coefficient_scales):
                    raise ValueError("enabled low-rank discrepancy requires positive coefficient_scales.")
                if shrinkage_strength <= 0.0:
                    raise ValueError("enabled low-rank discrepancy requires positive shrinkage_strength.")
            elif np.allclose(np.asarray(coefficient_covariance, dtype=float), 0.0):
                raise ValueError("enabled low-rank discrepancy requires nonzero coefficient_covariance.")
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "coefficient_scale", coefficient_scale)
        object.__setattr__(self, "coefficient_scales", coefficient_scales)
        object.__setattr__(self, "coefficient_covariance", coefficient_covariance)
        object.__setattr__(self, "shrinkage_strength", shrinkage_strength)
        object.__setattr__(self, "minimum_variance", minimum_variance)
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "max_jitter", max_jitter)


@dataclass(frozen=True)
class LowRankDiscrepancyInputs:
    predictions: tuple[float, ...]
    basis: tuple[tuple[float, ...], ...]
    basis_names: tuple[str, ...] | None = None
    curve_id: str = "curve"

    def __post_init__(self) -> None:
        predictions = _finite_vector(self.predictions, "predictions")
        basis = _coerce_basis(self.basis, len(predictions))
        basis_names = None
        if self.basis_names is not None:
            basis_names = tuple(str(name).strip() for name in self.basis_names)
            if any(not name for name in basis_names):
                raise ValueError("basis_names must be non-empty.")
            if len(set(basis_names)) != len(basis_names):
                raise ValueError("basis_names must be unique.")
            if len(basis_names) != basis.shape[1]:
                raise ValueError(f"basis_names count {len(basis_names)} does not match basis rank {basis.shape[1]}.")
        curve_id = str(self.curve_id).strip()
        if not curve_id:
            raise ValueError("curve_id must be non-empty.")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "basis", tuple(tuple(float(value) for value in row) for row in basis.tolist()))
        object.__setattr__(self, "basis_names", basis_names)
        object.__setattr__(self, "curve_id", curve_id)


@dataclass(frozen=True)
class LowRankDiscrepancyResult:
    covariance: CovarianceBuildResult
    covariance_components: Mapping[str, np.ndarray]
    variance_components: Mapping[str, tuple[float, ...]]
    coefficient_prior_variance: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class DiscrepancyFitResult:
    coefficients: tuple[float, ...]
    fitted_discrepancy: tuple[float, ...]
    residual_after_fit: tuple[float, ...]
    residual_rmse: float
    regularization: float
    protected_parameter_drift: Mapping[str, float]


def build_polynomial_discrepancy_basis(
    grid: Sequence[float],
    *,
    degree: int,
    include_intercept: bool = False,
) -> tuple[tuple[float, ...], ...]:
    values = np.asarray(_finite_vector(grid, "grid"), dtype=float)
    if degree < 1:
        raise ValueError("degree must be >= 1.")
    centered = values - float(np.mean(values))
    scale = float(np.max(np.abs(centered)))
    if scale > 0.0:
        centered = centered / scale
    powers = range(0 if include_intercept else 1, degree + 1)
    columns = []
    for power in powers:
        column = centered**power
        if power != 0:
            column = column - float(np.mean(column))
        columns.append(column)
    basis = np.column_stack(columns)
    return tuple(tuple(float(value) for value in row) for row in basis.tolist())


def build_low_rank_model_discrepancy_covariance(
    inputs: LowRankDiscrepancyInputs,
    config: LowRankDiscrepancyConfig,
) -> LowRankDiscrepancyResult:
    basis = np.asarray(inputs.basis, dtype=float)
    point_count, rank = basis.shape
    basis_names = inputs.basis_names or tuple(f"basis_{index}" for index in range(rank))
    if config.coefficient_scales is not None and len(config.coefficient_scales) != rank:
        raise ValueError(f"coefficient_scales count {len(config.coefficient_scales)} does not match basis rank {rank}.")
    if config.coefficient_covariance is not None and len(config.coefficient_covariance) != rank:
        raise ValueError(f"coefficient_covariance rank {len(config.coefficient_covariance)} does not match basis rank {rank}.")

    zero = np.zeros((point_count, point_count), dtype=float)
    if not config.enabled:
        components = {f"model_discrepancy:{name}": zero.copy() for name in basis_names}
        components[_FLOOR_COMPONENT] = zero.copy()
        return _build_result(inputs, config, components, (0.0,) * rank, coefficient_covariance_rank=0, enabled=False)

    if np.linalg.matrix_rank(basis) < 1:
        raise ValueError("enabled low-rank discrepancy requires basis rank > 0.")
    if config.coefficient_covariance is not None:
        coefficient_covariance = np.asarray(config.coefficient_covariance, dtype=float)
    else:
        if config.coefficient_scales is None:
            raw_scales = np.full(rank, config.coefficient_scale, dtype=float)
        else:
            raw_scales = np.asarray(config.coefficient_scales, dtype=float)
        shrinkage = 1.0 + config.shrinkage_strength * (np.arange(rank, dtype=float) + 1.0) ** 2
        with np.errstate(over="ignore", invalid="ignore"):
            coefficient_variance = (raw_scales * raw_scales) / shrinkage
        if not np.all(np.isfinite(coefficient_variance)):
            raise ValueError("coefficient prior variance values must be finite after shrinkage.")
        coefficient_covariance = np.diag(coefficient_variance)
    coefficient_variance = np.diag(coefficient_covariance)

    components: dict[str, np.ndarray] = {}
    for index, (name, variance) in enumerate(zip(basis_names, coefficient_variance)):
        column = basis[:, index]
        with np.errstate(over="ignore", invalid="ignore"):
            component = float(variance) * np.outer(column, column)
        components[f"model_discrepancy:{name}"] = _ensure_finite_matrix(component, f"model_discrepancy:{name}")
    for left in range(rank):
        for right in range(left + 1, rank):
            covariance = float(coefficient_covariance[left, right])
            if np.isclose(covariance, 0.0):
                continue
            left_name = basis_names[left]
            right_name = basis_names[right]
            with np.errstate(over="ignore", invalid="ignore"):
                cross = covariance * (np.outer(basis[:, left], basis[:, right]) + np.outer(basis[:, right], basis[:, left]))
            components[f"model_discrepancy_cross:{left_name}:{right_name}"] = _ensure_finite_matrix(
                cross,
                f"model_discrepancy_cross:{left_name}:{right_name}",
            )
    floor = np.diag(np.full(point_count, config.minimum_variance, dtype=float))
    components[_FLOOR_COMPONENT] = _ensure_finite_matrix(floor, _FLOOR_COMPONENT)
    return _build_result(
        inputs,
        config,
        components,
        tuple(float(value) for value in coefficient_variance),
        coefficient_covariance_rank=int(np.linalg.matrix_rank(coefficient_covariance)),
        enabled=True,
    )


def fit_low_rank_discrepancy_coefficients(
    residuals: Sequence[float],
    basis: Sequence[Sequence[float]],
    *,
    regularization: float,
    protected_sensitivities: Mapping[str, Sequence[float]] | None = None,
) -> DiscrepancyFitResult:
    residual = np.asarray(_finite_vector(residuals, "residuals"), dtype=float)
    design = _coerce_basis(basis, residual.size)
    regularization = _nonnegative_float(regularization, "regularization")
    normal = design.T @ design + regularization * np.eye(design.shape[1], dtype=float)
    rhs = design.T @ residual
    coefficients = np.linalg.solve(normal, rhs)
    fitted = design @ coefficients
    remaining = residual - fitted
    drift: dict[str, float] = {}
    if protected_sensitivities:
        for name, sensitivity_values in protected_sensitivities.items():
            sensitivity = np.asarray(_finite_vector(sensitivity_values, f"protected_sensitivities[{name}]"), dtype=float)
            if sensitivity.shape != residual.shape:
                raise ValueError(f"protected sensitivity '{name}' length {sensitivity.size} does not match residual count {residual.size}.")
            denominator = float(np.dot(sensitivity, sensitivity))
            if denominator <= 0.0:
                raise ValueError(f"protected sensitivity '{name}' must not be all zero.")
            drift[str(name)] = float(np.dot(sensitivity, fitted) / denominator)
    return DiscrepancyFitResult(
        coefficients=tuple(float(value) for value in coefficients),
        fitted_discrepancy=tuple(float(value) for value in fitted),
        residual_after_fit=tuple(float(value) for value in remaining),
        residual_rmse=float(np.sqrt(np.mean(remaining * remaining))),
        regularization=float(regularization),
        protected_parameter_drift=drift,
    )


def _build_result(
    inputs: LowRankDiscrepancyInputs,
    config: LowRankDiscrepancyConfig,
    components: Mapping[str, np.ndarray],
    coefficient_prior_variance: tuple[float, ...],
    *,
    coefficient_covariance_rank: int,
    enabled: bool,
) -> LowRankDiscrepancyResult:
    finite_components = {
        name: _ensure_finite_matrix(np.asarray(component, dtype=float), name)
        for name, component in components.items()
    }
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = sum(
            finite_components.values(),
            np.zeros_like(next(iter(finite_components.values()))),
        )
    _ensure_finite_matrix(covariance, _TOTAL_COMPONENT)
    cholesky, jitter_added = _cholesky_with_optional_jitter(covariance, config.jitter, config.max_jitter)
    if cholesky is not None and jitter_added > 0.0:
        covariance = covariance + jitter_added * np.eye(covariance.shape[0], dtype=float)
    correlation = _correlation_from_covariance(covariance)
    eigenvalues = np.linalg.eigvalsh(0.5 * covariance + 0.5 * covariance.T)
    condition_number = np.linalg.cond(covariance) if covariance.size and not np.allclose(covariance, 0.0) else float("nan")
    active = enabled and not np.allclose(covariance, 0.0)
    covariance_components = {name: np.asarray(component, dtype=float) for name, component in finite_components.items()}
    covariance_components[_TOTAL_COMPONENT] = covariance
    variance_components = {name: tuple(float(value) for value in np.diag(component)) for name, component in covariance_components.items()}
    standard_deviation = tuple(float(sqrt(max(value, 0.0))) for value in np.diag(covariance))
    summary = {
        "schema_version": 1,
        "enabled": bool(enabled),
        "active": bool(active),
        "curve_id": inputs.curve_id,
        "basis_rank": int(np.asarray(inputs.basis, dtype=float).shape[1]),
        "matrix_rank": int(np.linalg.matrix_rank(covariance)) if covariance.size else 0,
        "component_names": list(covariance_components.keys()),
        "coefficient_prior_variance": list(coefficient_prior_variance),
        "coefficient_covariance_rank": int(coefficient_covariance_rank),
        "shrinkage_strength": config.shrinkage_strength,
        "shape": list(covariance.shape),
        "diagonal_min": _finite_or_none(np.min(np.diag(covariance))),
        "diagonal_max": _finite_or_none(np.max(np.diag(covariance))),
        "correlation_min": _finite_or_none(np.min(correlation)),
        "correlation_max": _finite_or_none(np.max(correlation)),
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
    return LowRankDiscrepancyResult(
        covariance=covariance_result,
        covariance_components=covariance_components,
        variance_components=variance_components,
        coefficient_prior_variance=coefficient_prior_variance,
        standard_deviation=standard_deviation,
        summary=summary,
    )


__all__ = [
    "DiscrepancyFitResult",
    "LowRankDiscrepancyConfig",
    "LowRankDiscrepancyInputs",
    "LowRankDiscrepancyResult",
    "build_low_rank_model_discrepancy_covariance",
    "build_polynomial_discrepancy_basis",
    "fit_low_rank_discrepancy_coefficients",
]
