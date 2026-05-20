from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from meso_uq.noise.covariance import CovarianceBuildResult


class SurrogateCovarianceKind(str, Enum):
    DIAGONAL = "diagonal"
    FULL = "full"
    LOW_RANK = "low_rank"


_TOTAL_COMPONENT = "surrogate_covariance_total"
_DIAGONAL_COMPONENT = "surrogate_predictive_diagonal"
_FULL_COMPONENT = "surrogate_predictive_full"
_LOW_RANK_COMPONENT = "surrogate_predictive_low_rank"


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


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _nonnegative_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_nonnegative_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _coerce_kind(value: SurrogateCovarianceKind | str) -> SurrogateCovarianceKind:
    if isinstance(value, SurrogateCovarianceKind):
        return value
    normalized = str(value).strip().lower()
    try:
        return SurrogateCovarianceKind(normalized)
    except ValueError as exc:
        expected = ", ".join(kind.value for kind in SurrogateCovarianceKind)
        raise ValueError(f"Unsupported surrogate covariance kind '{value}'. Expected one of: {expected}.") from exc


def _coerce_covariance(value: Sequence[Sequence[float]], size: int, label: str) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (size, size):
        raise ValueError(f"{label} shape {covariance.shape} does not match prediction count {size}.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{label} values must be finite.")
    if not np.allclose(covariance, covariance.T):
        raise ValueError(f"{label} must be symmetric.")
    symmetric = 0.5 * (covariance + covariance.T)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
    if min_eigenvalue < -1e-12:
        raise ValueError(f"{label} must be positive semidefinite.")
    return symmetric


def _coerce_low_rank_factors(value: Sequence[Sequence[float]], size: int) -> np.ndarray:
    factors = np.asarray(value, dtype=float)
    if factors.ndim != 2:
        raise ValueError(f"low_rank_factors must be a 2D [points, rank] array; got shape {factors.shape}.")
    if factors.shape[0] != size:
        raise ValueError(f"low_rank_factors row count {factors.shape[0]} does not match prediction count {size}.")
    if factors.shape[1] < 1:
        raise ValueError("low_rank_factors must contain at least one rank column.")
    if not np.all(np.isfinite(factors)):
        raise ValueError("low_rank_factors values must be finite.")
    return factors


def _ensure_finite_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} values must be finite after covariance assembly.")
    return matrix


def _finite_diagonal_covariance(std: np.ndarray, label: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.diag(std * std)
    return _ensure_finite_matrix(covariance, label)


def _finite_low_rank_covariance(factors: np.ndarray, label: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = factors @ factors.T
    return _ensure_finite_matrix(covariance, label)


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
class SurrogateCovarianceConfig:
    enabled: bool = True
    kind: SurrogateCovarianceKind | str = SurrogateCovarianceKind.DIAGONAL
    jitter: float = 0.0
    max_jitter: float = 0.0

    def __post_init__(self) -> None:
        jitter = _nonnegative_float(self.jitter, "jitter")
        max_jitter = _nonnegative_float(self.max_jitter, "max_jitter")
        if max_jitter < jitter:
            raise ValueError("max_jitter must be >= jitter.")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "kind", _coerce_kind(self.kind))
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "max_jitter", max_jitter)


@dataclass(frozen=True)
class SurrogateCovarianceInputs:
    predictions: tuple[float, ...]
    predictive_standard_deviation: tuple[float, ...] | None = None
    predictive_covariance: tuple[tuple[float, ...], ...] | None = None
    low_rank_factors: tuple[tuple[float, ...], ...] | None = None
    curve_grid: tuple[float, ...] | None = None
    curve_id: str = "curve"

    def __post_init__(self) -> None:
        predictions = _finite_vector(self.predictions, "predictions")
        size = len(predictions)
        predictive_standard_deviation = None
        if self.predictive_standard_deviation is not None:
            predictive_standard_deviation = _nonnegative_vector(
                self.predictive_standard_deviation,
                "predictive_standard_deviation",
            )
            if len(predictive_standard_deviation) != size:
                raise ValueError(
                    "predictive_standard_deviation count "
                    f"{len(predictive_standard_deviation)} does not match prediction count {size}."
                )
        curve_grid = None
        if self.curve_grid is not None:
            curve_grid = _finite_vector(self.curve_grid, "curve_grid")
            if len(curve_grid) != size:
                raise ValueError(f"curve_grid count {len(curve_grid)} does not match prediction count {size}.")
        curve_id = str(self.curve_id).strip()
        if not curve_id:
            raise ValueError("curve_id must be non-empty.")
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "predictive_standard_deviation", predictive_standard_deviation)
        object.__setattr__(self, "curve_grid", curve_grid)
        object.__setattr__(self, "curve_id", curve_id)


@dataclass(frozen=True)
class SurrogateCovarianceResult:
    covariance: CovarianceBuildResult
    covariance_components: Mapping[str, np.ndarray]
    variance_components: Mapping[str, tuple[float, ...]]
    standard_deviation: tuple[float, ...]
    summary: Mapping[str, Any]


def build_surrogate_covariance(
    inputs: SurrogateCovarianceInputs,
    config: SurrogateCovarianceConfig,
) -> SurrogateCovarianceResult:
    size = len(inputs.predictions)
    if not config.enabled:
        zero = np.zeros((size, size), dtype=float)
        return _build_result(inputs, config, {_kind_component_name(config.kind): zero}, enabled=False)

    components: dict[str, np.ndarray] = {}
    if config.kind is SurrogateCovarianceKind.DIAGONAL:
        if inputs.predictive_standard_deviation is None:
            raise ValueError("diagonal surrogate covariance requires predictive_standard_deviation.")
        std = np.asarray(inputs.predictive_standard_deviation, dtype=float)
        components[_DIAGONAL_COMPONENT] = _finite_diagonal_covariance(std, _DIAGONAL_COMPONENT)
    elif config.kind is SurrogateCovarianceKind.FULL:
        if inputs.predictive_covariance is None:
            raise ValueError("full surrogate covariance requires predictive_covariance.")
        components[_FULL_COMPONENT] = _coerce_covariance(inputs.predictive_covariance, size, "predictive_covariance")
    elif config.kind is SurrogateCovarianceKind.LOW_RANK:
        if inputs.low_rank_factors is None:
            raise ValueError("low_rank surrogate covariance requires low_rank_factors.")
        factors = _coerce_low_rank_factors(inputs.low_rank_factors, size)
        components[_LOW_RANK_COMPONENT] = _finite_low_rank_covariance(factors, _LOW_RANK_COMPONENT)
        if inputs.predictive_standard_deviation is not None:
            std = np.asarray(inputs.predictive_standard_deviation, dtype=float)
            components[_DIAGONAL_COMPONENT] = _finite_diagonal_covariance(std, _DIAGONAL_COMPONENT)
    else:  # pragma: no cover - enum exhaustiveness guard
        raise AssertionError(f"Unhandled surrogate covariance kind: {config.kind}")

    return _build_result(inputs, config, components, enabled=True)


def _kind_component_name(kind: SurrogateCovarianceKind) -> str:
    if kind is SurrogateCovarianceKind.DIAGONAL:
        return _DIAGONAL_COMPONENT
    if kind is SurrogateCovarianceKind.FULL:
        return _FULL_COMPONENT
    return _LOW_RANK_COMPONENT


def _build_result(
    inputs: SurrogateCovarianceInputs,
    config: SurrogateCovarianceConfig,
    components: Mapping[str, np.ndarray],
    *,
    enabled: bool,
) -> SurrogateCovarianceResult:
    finite_components = {
        name: _ensure_finite_matrix(np.asarray(component, dtype=float), name)
        for name, component in components.items()
    }
    covariance = sum(
        finite_components.values(),
        np.zeros_like(next(iter(finite_components.values()))),
    )
    _ensure_finite_matrix(covariance, _TOTAL_COMPONENT)
    cholesky, jitter_added = _cholesky_with_optional_jitter(covariance, config.jitter, config.max_jitter)
    if cholesky is not None and jitter_added > 0.0:
        covariance = covariance + jitter_added * np.eye(covariance.shape[0], dtype=float)
    correlation = _correlation_from_covariance(covariance)
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    condition_number = np.linalg.cond(covariance) if covariance.size and not np.allclose(covariance, 0.0) else float("nan")
    active = enabled and not np.allclose(covariance, 0.0)
    covariance_components = {name: np.asarray(component, dtype=float) for name, component in finite_components.items()}
    covariance_components[_TOTAL_COMPONENT] = covariance
    variance_components = {name: tuple(float(value) for value in np.diag(component)) for name, component in covariance_components.items()}
    standard_deviation = tuple(float(sqrt(max(value, 0.0))) for value in np.diag(covariance))
    low_rank_factor_rank = None
    if config.kind is SurrogateCovarianceKind.LOW_RANK and inputs.low_rank_factors is not None:
        low_rank_factor_rank = int(np.asarray(inputs.low_rank_factors, dtype=float).shape[1])
    summary = {
        "schema_version": 1,
        "enabled": bool(enabled),
        "active": bool(active),
        "curve_id": inputs.curve_id,
        "kind": config.kind.value,
        "component_names": list(covariance_components.keys()),
        "shape": list(covariance.shape),
        "prediction_count": len(inputs.predictions),
        "grid_count": None if inputs.curve_grid is None else len(inputs.curve_grid),
        "matrix_rank": int(np.linalg.matrix_rank(covariance)) if covariance.size else 0,
        "low_rank_factor_rank": low_rank_factor_rank,
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
    return SurrogateCovarianceResult(
        covariance=covariance_result,
        covariance_components=covariance_components,
        variance_components=variance_components,
        standard_deviation=standard_deviation,
        summary=summary,
    )


__all__ = [
    "SurrogateCovarianceConfig",
    "SurrogateCovarianceInputs",
    "SurrogateCovarianceKind",
    "SurrogateCovarianceResult",
    "build_surrogate_covariance",
]
