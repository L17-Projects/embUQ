from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log, pi
from typing import Any, Mapping, Sequence

import numpy as np


class CorrelationKernelKind(str, Enum):
    SQUARED_EXPONENTIAL = "squared_exponential"


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


def _coerce_kernel(value: CorrelationKernelKind | str) -> CorrelationKernelKind:
    if isinstance(value, CorrelationKernelKind):
        return value
    try:
        return CorrelationKernelKind(str(value).strip().lower())
    except ValueError as exc:
        expected = ", ".join(item.value for item in CorrelationKernelKind)
        raise ValueError(f"Unsupported correlation kernel '{value}'. Expected one of: {expected}.") from exc


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if isfinite(value) else None


@dataclass(frozen=True)
class CorrelatedCurveNoiseConfig:
    enabled: bool = True
    kernel: CorrelationKernelKind | str = CorrelationKernelKind.SQUARED_EXPONENTIAL
    amplitude: float = 0.0
    length_scale: float = 1.0
    jitter: float = 1e-10
    max_jitter: float = 1e-6

    def __post_init__(self) -> None:
        amplitude = _nonnegative_float(self.amplitude, "amplitude")
        length_scale = _positive_float(self.length_scale, "length_scale")
        jitter = _nonnegative_float(self.jitter, "jitter")
        max_jitter = _nonnegative_float(self.max_jitter, "max_jitter")
        if max_jitter < jitter:
            raise ValueError("max_jitter must be >= jitter.")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "kernel", _coerce_kernel(self.kernel))
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "length_scale", length_scale)
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "max_jitter", max_jitter)


@dataclass(frozen=True)
class CurveGrid:
    points: tuple[float, ...]
    curve_id: str = "curve"

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _finite_vector(self.points, "curve grid points"))
        curve_id = str(self.curve_id).strip()
        if not curve_id:
            raise ValueError("curve_id must be non-empty.")
        object.__setattr__(self, "curve_id", curve_id)


@dataclass(frozen=True)
class CovarianceBuildResult:
    covariance: np.ndarray
    correlation: np.ndarray
    cholesky: np.ndarray | None
    jitter_added: float
    summary: Mapping[str, Any]


def _squared_exponential(points: tuple[float, ...], length_scale: float) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    distance = values[:, None] - values[None, :]
    return np.exp(-0.5 * (distance / length_scale) ** 2)


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


def _summary(
    covariance: np.ndarray,
    correlation: np.ndarray,
    *,
    jitter_added: float,
    cholesky_success: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    condition_number = np.linalg.cond(covariance) if covariance.size else float("nan")
    result: dict[str, Any] = {
        "schema_version": 1,
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
    if extra:
        result.update(dict(extra))
    return result


def _cholesky_with_jitter(covariance: np.ndarray, jitter: float, max_jitter: float) -> tuple[np.ndarray, np.ndarray, float]:
    symmetric = 0.5 * (covariance + covariance.T)
    try:
        return symmetric, np.linalg.cholesky(symmetric), 0.0
    except np.linalg.LinAlgError:
        pass

    if max_jitter <= 0.0:
        raise ValueError("Covariance matrix is not positive definite and no jitter is allowed.")
    candidate = jitter if jitter > 0.0 else min(max_jitter, 1e-12)
    eye = np.eye(symmetric.shape[0], dtype=float)
    while candidate <= max_jitter * (1.0 + 1e-12):
        stabilized = symmetric + candidate * eye
        try:
            return stabilized, np.linalg.cholesky(stabilized), candidate
        except np.linalg.LinAlgError:
            candidate *= 10.0
    raise ValueError("Covariance matrix is not positive definite within max_jitter.")


def build_correlated_curve_covariance(
    grid: CurveGrid,
    config: CorrelatedCurveNoiseConfig,
) -> CovarianceBuildResult:
    return build_block_correlated_curve_covariance((grid,), config)


def build_block_correlated_curve_covariance(
    grids: Sequence[CurveGrid],
    config: CorrelatedCurveNoiseConfig,
) -> CovarianceBuildResult:
    grid_values = tuple(grids)
    if not grid_values:
        raise ValueError("At least one curve grid is required.")
    total_points = sum(len(grid.points) for grid in grid_values)
    covariance = np.zeros((total_points, total_points), dtype=float)
    correlation = np.eye(total_points, dtype=float)
    curve_ids: list[str] = []

    offset = 0
    for grid in grid_values:
        curve_ids.append(grid.curve_id)
        size = len(grid.points)
        if config.enabled and config.amplitude > 0.0:
            block_correlation = _squared_exponential(grid.points, config.length_scale)
            block_covariance = (config.amplitude * config.amplitude) * block_correlation
            covariance[offset : offset + size, offset : offset + size] = block_covariance
            correlation[offset : offset + size, offset : offset + size] = block_correlation
        offset += size

    if not config.enabled or config.amplitude == 0.0:
        return CovarianceBuildResult(
            covariance=covariance,
            correlation=correlation,
            cholesky=None,
            jitter_added=0.0,
            summary=_summary(
                covariance,
                correlation,
                jitter_added=0.0,
                cholesky_success=False,
                extra={
                    "kernel": config.kernel.value,
                    "enabled": bool(config.enabled),
                    "amplitude": config.amplitude,
                    "length_scale": config.length_scale,
                    "curve_ids": curve_ids,
                },
            ),
        )

    stabilized, cholesky, jitter_added = _cholesky_with_jitter(covariance, config.jitter, config.max_jitter)
    stabilized_correlation = _correlation_from_covariance(stabilized)
    return CovarianceBuildResult(
        covariance=stabilized,
        correlation=stabilized_correlation,
        cholesky=cholesky,
        jitter_added=jitter_added,
        summary=_summary(
            stabilized,
            stabilized_correlation,
            jitter_added=jitter_added,
            cholesky_success=True,
            extra={
                "kernel": config.kernel.value,
                "enabled": bool(config.enabled),
                "amplitude": config.amplitude,
                "length_scale": config.length_scale,
                "curve_ids": curve_ids,
            },
        ),
    )


def compose_total_covariance(
    diagonal_variance: Sequence[float],
    correlated: CovarianceBuildResult,
    *,
    jitter: float = 0.0,
    max_jitter: float = 0.0,
) -> CovarianceBuildResult:
    diagonal = np.asarray(_finite_vector(diagonal_variance, "diagonal variance"), dtype=float)
    if np.any(diagonal < 0.0):
        raise ValueError("diagonal variance values must be >= 0.0.")
    covariance = np.asarray(correlated.covariance, dtype=float)
    if covariance.shape != (len(diagonal), len(diagonal)):
        raise ValueError(
            f"Correlated covariance shape {covariance.shape} does not match diagonal length {len(diagonal)}."
        )
    total = covariance + np.diag(diagonal)
    stabilized, cholesky, jitter_added = _cholesky_with_jitter(
        total,
        _nonnegative_float(jitter, "jitter"),
        _nonnegative_float(max_jitter, "max_jitter"),
    )
    correlation = _correlation_from_covariance(stabilized)
    return CovarianceBuildResult(
        covariance=stabilized,
        correlation=correlation,
        cholesky=cholesky,
        jitter_added=jitter_added,
        summary=_summary(
            stabilized,
            correlation,
            jitter_added=jitter_added,
            cholesky_success=True,
            extra={"correlated_jitter_added": correlated.jitter_added},
        ),
    )


def gaussian_log_likelihood_from_covariance(residuals: Sequence[float], covariance: CovarianceBuildResult) -> float:
    residual = np.asarray(_finite_vector(residuals, "residuals"), dtype=float)
    if covariance.covariance.shape != (len(residual), len(residual)):
        raise ValueError(
            f"Covariance shape {covariance.covariance.shape} does not match residual count {len(residual)}."
        )
    if covariance.cholesky is None:
        raise ValueError("Covariance log likelihood requires a positive-definite Cholesky factor.")
    whitened = np.linalg.solve(covariance.cholesky, residual)
    log_det = 2.0 * float(np.sum(np.log(np.diag(covariance.cholesky))))
    value = -0.5 * (len(residual) * log(2.0 * pi) + log_det + float(np.dot(whitened, whitened)))
    if not isfinite(value):
        raise ValueError("Gaussian covariance log likelihood must be finite.")
    return value
