from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, lgamma, log, pi
from typing import Any, Mapping, Sequence

import numpy as np


class RobustLikelihoodKind(str, Enum):
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"


def _finite_float(value: float, label: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{label} must be finite; got {value}.")
    return value


def _finite_tuple(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _positive_tuple(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = _finite_tuple(values, label)
    if any(value <= 0.0 for value in result):
        raise ValueError(f"{label} values must be positive.")
    return result


def _coerce_kind(value: RobustLikelihoodKind | str) -> RobustLikelihoodKind:
    if isinstance(value, RobustLikelihoodKind):
        return value
    try:
        return RobustLikelihoodKind(str(value).strip().lower())
    except ValueError as exc:
        expected = ", ".join(item.value for item in RobustLikelihoodKind)
        raise ValueError(f"Unsupported robust likelihood kind '{value}'. Expected one of: {expected}.") from exc


def _as_covariance(value: Any, dimension: int) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (dimension, dimension):
        raise ValueError(f"Covariance shape {covariance.shape} does not match dimension {dimension}.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance values must be finite.")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("Covariance matrix must be symmetric.")
    return 0.5 * (covariance + covariance.T)


@dataclass(frozen=True)
class RobustLikelihoodConfig:
    kind: RobustLikelihoodKind | str = RobustLikelihoodKind.GAUSSIAN
    degrees_of_freedom: float | None = None

    def __post_init__(self) -> None:
        kind = _coerce_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        if kind is RobustLikelihoodKind.GAUSSIAN:
            object.__setattr__(self, "degrees_of_freedom", None)
            return
        nu = 4.0 if self.degrees_of_freedom is None else _finite_float(self.degrees_of_freedom, "degrees_of_freedom")
        if nu <= 2.0 or nu > 100.0:
            raise ValueError("Student-t degrees_of_freedom must be > 2.0 and <= 100.0.")
        object.__setattr__(self, "degrees_of_freedom", nu)


@dataclass(frozen=True)
class LikelihoodInputs:
    observed: tuple[float, ...]
    predicted: tuple[float, ...]
    standard_deviation: tuple[float, ...] | None = None
    covariance: Any | None = None
    variance_components: Mapping[str, tuple[float, ...]] | None = None

    def __post_init__(self) -> None:
        observed = _finite_tuple(self.observed, "observed")
        predicted = _finite_tuple(self.predicted, "predicted")
        if len(observed) != len(predicted):
            raise ValueError(f"Observed count {len(observed)} does not match predicted count {len(predicted)}.")
        if self.standard_deviation is not None and self.covariance is not None:
            raise ValueError("Provide either standard_deviation or covariance, not both.")
        standard_deviation = None
        if self.standard_deviation is not None:
            standard_deviation = _positive_tuple(self.standard_deviation, "standard_deviation")
            if len(standard_deviation) != len(observed):
                raise ValueError(
                    f"standard_deviation count {len(standard_deviation)} does not match observation count {len(observed)}."
                )
        covariance = None if self.covariance is None else _as_covariance(self.covariance, len(observed))
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "predicted", predicted)
        object.__setattr__(self, "standard_deviation", standard_deviation)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "variance_components", None if self.variance_components is None else dict(self.variance_components))

    @property
    def residuals(self) -> tuple[float, ...]:
        return tuple(observed - predicted for observed, predicted in zip(self.observed, self.predicted))


@dataclass(frozen=True)
class LikelihoodEvaluation:
    log_likelihood: float
    pointwise_log_likelihood: tuple[float, ...]
    standardized_residuals: tuple[float, ...]
    influence_weights: tuple[float, ...]
    covariance_mode: str


def evaluate_observation_likelihood(inputs: LikelihoodInputs, config: RobustLikelihoodConfig | None = None) -> LikelihoodEvaluation:
    robust = RobustLikelihoodConfig() if config is None else config
    if inputs.covariance is not None:
        if robust.kind is RobustLikelihoodKind.GAUSSIAN:
            return _evaluate_covariance_gaussian(inputs)
        return _evaluate_covariance_student_t(inputs, robust)
    if inputs.standard_deviation is not None:
        if robust.kind is RobustLikelihoodKind.GAUSSIAN:
            return _evaluate_diagonal_gaussian(inputs)
        return _evaluate_diagonal_student_t(inputs, robust)
    raise ValueError("Likelihood evaluation requires standard_deviation or covariance.")


def _evaluate_diagonal_gaussian(inputs: LikelihoodInputs) -> LikelihoodEvaluation:
    assert inputs.standard_deviation is not None
    pointwise: list[float] = []
    standardized: list[float] = []
    for residual, sigma in zip(inputs.residuals, inputs.standard_deviation):
        z = residual / sigma
        value = -0.5 * (z * z + log(2.0 * pi * sigma * sigma))
        if not isfinite(value):
            raise ValueError("Gaussian log likelihood must be finite.")
        pointwise.append(value)
        standardized.append(z)
    return LikelihoodEvaluation(
        log_likelihood=sum(pointwise),
        pointwise_log_likelihood=tuple(pointwise),
        standardized_residuals=tuple(standardized),
        influence_weights=(1.0,) * len(pointwise),
        covariance_mode="diagonal",
    )


def _evaluate_diagonal_student_t(inputs: LikelihoodInputs, config: RobustLikelihoodConfig) -> LikelihoodEvaluation:
    assert inputs.standard_deviation is not None
    assert config.degrees_of_freedom is not None
    nu = config.degrees_of_freedom
    pointwise: list[float] = []
    standardized: list[float] = []
    weights: list[float] = []
    constant = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(nu * pi)
    for residual, sigma in zip(inputs.residuals, inputs.standard_deviation):
        z = residual / sigma
        value = constant - log(sigma) - ((nu + 1.0) / 2.0) * log(1.0 + (z * z) / nu)
        if not isfinite(value):
            raise ValueError("Student-t log likelihood must be finite.")
        pointwise.append(value)
        standardized.append(z)
        weights.append((nu + 1.0) / (nu + z * z))
    return LikelihoodEvaluation(
        log_likelihood=sum(pointwise),
        pointwise_log_likelihood=tuple(pointwise),
        standardized_residuals=tuple(standardized),
        influence_weights=tuple(weights),
        covariance_mode="diagonal",
    )


def _cholesky(covariance: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance matrix must be positive definite.") from exc


def _evaluate_covariance_gaussian(inputs: LikelihoodInputs) -> LikelihoodEvaluation:
    assert inputs.covariance is not None
    residual = np.asarray(inputs.residuals, dtype=float)
    cholesky = _cholesky(inputs.covariance)
    whitened = np.linalg.solve(cholesky, residual)
    log_det = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    value = -0.5 * (len(residual) * log(2.0 * pi) + log_det + float(np.dot(whitened, whitened)))
    if not isfinite(value):
        raise ValueError("Gaussian covariance log likelihood must be finite.")
    return LikelihoodEvaluation(
        log_likelihood=value,
        pointwise_log_likelihood=(value,),
        standardized_residuals=tuple(float(item) for item in whitened),
        influence_weights=(1.0,) * len(residual),
        covariance_mode="full",
    )


def _evaluate_covariance_student_t(inputs: LikelihoodInputs, config: RobustLikelihoodConfig) -> LikelihoodEvaluation:
    assert inputs.covariance is not None
    assert config.degrees_of_freedom is not None
    residual = np.asarray(inputs.residuals, dtype=float)
    dimension = len(residual)
    nu = config.degrees_of_freedom
    cholesky = _cholesky(inputs.covariance)
    whitened = np.linalg.solve(cholesky, residual)
    mahalanobis = float(np.dot(whitened, whitened))
    log_det = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    value = (
        lgamma((nu + dimension) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * (dimension * log(nu * pi) + log_det)
        - ((nu + dimension) / 2.0) * log(1.0 + mahalanobis / nu)
    )
    if not isfinite(value):
        raise ValueError("Student-t covariance log likelihood must be finite.")
    weight = (nu + dimension) / (nu + mahalanobis)
    return LikelihoodEvaluation(
        log_likelihood=value,
        pointwise_log_likelihood=(value,),
        standardized_residuals=tuple(float(item) for item in whitened),
        influence_weights=(weight,) * dimension,
        covariance_mode="full",
    )
