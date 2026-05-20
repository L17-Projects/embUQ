from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, pi, sqrt
from typing import Mapping, Sequence


NumberSequence = Sequence[float]


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


def _finite_tuple(values: NumberSequence, label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(item, label) for item in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _validate_lengths(first: Sequence[float], second: Sequence[float], first_label: str, second_label: str) -> None:
    if len(first) != len(second):
        raise ValueError(
            f"{first_label} count {len(first)} does not match {second_label} count {len(second)}."
        )


@dataclass(frozen=True)
class AdditiveRelativeObservationNoiseConfig:
    additive_sigma: float = 0.0
    relative_sigma: float = 0.0
    prediction_scale_floor: float = 0.0
    minimum_total_variance: float = 0.0

    def __post_init__(self) -> None:
        additive = _nonnegative_float(self.additive_sigma, "additive_sigma")
        relative = _nonnegative_float(self.relative_sigma, "relative_sigma")
        scale_floor = _nonnegative_float(self.prediction_scale_floor, "prediction_scale_floor")
        variance_floor = _nonnegative_float(self.minimum_total_variance, "minimum_total_variance")
        if additive == 0.0 and relative == 0.0 and variance_floor == 0.0:
            raise ValueError("At least one observation noise contribution or variance floor must be positive.")
        object.__setattr__(self, "additive_sigma", additive)
        object.__setattr__(self, "relative_sigma", relative)
        object.__setattr__(self, "prediction_scale_floor", scale_floor)
        object.__setattr__(self, "minimum_total_variance", variance_floor)

    @classmethod
    def legacy_equivalent(cls, sigma: float) -> "AdditiveRelativeObservationNoiseConfig":
        return cls(relative_sigma=sigma)


@dataclass(frozen=True)
class ObservationNoiseResult:
    predictions: tuple[float, ...]
    additive_variance: tuple[float, ...]
    relative_variance: tuple[float, ...]
    floor_variance: tuple[float, ...]
    total_variance: tuple[float, ...]
    standard_deviation: tuple[float, ...]

    @property
    def variance_components(self) -> Mapping[str, tuple[float, ...]]:
        return {
            "additive": self.additive_variance,
            "relative": self.relative_variance,
            "floor": self.floor_variance,
            "total": self.total_variance,
        }


@dataclass(frozen=True)
class ObservationNoiseLikelihood:
    observations: tuple[float, ...]
    predictions: tuple[float, ...]
    residuals: tuple[float, ...]
    noise: ObservationNoiseResult
    pointwise_log_likelihood: tuple[float, ...]
    log_likelihood: float


def build_additive_relative_observation_noise(
    predictions: NumberSequence,
    config: AdditiveRelativeObservationNoiseConfig,
) -> ObservationNoiseResult:
    prediction_values = _finite_tuple(predictions, "predictions")
    additive_variance: list[float] = []
    relative_variance: list[float] = []
    floor_variance: list[float] = []
    total_variance: list[float] = []
    standard_deviation: list[float] = []

    additive = config.additive_sigma * config.additive_sigma
    if not isfinite(additive):
        raise ValueError("additive variance must be finite.")
    for prediction in prediction_values:
        scale = max(abs(prediction), config.prediction_scale_floor)
        relative = (config.relative_sigma * scale) ** 2
        raw_total = additive + relative
        if not isfinite(relative) or not isfinite(raw_total):
            raise ValueError("observation variance components must be finite.")
        total = max(raw_total, config.minimum_total_variance)
        floor = total - raw_total
        if not isfinite(total) or not isfinite(floor):
            raise ValueError("total observation variance must be finite.")
        additive_variance.append(additive)
        relative_variance.append(relative)
        floor_variance.append(floor)
        total_variance.append(total)
        standard_deviation.append(sqrt(total))

    return ObservationNoiseResult(
        predictions=prediction_values,
        additive_variance=tuple(additive_variance),
        relative_variance=tuple(relative_variance),
        floor_variance=tuple(floor_variance),
        total_variance=tuple(total_variance),
        standard_deviation=tuple(standard_deviation),
    )


def compose_additive_relative_gaussian_likelihood(
    observations: NumberSequence,
    predictions: NumberSequence,
    config: AdditiveRelativeObservationNoiseConfig,
) -> ObservationNoiseLikelihood:
    observation_values = _finite_tuple(observations, "observations")
    noise = build_additive_relative_observation_noise(predictions, config)
    _validate_lengths(observation_values, noise.predictions, "observation", "prediction")
    residuals = tuple(
        observation - prediction for observation, prediction in zip(observation_values, noise.predictions)
    )

    pointwise: list[float] = []
    for residual, variance in zip(residuals, noise.total_variance):
        if variance <= 0.0:
            raise ValueError("Total observation variance must be positive for likelihood evaluation.")
        value = -0.5 * ((residual * residual) / variance + log(2.0 * pi * variance))
        if not isfinite(value):
            raise ValueError("Gaussian observation log likelihood must be finite.")
        pointwise.append(value)

    return ObservationNoiseLikelihood(
        observations=observation_values,
        predictions=noise.predictions,
        residuals=residuals,
        noise=noise,
        pointwise_log_likelihood=tuple(pointwise),
        log_likelihood=sum(pointwise),
    )
