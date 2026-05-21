from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import log, pi
from typing import Any, Mapping, Sequence


NumberSequence = Sequence[float]


class MeasurementErrorKind(str, Enum):
    ABSOLUTE_GAUSSIAN = "absolute_gaussian"
    RELATIVE_GAUSSIAN = "relative_gaussian"


class DiscrepancyKind(str, Enum):
    ADDITIVE_GAUSSIAN = "additive_gaussian"


class SurrogateUncertaintyKind(str, Enum):
    GAUSSIAN = "gaussian"
    DETERMINISTIC = "deterministic"


class PosteriorUncertaintyKind(str, Enum):
    GAUSSIAN = "gaussian"
    NONE = "none"


def _coerce_enum(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        expected = ", ".join(item.value for item in enum_type)
        raise ValueError(f"Unsupported {field_name} '{value}'. Expected one of: {expected}.") from exc


def _as_float_tuple(values: NumberSequence) -> tuple[float, ...]:
    return tuple(float(item) for item in values)


def _positive_sigma(value: float, field_name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"'{field_name}' must be positive; got {value}.")
    return value


def _zero_or_positive_sigma(value: float, field_name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"'{field_name}' must be >= 0.0; got {value}.")
    return value


def _unit_map(values: Mapping[str, str]) -> dict[str, str]:
    return {str(key): str(item) for key, item in values.items()}


@dataclass(frozen=True)
class MeasurementErrorConfig:
    kind: MeasurementErrorKind
    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(MeasurementErrorKind, self.kind, "measurement error kind"))
        object.__setattr__(self, "sigma", _positive_sigma(self.sigma, "measurement_error.sigma"))

    def variances(self, residuals: NumberSequence, predictions: NumberSequence | None = None) -> tuple[float, ...]:
        residual_count = len(_as_float_tuple(residuals))
        if self.kind is MeasurementErrorKind.ABSOLUTE_GAUSSIAN:
            return (self.sigma * self.sigma,) * residual_count
        if predictions is None:
            raise ValueError("Relative measurement error requires predictions.")
        scaled = _as_float_tuple(predictions)
        if len(scaled) != residual_count:
            raise ValueError(f"Prediction count {len(scaled)} does not match residual count {residual_count}.")
        return tuple((self.sigma * abs(prediction)) ** 2 for prediction in scaled)


@dataclass(frozen=True)
class DiscrepancyConfig:
    kind: DiscrepancyKind
    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(DiscrepancyKind, self.kind, "discrepancy kind"))
        object.__setattr__(self, "sigma", _positive_sigma(self.sigma, "discrepancy.sigma"))

    def variances(self, residuals: NumberSequence) -> tuple[float, ...]:
        return (self.sigma * self.sigma,) * len(_as_float_tuple(residuals))


@dataclass(frozen=True)
class SurrogateErrorConfig:
    kind: SurrogateUncertaintyKind
    sigma: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(SurrogateUncertaintyKind, self.kind, "surrogate uncertainty kind"))
        if self.kind is SurrogateUncertaintyKind.DETERMINISTIC:
            object.__setattr__(self, "sigma", 0.0)
        else:
            object.__setattr__(self, "sigma", _positive_sigma(self.sigma, "surrogate_error.sigma"))

    def variances(self, residuals: NumberSequence) -> tuple[float, ...]:
        return (self.sigma * self.sigma,) * len(_as_float_tuple(residuals))


@dataclass(frozen=True)
class PosteriorUncertaintyConfig:
    kind: PosteriorUncertaintyKind
    sigma: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _coerce_enum(PosteriorUncertaintyKind, self.kind, "posterior uncertainty kind"))
        if self.kind is PosteriorUncertaintyKind.NONE:
            object.__setattr__(self, "sigma", _zero_or_positive_sigma(self.sigma, "posterior_uncertainty.sigma"))
            return
        object.__setattr__(self, "sigma", _positive_sigma(self.sigma, "posterior_uncertainty.sigma"))

    def variances(self, residuals: NumberSequence) -> tuple[float, ...]:
        if self.kind is PosteriorUncertaintyKind.NONE:
            return (0.0,) * len(_as_float_tuple(residuals))
        return (self.sigma * self.sigma,) * len(_as_float_tuple(residuals))


@dataclass(frozen=True)
class NoiseModelSupportMetadata:
    model_id: str
    family: str
    supported_observables: tuple[str, ...]
    observable_units: Mapping[str, str]
    requires_measurement_error: bool
    required_measurement_error_kinds: tuple[MeasurementErrorKind, ...]
    supports_discrepancy: bool = False
    supports_surrogate_error: bool = False
    supports_posterior_uncertainty: bool = False

    def __post_init__(self) -> None:
        observables = tuple(str(name).strip() for name in self.supported_observables)
        if not observables:
            raise ValueError(f"Model '{self.model_id}' must declare at least one supported observable.")
        if self.requires_measurement_error and not self.required_measurement_error_kinds:
            raise ValueError(f"Model '{self.model_id}' requires measurement error but declares no valid choices.")
        object.__setattr__(self, "model_id", str(self.model_id).strip())
        object.__setattr__(self, "family", str(self.family).strip())
        object.__setattr__(self, "supported_observables", observables)
        object.__setattr__(self, "observable_units", _unit_map(self.observable_units))
        for observable in observables:
            if observable not in self.observable_units:
                raise ValueError(f"Model '{self.model_id}' has no unit for observable '{observable}'.")
        object.__setattr__(
            self,
            "required_measurement_error_kinds",
            tuple(_coerce_enum(MeasurementErrorKind, item, "required measurement error kind") for item in self.required_measurement_error_kinds),
        )

    @property
    def allowed_measurement_error_values(self) -> tuple[str, ...]:
        return tuple(item.value for item in self.required_measurement_error_kinds)

    def supports_observable(self, observable: str) -> bool:
        return str(observable).strip() in self.supported_observables

    def unit_for_observable(self, observable: str) -> str:
        selected = str(observable).strip()
        if selected not in self.supported_observables:
            supported = ", ".join(self.supported_observables)
            raise ValueError(f"Model '{self.model_id}' does not support '{selected}'. Supported: {supported}.")
        return self.observable_units[selected]


@dataclass(frozen=True)
class NoiseModelConfig:
    support: NoiseModelSupportMetadata
    observable: str
    unit: str
    measurement_error: MeasurementErrorConfig | None = None
    discrepancy: DiscrepancyConfig | None = None
    surrogate_error: SurrogateErrorConfig | None = None
    posterior_uncertainty: PosteriorUncertaintyConfig | None = None

    def __post_init__(self) -> None:
        selected_observable = str(self.observable).strip()
        if not self.support.supports_observable(selected_observable):
            supported = ", ".join(self.support.supported_observables)
            raise ValueError(f"Model '{self.support.model_id}' does not support observable '{selected_observable}'. Supported: {supported}.")
        expected_unit = self.support.unit_for_observable(selected_observable)
        actual_unit = str(self.unit).strip()
        if expected_unit != actual_unit:
            raise ValueError(
                f"Unit mismatch for model '{self.support.model_id}' + observable '{selected_observable}': "
                f"expected '{expected_unit}', got '{actual_unit}'."
            )
        if self.support.requires_measurement_error and self.measurement_error is None:
            raise ValueError(
                f"Model '{self.support.model_id}' requires a measurement error choice. "
                f"Supported: {self.support.allowed_measurement_error_values}."
            )
        if not self.support.supports_discrepancy and self.discrepancy is not None:
            raise ValueError(f"Model '{self.support.model_id}' does not support model discrepancy.")
        if not self.support.supports_surrogate_error and self.surrogate_error is not None:
            raise ValueError(f"Model '{self.support.model_id}' does not support surrogate uncertainty.")
        if not self.support.supports_posterior_uncertainty and self.posterior_uncertainty is not None:
            raise ValueError(f"Model '{self.support.model_id}' does not support posterior uncertainty.")
        if (
            self.measurement_error is not None
            and self.support.required_measurement_error_kinds
            and self.measurement_error.kind not in self.support.required_measurement_error_kinds
        ):
            choices = ", ".join(self.support.allowed_measurement_error_values)
            raise ValueError(
                f"Measurement error '{self.measurement_error.kind.value}' is not supported by '{self.support.model_id}'. "
                f"Supported: {choices}."
            )

        object.__setattr__(self, "observable", selected_observable)
        object.__setattr__(self, "unit", actual_unit)


@dataclass(frozen=True)
class ToyLikelihood:
    residuals: tuple[float, ...]
    total_variance: tuple[float, ...]
    measurement_variance: tuple[float, ...]
    discrepancy_variance: tuple[float, ...]
    surrogate_variance: tuple[float, ...]
    posterior_variance: tuple[float, ...]

    @property
    def log_likelihood(self) -> float:
        total = 0.0
        for residual, variance in zip(self.residuals, self.total_variance):
            if variance <= 0.0:
                raise ValueError("Total variance must be positive for likelihood evaluation.")
            total += (residual * residual) / variance + log(2.0 * pi * variance)
        return -0.5 * total


def compose_toy_likelihood(
    residuals: NumberSequence,
    config: NoiseModelConfig,
    *,
    predictions: NumberSequence | None = None,
) -> ToyLikelihood:
    residuals_tuple = _as_float_tuple(residuals)
    if not residuals_tuple:
        raise ValueError("Residuals must contain at least one value.")

    if config.measurement_error is None:
        measurement_variance = (0.0,) * len(residuals_tuple)
    else:
        measurement_variance = config.measurement_error.variances(residuals_tuple, predictions=predictions)

    discrepancy_variance = (
        (0.0,) * len(residuals_tuple)
        if config.discrepancy is None
        else config.discrepancy.variances(residuals_tuple)
    )

    surrogate_variance = (
        (0.0,) * len(residuals_tuple)
        if config.surrogate_error is None
        else config.surrogate_error.variances(residuals_tuple)
    )

    posterior_variance = (
        (0.0,) * len(residuals_tuple)
        if config.posterior_uncertainty is None
        else config.posterior_uncertainty.variances(residuals_tuple)
    )

    total_variance = tuple(
        measurement + discrepancy + surrogate + posterior
        for measurement, discrepancy, surrogate, posterior in zip(
            measurement_variance,
            discrepancy_variance,
            surrogate_variance,
            posterior_variance,
        )
    )

    return ToyLikelihood(
        residuals=residuals_tuple,
        total_variance=total_variance,
        measurement_variance=measurement_variance,
        discrepancy_variance=discrepancy_variance,
        surrogate_variance=surrogate_variance,
        posterior_variance=posterior_variance,
    )


__all__ = [
    "DiscrepancyConfig",
    "DiscrepancyKind",
    "MeasurementErrorConfig",
    "MeasurementErrorKind",
    "NoiseModelConfig",
    "NoiseModelSupportMetadata",
    "PosteriorUncertaintyConfig",
    "PosteriorUncertaintyKind",
    "SurrogateErrorConfig",
    "SurrogateUncertaintyKind",
    "ToyLikelihood",
    "compose_toy_likelihood",
]
