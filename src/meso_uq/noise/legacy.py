from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping, Sequence


NumberSequence = Sequence[float]
NumberMatrix = Sequence[Sequence[float]]


def _as_float_tuple(values: NumberSequence) -> tuple[float, ...]:
    return tuple(float(item) for item in values)


def _as_float_rows(values: NumberMatrix) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(item) for item in row) for row in values)
    if not rows:
        return rows
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("Legacy likelihood batch rows must all have the same width.")
    return rows


def _as_sigma_tuple(values: NumberSequence | float, row_count: int) -> tuple[float, ...]:
    try:
        sigmas = tuple(float(item) for item in values)  # type: ignore[arg-type]
    except TypeError:
        sigmas = (float(values),) * row_count
    if len(sigmas) != row_count:
        raise ValueError(f"Expected {row_count} sigma values, got {len(sigmas)}.")
    return sigmas


def _same_length(left: tuple[Any, ...], right: tuple[Any, ...], label: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{label} length {len(right)} does not match reference length {len(left)}.")


def _same_row_shape(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
    label: str,
) -> None:
    _same_length(left, right, label)
    for row_index, (left_row, right_row) in enumerate(zip(left, right)):
        if len(left_row) != len(right_row):
            raise ValueError(
                f"{label} row {row_index} length {len(right_row)} does not match "
                f"reference length {len(left_row)}."
            )


def _quadrature(first: float, second: float) -> float:
    return sqrt(first * first + second * second)


def _positive_sigma(value: float, label: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{label} must be positive; got {value}.")
    return value


def _float_list(values: tuple[float, ...]) -> list[float]:
    return [float(item) for item in values]


def _float_rows(values: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


@dataclass(frozen=True)
class LegacyLikelihoodResult:
    reference_evaluations: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    variance_components: Mapping[str, tuple[float, ...]]

    def assign_to_sample(self, sample: dict[str, Any]) -> None:
        sample["Reference Evaluations"] = _float_list(self.reference_evaluations)
        sample["Standard Deviation"] = _float_list(self.standard_deviation)


@dataclass(frozen=True)
class LegacyBatchLikelihoodResult:
    reference_evaluations: tuple[tuple[float, ...], ...]
    standard_deviation: tuple[tuple[float, ...], ...]
    variance_components: Mapping[str, tuple[tuple[float, ...], ...]]

    def assign_to_sample(self, sample: dict[str, Any]) -> None:
        sample["Batch Reference Evaluations"] = _float_rows(self.reference_evaluations)
        sample["Batch Standard Deviation"] = _float_rows(self.standard_deviation)


def legacy_compression_surrogate_likelihood(
    reference_evaluations: NumberSequence,
    sigma: float,
    *,
    surrogate_standard_deviation: NumberSequence | None = None,
) -> LegacyLikelihoodResult:
    """Preserve legacy EMB compression surrogate likelihood outputs."""

    references = _as_float_tuple(reference_evaluations)
    sigma = float(sigma)
    if surrogate_standard_deviation is None:
        observation_std = tuple(sigma * value for value in references)
        surrogate_std = (0.0,) * len(references)
        total_std = observation_std
    else:
        surrogate_std = _as_float_tuple(surrogate_standard_deviation)
        _same_length(references, surrogate_std, "Surrogate standard deviation")
        observation_std = tuple(sigma * abs(value) for value in references)
        total_std = tuple(_quadrature(model, observation) for model, observation in zip(surrogate_std, observation_std))
    return LegacyLikelihoodResult(
        reference_evaluations=references,
        standard_deviation=total_std,
        variance_components={
            "observation_std": observation_std,
            "surrogate_std": surrogate_std,
        },
    )


def legacy_compression_surrogate_batch_likelihood(
    reference_evaluations: NumberMatrix,
    sigma: NumberSequence | float,
    *,
    surrogate_standard_deviation: NumberMatrix | None = None,
) -> LegacyBatchLikelihoodResult:
    references = _as_float_rows(reference_evaluations)
    sigmas = _as_sigma_tuple(sigma, len(references))
    if surrogate_standard_deviation is None:
        observation_std = tuple(tuple(row_sigma * value for value in row) for row_sigma, row in zip(sigmas, references))
        surrogate_std = tuple(tuple(0.0 for _ in row) for row in references)
        total_std = observation_std
    else:
        surrogate_std = _as_float_rows(surrogate_standard_deviation)
        _same_row_shape(references, surrogate_std, "Surrogate standard deviation")
        observation_std = tuple(tuple(row_sigma * abs(value) for value in row) for row_sigma, row in zip(sigmas, references))
        total_std = tuple(
            tuple(_quadrature(model, observation) for model, observation in zip(model_row, observation_row))
            for model_row, observation_row in zip(surrogate_std, observation_std)
        )
    return LegacyBatchLikelihoodResult(
        reference_evaluations=references,
        standard_deviation=total_std,
        variance_components={
            "observation_std": observation_std,
            "surrogate_std": surrogate_std,
        },
    )


def legacy_compression_direct_likelihood(reference_evaluations: NumberSequence, sigma: float) -> LegacyLikelihoodResult:
    references = _as_float_tuple(reference_evaluations)
    std = (float(sigma),) * len(references)
    return LegacyLikelihoodResult(
        reference_evaluations=references,
        standard_deviation=std,
        variance_components={"observation_std": std},
    )


def legacy_indentation_surrogate_likelihood(
    predicted_displacements: NumberSequence,
    sigma: float,
    *,
    d0: float,
    surrogate_standard_deviation: NumberSequence | None = None,
) -> LegacyLikelihoodResult:
    adjusted = tuple(max(0.0, value + float(d0)) for value in _as_float_tuple(predicted_displacements))
    return legacy_indentation_adjusted_likelihood(
        adjusted,
        sigma,
        surrogate_standard_deviation=surrogate_standard_deviation,
    )


def legacy_indentation_adjusted_likelihood(
    reference_evaluations: NumberSequence,
    sigma: float,
    *,
    surrogate_standard_deviation: NumberSequence | None = None,
) -> LegacyLikelihoodResult:
    references = _as_float_tuple(reference_evaluations)
    sigma = float(sigma)
    observation_std = tuple(sigma * value for value in references)
    if surrogate_standard_deviation is None:
        surrogate_std = (0.0,) * len(references)
        total_std = observation_std
    else:
        surrogate_std = _as_float_tuple(surrogate_standard_deviation)
        _same_length(references, surrogate_std, "Surrogate standard deviation")
        total_std = tuple(_quadrature(model, observation) for model, observation in zip(surrogate_std, observation_std))
    return LegacyLikelihoodResult(
        reference_evaluations=references,
        standard_deviation=total_std,
        variance_components={
            "observation_std": observation_std,
            "surrogate_std": surrogate_std,
        },
    )


def legacy_indentation_adjusted_batch_likelihood(
    reference_evaluations: NumberMatrix,
    sigma: NumberSequence | float,
    *,
    surrogate_standard_deviation: NumberMatrix | None = None,
) -> LegacyBatchLikelihoodResult:
    references = _as_float_rows(reference_evaluations)
    sigmas = _as_sigma_tuple(sigma, len(references))
    observation_std = tuple(tuple(row_sigma * value for value in row) for row_sigma, row in zip(sigmas, references))
    if surrogate_standard_deviation is None:
        surrogate_std = tuple(tuple(0.0 for _ in row) for row in references)
        total_std = observation_std
    else:
        surrogate_std = _as_float_rows(surrogate_standard_deviation)
        _same_row_shape(references, surrogate_std, "Surrogate standard deviation")
        total_std = tuple(
            tuple(_quadrature(model, observation) for model, observation in zip(model_row, observation_row))
            for model_row, observation_row in zip(surrogate_std, observation_std)
        )
    return LegacyBatchLikelihoodResult(
        reference_evaluations=references,
        standard_deviation=total_std,
        variance_components={
            "observation_std": observation_std,
            "surrogate_std": surrogate_std,
        },
    )


def legacy_indentation_direct_standard_deviation(reference_evaluation: float, sigma: float) -> float:
    return float(sigma) * float(reference_evaluation)


def legacy_multiplicative_likelihood(
    reference_evaluations: NumberSequence,
    sigma: float,
    *,
    absolute_reference: bool = True,
) -> LegacyLikelihoodResult:
    references = _as_float_tuple(reference_evaluations)
    sigma = _positive_sigma(sigma, "legacy multiplicative sigma")
    observation_std = tuple(sigma * (abs(value) if absolute_reference else value) for value in references)
    if any(value <= 0.0 for value in observation_std):
        raise ValueError("legacy multiplicative standard deviations must be positive.")
    return LegacyLikelihoodResult(
        reference_evaluations=references,
        standard_deviation=observation_std,
        variance_components={"observation_std": observation_std},
    )


__all__ = [
    "LegacyBatchLikelihoodResult",
    "LegacyLikelihoodResult",
    "legacy_compression_direct_likelihood",
    "legacy_compression_surrogate_batch_likelihood",
    "legacy_compression_surrogate_likelihood",
    "legacy_indentation_adjusted_batch_likelihood",
    "legacy_indentation_adjusted_likelihood",
    "legacy_indentation_direct_standard_deviation",
    "legacy_indentation_surrogate_likelihood",
    "legacy_multiplicative_likelihood",
]
