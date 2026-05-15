"""Posterior parity helpers for Phase 2 backend validation."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_LOG_COLUMNS = frozenset({"logLikelihood", "logPrior", "logPosterior"})
_NON_PARAMETER_COLUMNS = frozenset({"Sample Id", "Sample ID", "Generation", "Chain", *_LOG_COLUMNS})


@dataclass(frozen=True)
class PosteriorEquivalenceThresholds:
    """Thresholds for stochastic CPU-MPI vs NativeCuda posterior comparisons."""

    min_sample_count: int = 2
    min_finite_logposterior_ratio: float = 1.0
    max_mean_abs_delta: float = 5.0e-2
    max_std_scaled_delta: float = 2.0e-1
    max_quantile_abs_delta: float = 1.0e-1
    max_logposterior_max_abs_delta: float = 1.0
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)

    def __post_init__(self) -> None:
        if self.min_sample_count < 1:
            raise ValueError("min_sample_count must be at least 1.")
        if not 0.0 <= self.min_finite_logposterior_ratio <= 1.0:
            raise ValueError("min_finite_logposterior_ratio must be between 0 and 1.")
        for name in (
            "max_mean_abs_delta",
            "max_std_scaled_delta",
            "max_quantile_abs_delta",
            "max_logposterior_max_abs_delta",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value.")
        quantiles = tuple(float(value) for value in self.quantiles)
        if not quantiles:
            raise ValueError("At least one quantile is required.")
        for value in quantiles:
            if not 0.0 <= value <= 1.0:
                raise ValueError("quantiles must be between 0 and 1.")
        object.__setattr__(self, "quantiles", quantiles)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "min_sample_count": self.min_sample_count,
            "min_finite_logposterior_ratio": self.min_finite_logposterior_ratio,
            "max_mean_abs_delta": self.max_mean_abs_delta,
            "max_std_scaled_delta": self.max_std_scaled_delta,
            "max_quantile_abs_delta": self.max_quantile_abs_delta,
            "max_logposterior_max_abs_delta": self.max_logposterior_max_abs_delta,
            "quantiles": list(self.quantiles),
        }


@dataclass(frozen=True)
class PosteriorSummary:
    backend: str
    variables: tuple[str, ...]
    sample_count: int
    finite_logposterior_ratio: float
    mean: Mapping[str, float]
    standard_deviation: Mapping[str, float]
    quantiles: Mapping[str, Mapping[str, float]]
    logposterior_max: float | None = None
    source: str | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "variables": list(self.variables),
            "sample_count": self.sample_count,
            "finite_logposterior_ratio": self.finite_logposterior_ratio,
            "mean": dict(self.mean),
            "standard_deviation": dict(self.standard_deviation),
            "quantiles": {name: dict(values) for name, values in self.quantiles.items()},
            "logposterior_max": self.logposterior_max,
            "source": self.source,
        }


@dataclass(frozen=True)
class PosteriorEquivalenceReport:
    reference_backend: str
    candidate_backend: str
    passed: bool
    mismatches: tuple[str, ...]
    thresholds: PosteriorEquivalenceThresholds
    reference: PosteriorSummary
    candidate: PosteriorSummary

    def to_manifest(self) -> dict[str, Any]:
        return {
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "passed": self.passed,
            "mismatches": list(self.mismatches),
            "thresholds": self.thresholds.to_manifest(),
            "reference": self.reference.to_manifest(),
            "candidate": self.candidate.to_manifest(),
        }


DEFAULT_POSTERIOR_EQUIVALENCE_THRESHOLDS = PosteriorEquivalenceThresholds()


def load_phase2_posterior_samples(run_dir: str | Path):
    """Load Korali posterior samples through the shared postprocess reader."""

    from meso_uq.postprocess.maps import load_korali_state, load_posterior_samples

    state_path, state = load_korali_state(run_dir)
    _validate_raw_phase2_log_evidence(state, state_path=state_path)
    return load_posterior_samples(run_dir)


def summarize_posterior_samples(
    samples: Mapping[str, Sequence[object]] | Any,
    *,
    backend: str,
    parameter_columns: Sequence[str] | None = None,
    source: str | Path | None = None,
    quantiles: Sequence[float] = DEFAULT_POSTERIOR_EQUIVALENCE_THRESHOLDS.quantiles,
) -> PosteriorSummary:
    columns = _columns_from_samples(samples)
    variables = tuple(parameter_columns) if parameter_columns is not None else _infer_parameter_columns(columns)
    if not variables:
        raise ValueError("Posterior samples do not contain any parameter columns.")

    arrays = {name: _numeric_column(columns, name) for name in variables}
    sample_counts = {values.shape[0] for values in arrays.values()}
    if len(sample_counts) != 1:
        raise ValueError(f"Posterior parameter columns have inconsistent lengths: {sorted(sample_counts)}.")
    sample_count = sample_counts.pop()
    if sample_count < 1:
        raise ValueError("Posterior samples must contain at least one row.")

    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Posterior parameter column {name!r} contains non-finite values.")

    quantile_values = tuple(float(value) for value in quantiles)
    return PosteriorSummary(
        backend=str(backend),
        variables=variables,
        sample_count=sample_count,
        finite_logposterior_ratio=_finite_logposterior_ratio(columns, expected_count=sample_count),
        mean={name: float(np.mean(values)) for name, values in arrays.items()},
        standard_deviation={
            name: float(np.std(values, ddof=1)) if sample_count > 1 else 0.0
            for name, values in arrays.items()
        },
        quantiles={
            name: {
                _quantile_key(value): float(np.quantile(values, value))
                for value in quantile_values
            }
            for name, values in arrays.items()
        },
        logposterior_max=_logposterior_max(columns, expected_count=sample_count),
        source=str(source) if source is not None else None,
    )


def compare_posterior_samples(
    reference_samples: Mapping[str, Sequence[object]] | Any,
    candidate_samples: Mapping[str, Sequence[object]] | Any,
    *,
    reference_backend: str = "cpu-mpi",
    candidate_backend: str = "native-cuda",
    parameter_columns: Sequence[str] | None = None,
    thresholds: PosteriorEquivalenceThresholds = DEFAULT_POSTERIOR_EQUIVALENCE_THRESHOLDS,
) -> PosteriorEquivalenceReport:
    reference = summarize_posterior_samples(
        reference_samples,
        backend=reference_backend,
        parameter_columns=parameter_columns,
        quantiles=thresholds.quantiles,
    )
    candidate = summarize_posterior_samples(
        candidate_samples,
        backend=candidate_backend,
        parameter_columns=parameter_columns,
        quantiles=thresholds.quantiles,
    )
    return compare_posterior_summaries(reference, candidate, thresholds=thresholds)


def compare_posterior_summaries(
    reference: PosteriorSummary,
    candidate: PosteriorSummary,
    *,
    thresholds: PosteriorEquivalenceThresholds = DEFAULT_POSTERIOR_EQUIVALENCE_THRESHOLDS,
) -> PosteriorEquivalenceReport:
    mismatches: list[str] = []
    reference_variables = set(reference.variables)
    candidate_variables = set(candidate.variables)
    if reference_variables != candidate_variables:
        mismatches.append(
            "variables differ: "
            f"missing_from_candidate={sorted(reference_variables - candidate_variables)!r}, "
            f"extra_in_candidate={sorted(candidate_variables - reference_variables)!r}"
        )

    for label, summary in (("reference", reference), ("candidate", candidate)):
        if summary.sample_count < thresholds.min_sample_count:
            mismatches.append(
                f"{label} sample_count={summary.sample_count} below "
                f"min_sample_count={thresholds.min_sample_count}"
            )
        if summary.finite_logposterior_ratio < thresholds.min_finite_logposterior_ratio:
            mismatches.append(
                f"{label} finite_logposterior_ratio={summary.finite_logposterior_ratio:.6g} below "
                f"{thresholds.min_finite_logposterior_ratio:.6g}"
            )
    if reference.sample_count != candidate.sample_count:
        mismatches.append(
            "sample_count differs: "
            f"reference={reference.sample_count} != candidate={candidate.sample_count}"
        )

    for variable in reference.variables:
        if variable not in candidate.variables:
            continue
        _compare_variable_summary(
            variable=variable,
            reference=reference,
            candidate=candidate,
            thresholds=thresholds,
            mismatches=mismatches,
        )

    if reference.logposterior_max is not None and candidate.logposterior_max is not None:
        delta = abs(reference.logposterior_max - candidate.logposterior_max)
        if delta > thresholds.max_logposterior_max_abs_delta:
            mismatches.append(
                f"logposterior_max delta {delta:.6g} exceeds "
                f"{thresholds.max_logposterior_max_abs_delta:.6g}"
            )

    return PosteriorEquivalenceReport(
        reference_backend=reference.backend,
        candidate_backend=candidate.backend,
        passed=not mismatches,
        mismatches=tuple(mismatches),
        thresholds=thresholds,
        reference=reference,
        candidate=candidate,
    )


def _compare_variable_summary(
    *,
    variable: str,
    reference: PosteriorSummary,
    candidate: PosteriorSummary,
    thresholds: PosteriorEquivalenceThresholds,
    mismatches: list[str],
) -> None:
    mean_delta = abs(reference.mean[variable] - candidate.mean[variable])
    if mean_delta > thresholds.max_mean_abs_delta:
        mismatches.append(
            f"{variable}: mean delta {mean_delta:.6g} exceeds {thresholds.max_mean_abs_delta:.6g}"
        )

    std_delta = abs(reference.standard_deviation[variable] - candidate.standard_deviation[variable])
    std_scale = max(
        abs(reference.standard_deviation[variable]),
        abs(candidate.standard_deviation[variable]),
        1.0,
    )
    std_scaled_delta = std_delta / std_scale
    if std_scaled_delta > thresholds.max_std_scaled_delta:
        mismatches.append(
            f"{variable}: std scaled delta {std_scaled_delta:.6g} exceeds "
            f"{thresholds.max_std_scaled_delta:.6g}"
        )

    for quantile, reference_value in reference.quantiles[variable].items():
        candidate_value = candidate.quantiles[variable][quantile]
        quantile_delta = abs(reference_value - candidate_value)
        if quantile_delta > thresholds.max_quantile_abs_delta:
            mismatches.append(
                f"{variable}: quantile {quantile} delta {quantile_delta:.6g} exceeds "
                f"{thresholds.max_quantile_abs_delta:.6g}"
            )


def _columns_from_samples(samples: Mapping[str, Sequence[object]] | Any) -> dict[str, Sequence[object]]:
    if isinstance(samples, Mapping):
        return {str(name): values for name, values in samples.items()}
    columns = getattr(samples, "columns", None)
    if columns is None:
        raise ValueError("Posterior samples must be a mapping or DataFrame-like object with columns.")
    return {str(name): samples[name].to_numpy() for name in columns}


def _infer_parameter_columns(columns: Mapping[str, Sequence[object]]) -> tuple[str, ...]:
    inferred: list[str] = []
    for name, values in columns.items():
        if name in _NON_PARAMETER_COLUMNS:
            continue
        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        if numeric.ndim != 1:
            continue
        inferred.append(name)
    return tuple(inferred)


def _numeric_column(columns: Mapping[str, Sequence[object]], name: str) -> np.ndarray:
    if name not in columns:
        raise ValueError(f"Posterior samples are missing parameter column {name!r}.")
    values = np.asarray(columns[name], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Posterior parameter column {name!r} must be one-dimensional.")
    return values


def _validate_raw_phase2_log_evidence(state: Mapping[str, Any], *, state_path: Path) -> None:
    results = state.get("Results", {})
    sample_db = results.get("Posterior Sample Database")
    if sample_db is None:
        return
    try:
        expected_count = len(sample_db)
    except TypeError as exc:
        raise ValueError(f"Posterior Sample Database at {state_path} must be a sequence.") from exc

    for name in (
        "Posterior Sample LogLikelihood Database",
        "Posterior Sample LogPrior Database",
    ):
        if name not in results or results[name] is None:
            continue
        _validate_raw_log_evidence_length(
            results[name],
            name=name,
            expected_count=expected_count,
            state_path=state_path,
        )


def _validate_raw_log_evidence_length(
    values: object,
    *,
    name: str,
    expected_count: int,
    state_path: Path,
) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, SequenceABC):
        raise ValueError(f"{name} at {state_path} must contain one value per posterior sample.")
    if len(values) != expected_count:
        raise ValueError(
            f"{name} length {len(values)} at {state_path} does not match "
            f"Posterior Sample Database length {expected_count}."
        )


def _finite_logposterior_ratio(columns: Mapping[str, Sequence[object]], *, expected_count: int) -> float:
    values = _logposterior_values(columns, expected_count=expected_count)
    return float(np.count_nonzero(np.isfinite(values)) / expected_count)


def _logposterior_max(columns: Mapping[str, Sequence[object]], *, expected_count: int) -> float | None:
    values = _logposterior_values(columns, expected_count=expected_count)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _logposterior_values(columns: Mapping[str, Sequence[object]], *, expected_count: int) -> np.ndarray:
    if "logPosterior" in columns:
        values = _log_evidence_column(columns, "logPosterior", expected_count=expected_count)
    elif "logLikelihood" in columns and "logPrior" in columns:
        values = _log_evidence_column(
            columns,
            "logLikelihood",
            expected_count=expected_count,
        ) + _log_evidence_column(columns, "logPrior", expected_count=expected_count)
    else:
        raise ValueError(
            "Posterior samples must include logPosterior or both logLikelihood and logPrior."
        )
    return values


def _log_evidence_column(
    columns: Mapping[str, Sequence[object]],
    name: str,
    *,
    expected_count: int,
) -> np.ndarray:
    values = np.asarray(columns[name], dtype=float)
    if values.ndim != 1 or values.shape[0] != expected_count:
        raise ValueError(f"{name} column length does not match parameter samples.")
    return values


def _quantile_key(value: float) -> str:
    return f"q{value:g}"


__all__ = [
    "DEFAULT_POSTERIOR_EQUIVALENCE_THRESHOLDS",
    "PosteriorEquivalenceReport",
    "PosteriorEquivalenceThresholds",
    "PosteriorSummary",
    "compare_posterior_samples",
    "compare_posterior_summaries",
    "load_phase2_posterior_samples",
    "summarize_posterior_samples",
]
