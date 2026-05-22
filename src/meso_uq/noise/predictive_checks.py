from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


_EPSILON = 1e-12
_SUMMARY_NAMES = ("mean", "standard_deviation", "minimum", "maximum", "range")


def _finite_float(value: float, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be finite numeric scalars; got {value!r}.") from exc
    if not isfinite(numeric):
        raise ValueError(f"{label} must be finite; got {numeric}.")
    return numeric


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _numeric_sample_matrix(values: Sequence[Sequence[float]], label: str, *, columns: int) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be numeric scalars.") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a 2D array; got shape {matrix.shape}.")
    if matrix.shape[0] < 2:
        raise ValueError(f"{label} must contain at least two predictive draws.")
    if matrix.shape[1] != columns:
        raise ValueError(f"{label} column count {matrix.shape[1]} does not match observed count {columns}.")
    return matrix


def _tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _positive_float(value: float, label: str) -> float:
    numeric = _finite_float(value, label)
    if numeric <= 0.0:
        raise ValueError(f"{label} must be positive; got {numeric}.")
    return numeric


def _nonnegative_float(value: float, label: str) -> float:
    numeric = _finite_float(value, label)
    if numeric < 0.0:
        raise ValueError(f"{label} must be >= 0.0; got {numeric}.")
    return numeric


@dataclass(frozen=True)
class PredictiveCheckThresholds:
    max_summary_z_score: float = 2.5
    min_pointwise_interval_coverage: float = 0.85
    max_pointwise_rank_edge_fraction: float = 0.45
    max_sbc_rank_histogram_l1: float = 0.45
    max_sbc_mean_rank_quantile_error: float = 0.15
    max_sbc_edge_fraction: float = 0.35
    min_sbc_interval_coverage: float = 0.80
    min_finite_fraction: float = 1.0
    interval_alpha: float = 0.10
    rank_edge_alpha: float = 0.10
    rank_bin_count: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_summary_z_score", _nonnegative_float(self.max_summary_z_score, "max_summary_z_score"))
        object.__setattr__(self, "min_pointwise_interval_coverage", _nonnegative_float(self.min_pointwise_interval_coverage, "min_pointwise_interval_coverage"))
        object.__setattr__(self, "max_pointwise_rank_edge_fraction", _nonnegative_float(self.max_pointwise_rank_edge_fraction, "max_pointwise_rank_edge_fraction"))
        object.__setattr__(self, "max_sbc_rank_histogram_l1", _nonnegative_float(self.max_sbc_rank_histogram_l1, "max_sbc_rank_histogram_l1"))
        object.__setattr__(self, "max_sbc_mean_rank_quantile_error", _nonnegative_float(self.max_sbc_mean_rank_quantile_error, "max_sbc_mean_rank_quantile_error"))
        object.__setattr__(self, "max_sbc_edge_fraction", _nonnegative_float(self.max_sbc_edge_fraction, "max_sbc_edge_fraction"))
        object.__setattr__(self, "min_sbc_interval_coverage", _nonnegative_float(self.min_sbc_interval_coverage, "min_sbc_interval_coverage"))
        object.__setattr__(self, "min_finite_fraction", _nonnegative_float(self.min_finite_fraction, "min_finite_fraction"))
        object.__setattr__(self, "interval_alpha", _positive_float(self.interval_alpha, "interval_alpha"))
        object.__setattr__(self, "rank_edge_alpha", _positive_float(self.rank_edge_alpha, "rank_edge_alpha"))
        object.__setattr__(self, "rank_bin_count", int(self.rank_bin_count))
        if self.min_pointwise_interval_coverage > 1.0:
            raise ValueError("min_pointwise_interval_coverage must be <= 1.0.")
        if self.min_sbc_interval_coverage > 1.0:
            raise ValueError("min_sbc_interval_coverage must be <= 1.0.")
        if self.min_finite_fraction > 1.0:
            raise ValueError("min_finite_fraction must be <= 1.0.")
        if not 0.0 < self.interval_alpha < 1.0:
            raise ValueError("interval_alpha must be between 0 and 1.")
        if not 0.0 < self.rank_edge_alpha < 0.5:
            raise ValueError("rank_edge_alpha must be between 0 and 0.5.")
        if self.rank_bin_count < 2:
            raise ValueError("rank_bin_count must be at least 2.")

    def as_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {key: float(value) for key, value in self.__dict__.items() if key != "rank_bin_count"}
        payload["rank_bin_count"] = int(self.rank_bin_count)
        return payload


@dataclass(frozen=True)
class SbcRankRecord:
    name: str
    true_value: float
    posterior_samples: tuple[float, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("SBC rank record name must be non-empty.")
        samples = _finite_vector(self.posterior_samples, f"posterior_samples[{name}]")
        if len(samples) < 2:
            raise ValueError(f"posterior_samples[{name}] must contain at least two draws.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "true_value", _finite_float(self.true_value, f"true_value[{name}]"))
        object.__setattr__(self, "posterior_samples", samples)

    def rank(self) -> float:
        samples = np.asarray(self.posterior_samples, dtype=float)
        less_count = float(np.sum(samples < self.true_value))
        equal_count = float(np.sum(samples == self.true_value))
        return less_count + 0.5 * equal_count

    def rank_quantile(self) -> float:
        return float((self.rank() + 0.5) / (len(self.posterior_samples) + 1.0))

    def interval_covered(self, alpha: float) -> bool:
        samples = np.asarray(self.posterior_samples, dtype=float)
        lower, upper = np.quantile(samples, (alpha / 2.0, 1.0 - alpha / 2.0))
        return bool(lower <= self.true_value <= upper)

    def as_dict(self, *, alpha: float) -> dict[str, float | int | bool | str]:
        samples = np.asarray(self.posterior_samples, dtype=float)
        lower, upper = np.quantile(samples, (alpha / 2.0, 1.0 - alpha / 2.0))
        return {
            "name": self.name,
            "true_value": float(self.true_value),
            "draw_count": int(samples.size),
            "posterior_mean": float(np.mean(samples)),
            "posterior_standard_deviation": float(np.std(samples, ddof=1)),
            "rank": self.rank(),
            "rank_quantile": self.rank_quantile(),
            "interval_lower": float(lower),
            "interval_upper": float(upper),
            "interval_covered": self.interval_covered(alpha),
        }


@dataclass(frozen=True)
class PredictiveCheckInputs:
    scenario_id: str
    seed: int
    observed: tuple[float, ...]
    predictive_samples: tuple[tuple[float, ...], ...]
    sbc_rank_records: tuple[SbcRankRecord, ...] = ()
    summary_names: tuple[str, ...] = _SUMMARY_NAMES

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty.")
        observed = _finite_vector(self.observed, "observed")
        predictive_samples = _numeric_sample_matrix(self.predictive_samples, "predictive_samples", columns=len(observed))
        records = tuple(record if isinstance(record, SbcRankRecord) else SbcRankRecord(**record) for record in self.sbc_rank_records)
        summary_names = tuple(str(name).strip() for name in self.summary_names)
        if summary_names != _SUMMARY_NAMES:
            raise ValueError(f"summary_names must be {_SUMMARY_NAMES} for this deterministic checker.")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "predictive_samples", _tuple_matrix(predictive_samples))
        object.__setattr__(self, "sbc_rank_records", records)
        object.__setattr__(self, "summary_names", summary_names)


@dataclass(frozen=True)
class PredictiveCheckResult:
    scenario_id: str
    gate_status: str
    ppc_metrics: Mapping[str, Any]
    sbc_metrics: Mapping[str, Any]
    runtime_failures: tuple[str, ...]
    calibration_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]


def evaluate_predictive_checks(
    inputs: PredictiveCheckInputs,
    thresholds: PredictiveCheckThresholds | None = None,
) -> PredictiveCheckResult:
    thresholds = PredictiveCheckThresholds() if thresholds is None else thresholds
    observed = np.asarray(inputs.observed, dtype=float)
    all_samples = np.asarray(inputs.predictive_samples, dtype=float)
    finite_fraction = float(np.mean(np.isfinite(all_samples)))
    finite_row_mask = np.all(np.isfinite(all_samples), axis=1)
    samples = all_samples[finite_row_mask]

    runtime_failures: list[str] = []
    calibration_failures: list[str] = []
    warnings: list[str] = []
    if finite_fraction < thresholds.min_finite_fraction:
        runtime_failures.append("predictive samples contain nonfinite values below the required finite fraction.")
    if samples.shape[0] < 2:
        runtime_failures.append("fewer than two finite predictive draws remain after filtering nonfinite rows.")
        ppc_metrics = _empty_ppc_metrics(finite_fraction, all_samples.shape)
    else:
        summary_metrics, summary_failures, summary_warnings = _summary_metrics(observed, samples, thresholds)
        pointwise_metrics = _pointwise_metrics(observed, samples, thresholds)
        ppc_metrics = {
            "finite_fraction": finite_fraction,
            "draw_count": int(all_samples.shape[0]),
            "finite_draw_count": int(samples.shape[0]),
            "observable_count": int(all_samples.shape[1]),
            "summary_metrics": summary_metrics,
            "pointwise_metrics": pointwise_metrics,
        }
        calibration_failures.extend(summary_failures)
        warnings.extend(summary_warnings)
        if pointwise_metrics["interval_coverage"] < thresholds.min_pointwise_interval_coverage:
            calibration_failures.append("pointwise posterior predictive interval coverage is below threshold.")
        if pointwise_metrics["rank_edge_fraction"] > thresholds.max_pointwise_rank_edge_fraction:
            calibration_failures.append("pointwise posterior predictive ranks are over-represented at interval edges.")

    sbc_metrics, sbc_failures, sbc_warnings = _sbc_metrics(inputs.sbc_rank_records, thresholds)
    calibration_failures.extend(sbc_failures)
    warnings.extend(sbc_warnings)
    gate_status = "fail" if runtime_failures or calibration_failures else "warn" if warnings else "pass"
    summary = {
        "schema_version": 1,
        "scenario_id": inputs.scenario_id,
        "seed": int(inputs.seed),
        "gate_status": gate_status,
        "thresholds": thresholds.as_dict(),
        "ppc_metrics": ppc_metrics,
        "sbc_metrics": sbc_metrics,
        "failure_classes": {
            "runtime_or_numerical": tuple(runtime_failures),
            "calibration": tuple(calibration_failures),
        },
        "warnings": tuple(warnings),
    }
    return PredictiveCheckResult(
        scenario_id=inputs.scenario_id,
        gate_status=gate_status,
        ppc_metrics=ppc_metrics,
        sbc_metrics=sbc_metrics,
        runtime_failures=tuple(runtime_failures),
        calibration_failures=tuple(calibration_failures),
        warnings=tuple(warnings),
        summary=summary,
    )


def _empty_ppc_metrics(finite_fraction: float, shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "finite_fraction": finite_fraction,
        "draw_count": int(shape[0]),
        "finite_draw_count": 0,
        "observable_count": int(shape[1]),
        "summary_metrics": {},
        "pointwise_metrics": {
            "interval_coverage": 0.0,
            "rank_edge_fraction": 1.0,
            "mean_rank_quantile": None,
            "max_abs_predictive_mean_error": None,
            "mean_predictive_standard_deviation": None,
            "covered_count": 0,
            "observable_count": int(shape[1]),
            "rank_quantiles": tuple(),
        },
    }


def _observable_summaries(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        (
            float(np.mean(values)),
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            float(np.min(values)),
            float(np.max(values)),
            float(np.max(values) - np.min(values)),
        ),
        dtype=float,
    )


def _summary_metrics(
    observed: np.ndarray,
    samples: np.ndarray,
    thresholds: PredictiveCheckThresholds,
) -> tuple[dict[str, dict[str, float | bool]], list[str], list[str]]:
    observed_summaries = _observable_summaries(observed)
    sample_summaries = np.asarray([_observable_summaries(row) for row in samples], dtype=float)
    metrics: dict[str, dict[str, float | bool]] = {}
    failures: list[str] = []
    warnings: list[str] = []
    for index, name in enumerate(_SUMMARY_NAMES):
        distribution = sample_summaries[:, index]
        center = float(np.mean(distribution))
        spread = float(np.std(distribution, ddof=1))
        observed_value = float(observed_summaries[index])
        z_score = 0.0 if spread <= _EPSILON and abs(observed_value - center) <= _EPSILON else abs(observed_value - center) / max(spread, _EPSILON)
        lower, upper = np.quantile(distribution, (thresholds.interval_alpha / 2.0, 1.0 - thresholds.interval_alpha / 2.0))
        percentile = float(np.mean(distribution <= observed_value))
        metrics[name] = {
            "observed": observed_value,
            "predictive_mean": center,
            "predictive_standard_deviation": spread,
            "z_score": float(z_score),
            "percentile": percentile,
            "interval_lower": float(lower),
            "interval_upper": float(upper),
            "interval_covered": bool(lower <= observed_value <= upper),
        }
        if z_score > thresholds.max_summary_z_score:
            failures.append(f"posterior predictive summary {name!r} z-score exceeds threshold.")
        elif z_score > 0.75 * thresholds.max_summary_z_score:
            warnings.append(f"posterior predictive summary {name!r} z-score is close to threshold.")
    return metrics, failures, warnings


def _pointwise_metrics(observed: np.ndarray, samples: np.ndarray, thresholds: PredictiveCheckThresholds) -> dict[str, Any]:
    lower = np.quantile(samples, thresholds.interval_alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - thresholds.interval_alpha / 2.0, axis=0)
    covered = np.logical_and(lower <= observed, observed <= upper)
    less_counts = np.sum(samples < observed[None, :], axis=0)
    equal_counts = np.sum(samples == observed[None, :], axis=0)
    ranks = less_counts + 0.5 * equal_counts
    rank_quantiles = (ranks + 0.5) / (samples.shape[0] + 1.0)
    edge = np.logical_or(rank_quantiles <= thresholds.rank_edge_alpha, rank_quantiles >= 1.0 - thresholds.rank_edge_alpha)
    absolute_error = np.abs(np.mean(samples, axis=0) - observed)
    predictive_sd = np.std(samples, axis=0, ddof=1)
    return {
        "interval_coverage": float(np.mean(covered)),
        "rank_edge_fraction": float(np.mean(edge)),
        "mean_rank_quantile": float(np.mean(rank_quantiles)),
        "max_abs_predictive_mean_error": float(np.max(absolute_error)),
        "mean_predictive_standard_deviation": float(np.mean(predictive_sd)),
        "covered_count": int(np.sum(covered)),
        "observable_count": int(observed.size),
        "rank_quantiles": tuple(float(value) for value in rank_quantiles.tolist()),
    }


def _sbc_metrics(records: tuple[SbcRankRecord, ...], thresholds: PredictiveCheckThresholds) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if not records:
        return {
            "record_count": 0,
            "rank_histogram": tuple(0 for _ in range(thresholds.rank_bin_count)),
            "rank_histogram_fraction": tuple(0.0 for _ in range(thresholds.rank_bin_count)),
            "rank_histogram_l1": None,
            "mean_rank_quantile": None,
            "mean_rank_quantile_error": None,
            "edge_fraction": None,
            "interval_coverage": None,
            "records": tuple(),
        }, ["SBC rank records are required for M6 predictive calibration evidence."], warnings
    rank_quantiles = np.asarray([record.rank_quantile() for record in records], dtype=float)
    histogram, _ = np.histogram(rank_quantiles, bins=thresholds.rank_bin_count, range=(0.0, 1.0))
    histogram_fraction = histogram / float(np.sum(histogram))
    expected = np.full(thresholds.rank_bin_count, 1.0 / thresholds.rank_bin_count, dtype=float)
    l1 = float(np.sum(np.abs(histogram_fraction - expected)))
    mean_rank = float(np.mean(rank_quantiles))
    mean_error = abs(mean_rank - 0.5)
    edge_fraction = float(np.mean(np.logical_or(rank_quantiles <= thresholds.rank_edge_alpha, rank_quantiles >= 1.0 - thresholds.rank_edge_alpha)))
    interval_coverage = float(np.mean([record.interval_covered(thresholds.interval_alpha) for record in records]))
    if l1 > thresholds.max_sbc_rank_histogram_l1:
        failures.append("SBC rank histogram L1 distance exceeds threshold.")
    if mean_error > thresholds.max_sbc_mean_rank_quantile_error:
        failures.append("SBC mean rank quantile error exceeds threshold.")
    if edge_fraction > thresholds.max_sbc_edge_fraction:
        failures.append("SBC rank edge fraction exceeds threshold.")
    if interval_coverage < thresholds.min_sbc_interval_coverage:
        failures.append("SBC posterior interval coverage is below threshold.")
    if not failures and l1 > 0.75 * thresholds.max_sbc_rank_histogram_l1:
        warnings.append("SBC rank histogram L1 distance is close to threshold.")
    return {
        "record_count": int(len(records)),
        "rank_histogram": tuple(int(value) for value in histogram.tolist()),
        "rank_histogram_fraction": tuple(float(value) for value in histogram_fraction.tolist()),
        "rank_histogram_l1": l1,
        "mean_rank_quantile": mean_rank,
        "mean_rank_quantile_error": float(mean_error),
        "edge_fraction": edge_fraction,
        "interval_coverage": interval_coverage,
        "records": tuple(record.as_dict(alpha=thresholds.interval_alpha) for record in records),
    }, failures, warnings


__all__ = [
    "PredictiveCheckInputs",
    "PredictiveCheckResult",
    "PredictiveCheckThresholds",
    "SbcRankRecord",
    "evaluate_predictive_checks",
]
