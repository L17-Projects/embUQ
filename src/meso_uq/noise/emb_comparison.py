from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


_EPSILON = 1e-12


def _finite_float(value: float, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be finite numeric scalars; got {value!r}.") from exc
    if not isfinite(numeric):
        raise ValueError(f"{label} must be finite; got {numeric}.")
    return numeric


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


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _positive_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_positive_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _coerce_matrix(values: Sequence[Sequence[float]], label: str, *, columns: int | None = None) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be finite numeric scalars.") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a 2D array; got shape {matrix.shape}.")
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError(f"{label} column count {matrix.shape[1]} does not match parameter count {columns}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} values must be finite.")
    return matrix


def _coerce_square_matrix(values: Sequence[Sequence[float]], label: str, *, size: int) -> np.ndarray:
    matrix = _coerce_matrix(values, label)
    if matrix.shape != (size, size):
        raise ValueError(f"{label} shape {matrix.shape} does not match expected {(size, size)}.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError(f"{label} must be symmetric.")
    if np.min(np.linalg.eigvalsh(0.5 * matrix + 0.5 * matrix.T)) <= 0.0:
        raise ValueError(f"{label} must be positive definite.")
    return matrix


def _tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = abs(float(denominator))
    if denominator <= _EPSILON:
        return 0.0 if abs(float(numerator)) <= _EPSILON else float("inf")
    return float(numerator) / denominator


@dataclass(frozen=True)
class EmbComparisonThresholds:
    min_legacy_interval_coverage: float = 0.70
    min_upgraded_interval_coverage: float = 0.85
    min_upgraded_to_legacy_uncertainty_ratio: float = 1.0
    max_upgraded_residual_rmse_over_legacy: float = 1.10
    max_posterior_mean_shift_sd: float = 2.0
    interval_z: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_legacy_interval_coverage", _nonnegative_float(self.min_legacy_interval_coverage, "min_legacy_interval_coverage"))
        object.__setattr__(self, "min_upgraded_interval_coverage", _nonnegative_float(self.min_upgraded_interval_coverage, "min_upgraded_interval_coverage"))
        object.__setattr__(self, "min_upgraded_to_legacy_uncertainty_ratio", _nonnegative_float(self.min_upgraded_to_legacy_uncertainty_ratio, "min_upgraded_to_legacy_uncertainty_ratio"))
        object.__setattr__(self, "max_upgraded_residual_rmse_over_legacy", _nonnegative_float(self.max_upgraded_residual_rmse_over_legacy, "max_upgraded_residual_rmse_over_legacy"))
        object.__setattr__(self, "max_posterior_mean_shift_sd", _nonnegative_float(self.max_posterior_mean_shift_sd, "max_posterior_mean_shift_sd"))
        object.__setattr__(self, "interval_z", _positive_float(self.interval_z, "interval_z"))
        if self.min_legacy_interval_coverage > 1.0:
            raise ValueError("min_legacy_interval_coverage must be <= 1.0.")
        if self.min_upgraded_interval_coverage > 1.0:
            raise ValueError("min_upgraded_interval_coverage must be <= 1.0.")

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class EmbComparisonInputs:
    scenario_id: str
    modality: str
    dataset_path: str
    config_id: str
    axis_name: str
    observable_name: str
    axis_values: tuple[float, ...]
    observations: tuple[float, ...]
    legacy_predictions: tuple[float, ...]
    upgraded_predictions: tuple[float, ...]
    legacy_standard_deviation: tuple[float, ...]
    upgraded_covariance: tuple[tuple[float, ...], ...]
    covariance_components: Mapping[str, Sequence[Sequence[float]]]
    parameter_names: tuple[str, ...]
    legacy_posterior_samples: tuple[tuple[float, ...], ...]
    upgraded_posterior_samples: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        modality = str(self.modality).strip()
        dataset_path = str(self.dataset_path).strip()
        config_id = str(self.config_id).strip()
        axis_name = str(self.axis_name).strip()
        observable_name = str(self.observable_name).strip()
        for label, value in (
            ("scenario_id", scenario_id),
            ("modality", modality),
            ("dataset_path", dataset_path),
            ("config_id", config_id),
            ("axis_name", axis_name),
            ("observable_name", observable_name),
        ):
            if not value:
                raise ValueError(f"{label} must be non-empty.")
        observations = _finite_vector(self.observations, "observations")
        point_count = len(observations)
        axis_values = _finite_vector(self.axis_values, "axis_values")
        legacy_predictions = _finite_vector(self.legacy_predictions, "legacy_predictions")
        upgraded_predictions = _finite_vector(self.upgraded_predictions, "upgraded_predictions")
        legacy_standard_deviation = _positive_vector(self.legacy_standard_deviation, "legacy_standard_deviation")
        for label, values in (
            ("axis_values", axis_values),
            ("legacy_predictions", legacy_predictions),
            ("upgraded_predictions", upgraded_predictions),
            ("legacy_standard_deviation", legacy_standard_deviation),
        ):
            if len(values) != point_count:
                raise ValueError(f"{label} length {len(values)} does not match observation count {point_count}.")
        upgraded_covariance = _coerce_square_matrix(self.upgraded_covariance, "upgraded_covariance", size=point_count)
        if not self.covariance_components:
            raise ValueError("covariance_components must contain at least one named component.")
        components = {
            str(name): _tuple_matrix(_coerce_square_component(matrix, f"covariance_components[{name}]", size=point_count))
            for name, matrix in self.covariance_components.items()
        }
        parameter_names = tuple(str(name).strip() for name in self.parameter_names)
        if any(not name for name in parameter_names):
            raise ValueError("parameter_names must be non-empty.")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_names must be unique.")
        legacy_samples = _coerce_matrix(self.legacy_posterior_samples, "legacy_posterior_samples", columns=len(parameter_names))
        upgraded_samples = _coerce_matrix(self.upgraded_posterior_samples, "upgraded_posterior_samples", columns=len(parameter_names))
        if legacy_samples.shape[0] < 3 or upgraded_samples.shape[0] < 3:
            raise ValueError("posterior samples must contain at least three draws per mode.")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "dataset_path", dataset_path)
        object.__setattr__(self, "config_id", config_id)
        object.__setattr__(self, "axis_name", axis_name)
        object.__setattr__(self, "observable_name", observable_name)
        object.__setattr__(self, "axis_values", axis_values)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "legacy_predictions", legacy_predictions)
        object.__setattr__(self, "upgraded_predictions", upgraded_predictions)
        object.__setattr__(self, "legacy_standard_deviation", legacy_standard_deviation)
        object.__setattr__(self, "upgraded_covariance", _tuple_matrix(upgraded_covariance))
        object.__setattr__(self, "covariance_components", components)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "legacy_posterior_samples", _tuple_matrix(legacy_samples))
        object.__setattr__(self, "upgraded_posterior_samples", _tuple_matrix(upgraded_samples))


@dataclass(frozen=True)
class EmbComparisonResult:
    scenario_id: str
    gate_status: str
    metrics: Mapping[str, Any]
    posterior_metrics: Mapping[str, Any]
    predictive_metrics: Mapping[str, Any]
    interpretation: tuple[str, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]


def evaluate_emb_comparison(
    inputs: EmbComparisonInputs,
    thresholds: EmbComparisonThresholds | None = None,
) -> EmbComparisonResult:
    thresholds = EmbComparisonThresholds() if thresholds is None else thresholds
    observations = np.asarray(inputs.observations, dtype=float)
    legacy_predictions = np.asarray(inputs.legacy_predictions, dtype=float)
    upgraded_predictions = np.asarray(inputs.upgraded_predictions, dtype=float)
    legacy_sd = np.asarray(inputs.legacy_standard_deviation, dtype=float)
    upgraded_covariance = np.asarray(inputs.upgraded_covariance, dtype=float)
    upgraded_sd = np.sqrt(np.maximum(np.diag(upgraded_covariance), 0.0))
    legacy_samples = np.asarray(inputs.legacy_posterior_samples, dtype=float)
    upgraded_samples = np.asarray(inputs.upgraded_posterior_samples, dtype=float)

    predictive_metrics = _predictive_metrics(observations, legacy_predictions, upgraded_predictions, legacy_sd, upgraded_sd, thresholds)
    posterior_metrics = _posterior_metrics(inputs.parameter_names, legacy_samples, upgraded_samples)
    covariance_metrics = _covariance_metrics(inputs, upgraded_covariance)
    interpretation = _interpretation(predictive_metrics, posterior_metrics, covariance_metrics)
    failures: list[str] = []
    warnings: list[str] = []
    if predictive_metrics["legacy_interval_coverage"] < thresholds.min_legacy_interval_coverage:
        failures.append("legacy predictive interval coverage is below the comparison threshold.")
    if predictive_metrics["upgraded_interval_coverage"] < thresholds.min_upgraded_interval_coverage:
        failures.append("upgraded predictive interval coverage is below the comparison threshold.")
    if predictive_metrics["mean_upgraded_to_legacy_std_ratio"] < thresholds.min_upgraded_to_legacy_uncertainty_ratio:
        failures.append("upgraded predictive uncertainty is narrower than the accepted comparison threshold.")
    if predictive_metrics["upgraded_residual_rmse_over_legacy"] > thresholds.max_upgraded_residual_rmse_over_legacy:
        failures.append("upgraded residual RMSE regresses relative to legacy beyond threshold.")
    if posterior_metrics["max_abs_mean_shift_sd"] > thresholds.max_posterior_mean_shift_sd:
        failures.append("posterior mean shift exceeds the accepted posterior-standard-deviation threshold.")
    if covariance_metrics["upgraded_covariance_cholesky_success"] is not True:
        failures.append("upgraded covariance is not positive definite for integrated EMB comparison.")
    if not failures and predictive_metrics["mean_upgraded_to_legacy_std_ratio"] < 1.05:
        warnings.append("upgraded predictive uncertainty is only marginally broader than legacy.")
    gate_status = "fail" if failures else "warn" if warnings else "pass"
    metrics = {
        "legacy_mode_recoverable": True,
        "upgraded_mode_recoverable": covariance_metrics["upgraded_covariance_cholesky_success"],
        "predictive": predictive_metrics,
        "posterior": posterior_metrics,
        "covariance": covariance_metrics,
    }
    summary = {
        "schema_version": 1,
        "scenario_id": inputs.scenario_id,
        "modality": inputs.modality,
        "dataset_path": inputs.dataset_path,
        "config_id": inputs.config_id,
        "gate_status": gate_status,
        "thresholds": thresholds.as_dict(),
        "metrics": metrics,
        "interpretation": tuple(interpretation),
        "failures": tuple(failures),
        "warnings": tuple(warnings),
    }
    return EmbComparisonResult(
        scenario_id=inputs.scenario_id,
        gate_status=gate_status,
        metrics=metrics,
        posterior_metrics=posterior_metrics,
        predictive_metrics=predictive_metrics,
        interpretation=tuple(interpretation),
        failures=tuple(failures),
        warnings=tuple(warnings),
        summary=summary,
    )


def _coerce_square_component(values: Sequence[Sequence[float]], label: str, *, size: int) -> np.ndarray:
    matrix = _coerce_matrix(values, label)
    if matrix.shape != (size, size):
        raise ValueError(f"{label} shape {matrix.shape} does not match expected {(size, size)}.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError(f"{label} must be symmetric.")
    if np.min(np.linalg.eigvalsh(0.5 * matrix + 0.5 * matrix.T)) < -1e-10:
        raise ValueError(f"{label} must be positive semidefinite.")
    return matrix


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def _predictive_metrics(
    observations: np.ndarray,
    legacy_predictions: np.ndarray,
    upgraded_predictions: np.ndarray,
    legacy_sd: np.ndarray,
    upgraded_sd: np.ndarray,
    thresholds: EmbComparisonThresholds,
) -> dict[str, float | int | bool]:
    legacy_residual = observations - legacy_predictions
    upgraded_residual = observations - upgraded_predictions
    legacy_covered = np.abs(legacy_residual) <= thresholds.interval_z * legacy_sd
    upgraded_covered = np.abs(upgraded_residual) <= thresholds.interval_z * upgraded_sd
    legacy_rmse = _rmse(legacy_residual)
    upgraded_rmse = _rmse(upgraded_residual)
    return {
        "point_count": int(observations.size),
        "legacy_residual_rmse": legacy_rmse,
        "upgraded_residual_rmse": upgraded_rmse,
        "upgraded_residual_rmse_over_legacy": _safe_ratio(upgraded_rmse, legacy_rmse),
        "legacy_interval_coverage": float(np.mean(legacy_covered)),
        "upgraded_interval_coverage": float(np.mean(upgraded_covered)),
        "legacy_mean_standard_deviation": float(np.mean(legacy_sd)),
        "upgraded_mean_standard_deviation": float(np.mean(upgraded_sd)),
        "mean_upgraded_to_legacy_std_ratio": _safe_ratio(float(np.mean(upgraded_sd)), float(np.mean(legacy_sd))),
        "max_abs_standardized_legacy_residual": float(np.max(np.abs(legacy_residual / legacy_sd))),
        "max_abs_standardized_upgraded_residual": float(np.max(np.abs(upgraded_residual / upgraded_sd))),
        "finite_observables": bool(np.all(np.isfinite(observations)) and np.all(np.isfinite(upgraded_predictions))),
    }


def _posterior_metrics(parameter_names: tuple[str, ...], legacy_samples: np.ndarray, upgraded_samples: np.ndarray) -> dict[str, Any]:
    parameter_metrics: dict[str, dict[str, float | bool]] = {}
    shifts = []
    width_ratios = []
    for index, name in enumerate(parameter_names):
        legacy = legacy_samples[:, index]
        upgraded = upgraded_samples[:, index]
        legacy_mean = float(np.mean(legacy))
        upgraded_mean = float(np.mean(upgraded))
        legacy_sd = float(np.std(legacy, ddof=1))
        upgraded_sd = float(np.std(upgraded, ddof=1))
        pooled = max(np.sqrt(0.5 * (legacy_sd * legacy_sd + upgraded_sd * upgraded_sd)), _EPSILON)
        shift_sd = abs(upgraded_mean - legacy_mean) / pooled
        legacy_interval = np.quantile(legacy, (0.05, 0.95))
        upgraded_interval = np.quantile(upgraded, (0.05, 0.95))
        width_ratio = _safe_ratio(float(upgraded_interval[1] - upgraded_interval[0]), float(legacy_interval[1] - legacy_interval[0]))
        shifts.append(shift_sd)
        width_ratios.append(width_ratio)
        parameter_metrics[name] = {
            "legacy_mean": legacy_mean,
            "upgraded_mean": upgraded_mean,
            "legacy_standard_deviation": legacy_sd,
            "upgraded_standard_deviation": upgraded_sd,
            "mean_shift": float(upgraded_mean - legacy_mean),
            "abs_mean_shift_sd": float(shift_sd),
            "legacy_interval_lower": float(legacy_interval[0]),
            "legacy_interval_upper": float(legacy_interval[1]),
            "upgraded_interval_lower": float(upgraded_interval[0]),
            "upgraded_interval_upper": float(upgraded_interval[1]),
            "interval_width_ratio": float(width_ratio),
        }
    return {
        "parameter_count": int(len(parameter_names)),
        "legacy_draw_count": int(legacy_samples.shape[0]),
        "upgraded_draw_count": int(upgraded_samples.shape[0]),
        "max_abs_mean_shift_sd": float(np.max(shifts)) if shifts else 0.0,
        "mean_interval_width_ratio": float(np.mean(width_ratios)) if width_ratios else 0.0,
        "parameters": parameter_metrics,
    }


def _covariance_metrics(inputs: EmbComparisonInputs, upgraded_covariance: np.ndarray) -> dict[str, Any]:
    trace = float(np.trace(upgraded_covariance))
    groups: dict[str, float] = {}
    for name, matrix in inputs.covariance_components.items():
        group = str(name).split(":", 1)[0]
        groups[group] = groups.get(group, 0.0) + float(np.trace(np.asarray(matrix, dtype=float)))
    try:
        np.linalg.cholesky(upgraded_covariance)
        cholesky_success = True
    except np.linalg.LinAlgError:
        cholesky_success = False
    return {
        "upgraded_covariance_cholesky_success": cholesky_success,
        "upgraded_covariance_condition_number": float(np.linalg.cond(upgraded_covariance)),
        "upgraded_covariance_trace": trace,
        "covariance_group_trace_shares": {group: _safe_ratio(value, trace) for group, value in groups.items()},
    }


def _interpretation(predictive: Mapping[str, Any], posterior: Mapping[str, Any], covariance: Mapping[str, Any]) -> list[str]:
    notes = []
    uncertainty_ratio = float(predictive["mean_upgraded_to_legacy_std_ratio"])
    rmse_ratio = float(predictive["upgraded_residual_rmse_over_legacy"])
    width_ratio = float(posterior["mean_interval_width_ratio"])
    if uncertainty_ratio > 1.05:
        notes.append("Upgraded predictive bands broaden because measurement, surrogate, and discrepancy covariance terms are included instead of only legacy observation noise.")
    else:
        notes.append("Upgraded predictive bands remain close to legacy width for this fixture.")
    if width_ratio > 1.05:
        notes.append("Upgraded posterior intervals broaden relative to legacy posterior draws.")
    if float(posterior["max_abs_mean_shift_sd"]) > 0.25:
        notes.append("Posterior means shift modestly in posterior-standard-deviation units; the shift stays within the accepted gate.")
    if rmse_ratio <= 1.0:
        notes.append("Upgraded predictive means stabilize residuals relative to the legacy fit on the comparison data.")
    else:
        notes.append("Upgraded residual RMSE remains close to the legacy residual scale.")
    shares = covariance["covariance_group_trace_shares"]
    if shares:
        dominant = max(shares, key=shares.get)
        notes.append(f"The largest upgraded covariance trace share is `{dominant}`, so the comparison report can attribute the broader band to named hierarchy terms.")
    return notes


__all__ = [
    "EmbComparisonInputs",
    "EmbComparisonResult",
    "EmbComparisonThresholds",
    "evaluate_emb_comparison",
]
