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


def _nonnegative_float(value: float, label: str) -> float:
    numeric = _finite_float(value, label)
    if numeric < 0.0:
        raise ValueError(f"{label} must be >= 0.0; got {numeric}.")
    return numeric


def _positive_float(value: float, label: str) -> float:
    numeric = _finite_float(value, label)
    if numeric <= 0.0:
        raise ValueError(f"{label} must be positive; got {numeric}.")
    return numeric


def _finite_vector(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one value.")
    return result


def _coerce_matrix(value: Sequence[Sequence[float]], label: str, *, rows: int | None = None) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be finite numeric scalars.") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a 2D array; got shape {matrix.shape}.")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{label} row count {matrix.shape[0]} does not match observation count {rows}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} values must be finite.")
    return matrix


def _coerce_square_matrix(
    value: Sequence[Sequence[float]],
    label: str,
    *,
    size: int,
    require_psd: bool = True,
) -> np.ndarray:
    matrix = _coerce_matrix(value, label)
    if matrix.shape != (size, size):
        raise ValueError(f"{label} shape {matrix.shape} does not match expected {(size, size)}.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError(f"{label} must be symmetric.")
    if require_psd and np.min(np.linalg.eigvalsh(0.5 * matrix + 0.5 * matrix.T)) < -1e-10:
        raise ValueError(f"{label} must be positive semidefinite.")
    return matrix


def _as_tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _component_group(name: str) -> str | None:
    if name.startswith("observation:"):
        return "observation"
    if name.startswith("measurement:"):
        return "measurement"
    if name.startswith("surrogate:"):
        return "surrogate"
    if name.startswith("discrepancy:") or name.startswith("model_discrepancy"):
        return "model_discrepancy"
    return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = abs(float(denominator))
    if denominator <= _EPSILON:
        return 0.0 if abs(float(numerator)) <= _EPSILON else float("inf")
    return float(numerator) / denominator


@dataclass(frozen=True)
class SyntheticRecoveryThresholds:
    max_abs_parameter_bias: float = 0.08
    max_parameter_z_error: float = 2.0
    max_residual_rmse_over_observation_rms: float = 0.08
    min_interval_coverage: float = 1.0
    interval_z: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_abs_parameter_bias", _nonnegative_float(self.max_abs_parameter_bias, "max_abs_parameter_bias"))
        object.__setattr__(self, "max_parameter_z_error", _nonnegative_float(self.max_parameter_z_error, "max_parameter_z_error"))
        object.__setattr__(self, "max_residual_rmse_over_observation_rms", _nonnegative_float(self.max_residual_rmse_over_observation_rms, "max_residual_rmse_over_observation_rms"))
        object.__setattr__(self, "min_interval_coverage", _nonnegative_float(self.min_interval_coverage, "min_interval_coverage"))
        object.__setattr__(self, "interval_z", _positive_float(self.interval_z, "interval_z"))
        if self.min_interval_coverage > 1.0:
            raise ValueError("min_interval_coverage must be <= 1.0.")

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class SyntheticRecoveryInputs:
    scenario_id: str
    seed: int
    design_matrix: tuple[tuple[float, ...], ...]
    observations: tuple[float, ...]
    true_parameters: tuple[float, ...]
    parameter_names: tuple[str, ...]
    total_covariance: tuple[tuple[float, ...], ...]
    covariance_components: Mapping[str, Sequence[Sequence[float]]]
    expected_observables: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty.")
        seed = int(self.seed)
        observations = _finite_vector(self.observations, "observations")
        point_count = len(observations)
        design_matrix = _coerce_matrix(self.design_matrix, "design_matrix", rows=point_count)
        if design_matrix.shape[1] < 1:
            raise ValueError("design_matrix must contain at least one parameter column.")
        parameter_count = int(design_matrix.shape[1])
        true_parameters = _finite_vector(self.true_parameters, "true_parameters")
        if len(true_parameters) != parameter_count:
            raise ValueError("true_parameters count must match design_matrix column count.")
        parameter_names = tuple(str(name).strip() for name in self.parameter_names)
        if any(not name for name in parameter_names):
            raise ValueError("parameter_names must be non-empty.")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_names must be unique.")
        if len(parameter_names) != parameter_count:
            raise ValueError("parameter_names count must match design_matrix column count.")
        total_covariance = _coerce_square_matrix(self.total_covariance, "total_covariance", size=point_count)
        if np.min(np.linalg.eigvalsh(0.5 * total_covariance + 0.5 * total_covariance.T)) <= 0.0:
            raise ValueError("total_covariance must be positive definite for recovery.")
        if not self.covariance_components:
            raise ValueError("covariance_components must contain at least one named component.")
        components = {
            str(name): _as_tuple_matrix(
                _coerce_square_matrix(matrix, f"covariance_components[{name}]", size=point_count, require_psd=False)
            )
            for name, matrix in self.covariance_components.items()
        }
        expected = None
        if self.expected_observables is not None:
            expected = _finite_vector(self.expected_observables, "expected_observables")
            if len(expected) != point_count:
                raise ValueError("expected_observables length must match observations.")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "design_matrix", _as_tuple_matrix(design_matrix))
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "true_parameters", true_parameters)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "total_covariance", _as_tuple_matrix(total_covariance))
        object.__setattr__(self, "covariance_components", components)
        object.__setattr__(self, "expected_observables", expected)


@dataclass(frozen=True)
class SyntheticRecoveryResult:
    scenario_id: str
    gate_status: str
    parameter_estimates: Mapping[str, float]
    parameter_standard_deviation: Mapping[str, float]
    parameter_bias: Mapping[str, float]
    parameter_abs_error: Mapping[str, float]
    parameter_z_error: Mapping[str, float]
    interval_covered: Mapping[str, bool]
    metrics: Mapping[str, float | int | bool | None]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    covariance_group_trace_shares: Mapping[str, float | None]
    summary: Mapping[str, Any]


def evaluate_synthetic_recovery(
    inputs: SyntheticRecoveryInputs,
    thresholds: SyntheticRecoveryThresholds | None = None,
) -> SyntheticRecoveryResult:
    thresholds = SyntheticRecoveryThresholds() if thresholds is None else thresholds
    design = np.asarray(inputs.design_matrix, dtype=float)
    observations = np.asarray(inputs.observations, dtype=float)
    true_parameters = np.asarray(inputs.true_parameters, dtype=float)
    covariance = np.asarray(inputs.total_covariance, dtype=float)
    try:
        precision_design = np.linalg.solve(covariance, design)
        precision_observations = np.linalg.solve(covariance, observations)
    except np.linalg.LinAlgError as exc:
        raise ValueError("total_covariance must be invertible for synthetic recovery.") from exc
    normal = design.T @ precision_design
    rhs = design.T @ precision_observations
    try:
        parameter_covariance = np.linalg.inv(normal)
        estimate = parameter_covariance @ rhs
    except np.linalg.LinAlgError as exc:
        raise ValueError("design_matrix is not identifiable under total_covariance.") from exc
    fitted = design @ estimate
    residual = observations - fitted
    residual_rmse = float(np.sqrt(np.mean(residual * residual)))
    observation_rms = float(np.sqrt(np.mean(observations * observations)))
    expected_observables = design @ true_parameters if inputs.expected_observables is None else np.asarray(inputs.expected_observables, dtype=float)
    observable_error = fitted - expected_observables
    observable_rmse = float(np.sqrt(np.mean(observable_error * observable_error)))
    parameter_sd = np.sqrt(np.maximum(np.diag(parameter_covariance), 0.0))
    bias = estimate - true_parameters
    abs_error = np.abs(bias)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_error = np.divide(abs_error, parameter_sd, out=np.full_like(abs_error, np.inf), where=parameter_sd > _EPSILON)
    covered = abs_error <= thresholds.interval_z * parameter_sd
    coverage = float(np.mean(covered)) if covered.size else 0.0
    covariance_groups = _covariance_group_trace_shares(inputs)
    warnings: list[str] = []
    failures: list[str] = []
    max_abs_error = float(np.max(abs_error)) if abs_error.size else 0.0
    max_z_error = float(np.max(z_error)) if z_error.size else 0.0
    residual_rmse_over_observation_rms = _safe_ratio(residual_rmse, observation_rms)
    if max_abs_error > thresholds.max_abs_parameter_bias:
        failures.append("parameter bias exceeds the synthetic recovery threshold.")
    if max_z_error > thresholds.max_parameter_z_error:
        failures.append("parameter z-error exceeds the synthetic recovery threshold.")
    if residual_rmse_over_observation_rms > thresholds.max_residual_rmse_over_observation_rms:
        failures.append("residual RMSE exceeds the observation-scale threshold.")
    if coverage < thresholds.min_interval_coverage:
        failures.append("parameter interval coverage is below the synthetic recovery threshold.")
    if not np.all(np.isfinite(fitted)) or not np.all(np.isfinite(residual)):
        failures.append("recovered observables or residuals are not finite.")
    if not failures and max_z_error > 0.75 * thresholds.max_parameter_z_error:
        warnings.append("parameter z-error is close to the synthetic recovery threshold.")
    gate_status = "fail" if failures else "warn" if warnings else "pass"
    parameter_estimates = {name: float(value) for name, value in zip(inputs.parameter_names, estimate)}
    parameter_standard_deviation = {name: float(value) for name, value in zip(inputs.parameter_names, parameter_sd)}
    parameter_bias = {name: float(value) for name, value in zip(inputs.parameter_names, bias)}
    parameter_abs_error = {name: float(value) for name, value in zip(inputs.parameter_names, abs_error)}
    parameter_z_error = {name: float(value) for name, value in zip(inputs.parameter_names, z_error)}
    interval_covered = {name: bool(value) for name, value in zip(inputs.parameter_names, covered)}
    metrics: dict[str, float | int | bool | None] = {
        "seed": int(inputs.seed),
        "point_count": int(observations.size),
        "parameter_count": int(true_parameters.size),
        "max_abs_parameter_bias": max_abs_error,
        "max_parameter_z_error": max_z_error,
        "parameter_interval_coverage": coverage,
        "residual_rmse": residual_rmse,
        "observation_rms": observation_rms,
        "residual_rmse_over_observation_rms": residual_rmse_over_observation_rms,
        "observable_rmse": observable_rmse,
        "finite_observables": bool(np.all(np.isfinite(fitted))),
        "finite_residuals": bool(np.all(np.isfinite(residual))),
        "design_condition_number": float(np.linalg.cond(design)),
        "covariance_condition_number": float(np.linalg.cond(covariance)),
    }
    summary = {
        "schema_version": 1,
        "scenario_id": inputs.scenario_id,
        "gate_status": gate_status,
        "metrics": metrics,
        "thresholds": thresholds.as_dict(),
        "parameter_names": inputs.parameter_names,
        "true_parameters": {name: float(value) for name, value in zip(inputs.parameter_names, true_parameters)},
        "parameter_estimates": parameter_estimates,
        "parameter_standard_deviation": parameter_standard_deviation,
        "parameter_bias": parameter_bias,
        "parameter_abs_error": parameter_abs_error,
        "parameter_z_error": parameter_z_error,
        "interval_covered": interval_covered,
        "covariance_group_trace_shares": covariance_groups,
        "warnings": tuple(warnings),
        "failures": tuple(failures),
    }
    return SyntheticRecoveryResult(
        scenario_id=inputs.scenario_id,
        gate_status=gate_status,
        parameter_estimates=parameter_estimates,
        parameter_standard_deviation=parameter_standard_deviation,
        parameter_bias=parameter_bias,
        parameter_abs_error=parameter_abs_error,
        parameter_z_error=parameter_z_error,
        interval_covered=interval_covered,
        metrics=metrics,
        warnings=tuple(warnings),
        failures=tuple(failures),
        covariance_group_trace_shares=covariance_groups,
        summary=summary,
    )


def _covariance_group_trace_shares(inputs: SyntheticRecoveryInputs) -> dict[str, float | None]:
    total_trace = float(np.trace(np.asarray(inputs.total_covariance, dtype=float)))
    traces: dict[str, float] = {}
    for name, matrix in inputs.covariance_components.items():
        group = _component_group(name)
        if group is None:
            continue
        traces[group] = traces.get(group, 0.0) + float(np.trace(np.asarray(matrix, dtype=float)))
    return {group: _safe_ratio(trace, total_trace) for group, trace in traces.items()}


__all__ = [
    "SyntheticRecoveryInputs",
    "SyntheticRecoveryResult",
    "SyntheticRecoveryThresholds",
    "evaluate_synthetic_recovery",
]
