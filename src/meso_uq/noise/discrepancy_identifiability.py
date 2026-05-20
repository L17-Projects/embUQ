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


def _finite_vector(values: Sequence[float], label: str, *, allow_empty: bool = False) -> tuple[float, ...]:
    result = tuple(_finite_float(value, label) for value in values)
    if not result and not allow_empty:
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
        raise ValueError(f"{label} row count {matrix.shape[0]} does not match expected count {rows}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} values must be finite.")
    return matrix


def _coerce_square_matrix(value: Sequence[Sequence[float]], label: str, *, size: int | None = None) -> np.ndarray:
    matrix = _coerce_matrix(value, label)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be square; got shape {matrix.shape}.")
    if size is not None and matrix.shape[0] != size:
        raise ValueError(f"{label} size {matrix.shape[0]} does not match expected count {size}.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError(f"{label} must be symmetric.")
    return matrix


def _as_tuple_matrix(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = abs(float(denominator))
    if denominator <= _EPSILON:
        return 0.0 if abs(numerator) <= _EPSILON else float("inf")
    return float(numerator / denominator)


def _clip_unit(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def _column_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[1] if left.ndim == 2 else 0, right.shape[1] if right.ndim == 2 else 0), dtype=float)
    left_centered = left - np.mean(left, axis=0, keepdims=True)
    right_centered = right - np.mean(right, axis=0, keepdims=True)
    left_norm = np.linalg.norm(left_centered, axis=0)
    right_norm = np.linalg.norm(right_centered, axis=0)
    denominator = np.outer(left_norm, right_norm)
    numerator = left_centered.T @ right_centered
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > _EPSILON)


def _first_canonical_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left_centered = left - np.mean(left, axis=0, keepdims=True)
    right_centered = right - np.mean(right, axis=0, keepdims=True)
    try:
        left_u, left_s, _ = np.linalg.svd(left_centered, full_matrices=False)
        right_u, right_s, _ = np.linalg.svd(right_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    left_rank = int(np.sum(left_s > _EPSILON))
    right_rank = int(np.sum(right_s > _EPSILON))
    if left_rank == 0 or right_rank == 0:
        return 0.0
    overlap = left_u[:, :left_rank].T @ right_u[:, :right_rank]
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    return _clip_unit(float(singular_values[0])) if singular_values.size else 0.0


def _component_group(name: str) -> str | None:
    if name.startswith("observation:"):
        return "observation"
    if name.startswith("surrogate:"):
        return "surrogate"
    if name.startswith("measurement:"):
        return "measurement"
    if name.startswith("discrepancy:") or name.startswith("model_discrepancy"):
        return "model_discrepancy"
    return None


@dataclass(frozen=True)
class DiscrepancyIdentifiabilityThresholds:
    warning_discrepancy_rms_over_response: float = 0.05
    fail_discrepancy_rms_over_response: float = 0.15
    warning_discrepancy_abs_over_total_sigma_mean: float = 0.5
    fail_discrepancy_abs_over_total_sigma_mean: float = 1.0
    warning_explained_residual_share: float = 0.30
    fail_explained_residual_share: float = 0.60
    warning_parameter_shift_sigma: float = 0.5
    fail_parameter_shift_sigma: float = 1.0
    warning_parameter_shift_relative: float = 0.05
    fail_parameter_shift_relative: float = 0.05
    fail_parameter_shift_requires_discrepancy_rms_over_response: float = 0.03
    warning_max_abs_theta_beta_correlation: float = 0.60
    fail_max_abs_theta_beta_correlation: float = 0.80
    warning_theta_beta_canonical_correlation: float = 0.75
    fail_theta_beta_canonical_correlation: float = 0.90
    warning_model_discrepancy_trace_share: float = 0.25
    fail_model_discrepancy_trace_share: float = 0.50
    null_fail_discrepancy_rms_over_response: float = 0.03
    null_fail_explained_residual_share: float = 0.15
    warning_active_coefficient_count: int = 1
    warning_basis_condition_number: float = 1.0e8
    warning_basis_rank_fraction: float = 0.50

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if field_name == "warning_active_coefficient_count":
                if int(value) < 0:
                    raise ValueError("warning_active_coefficient_count must be >= 0.")
                object.__setattr__(self, field_name, int(value))
            else:
                object.__setattr__(self, field_name, _nonnegative_float(value, field_name))

    def as_dict(self) -> dict[str, float | int]:
        return {key: value for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class DiscrepancyIdentifiabilityInputs:
    observations: tuple[float, ...]
    predictions_without_discrepancy: tuple[float, ...]
    discrepancy_mean: tuple[float, ...]
    predictions_with_discrepancy: tuple[float, ...] | None = None
    basis: tuple[tuple[float, ...], ...] | None = None
    basis_names: tuple[str, ...] | None = None
    parameter_names: tuple[str, ...] = ()
    parameters_without_discrepancy: tuple[float, ...] = ()
    parameters_with_discrepancy: tuple[float, ...] = ()
    parameter_posterior_sd_without_discrepancy: tuple[float, ...] | None = None
    parameter_prior_sd: tuple[float, ...] | None = None
    parameter_sensitivities: Mapping[str, Sequence[float]] | None = None
    theta_beta_correlation: tuple[tuple[float, ...], ...] | None = None
    coefficient_names: tuple[str, ...] | None = None
    coefficient_prior_sd: tuple[float, ...] | None = None
    coefficient_posterior_mean: tuple[float, ...] | None = None
    coefficient_posterior_sd: tuple[float, ...] | None = None
    covariance_components: Mapping[str, Sequence[Sequence[float]]] | None = None
    total_covariance: tuple[tuple[float, ...], ...] | None = None
    discrepancy_enabled: bool = True
    discrepancy_opt_in: bool = False
    expect_discrepancy: bool = True
    negative_control: bool = False
    fixture_id: str = "fixture"
    required_covariance_groups: tuple[str, ...] = ("observation", "surrogate", "model_discrepancy")

    def __post_init__(self) -> None:
        observations = _finite_vector(self.observations, "observations")
        point_count = len(observations)
        predictions_without = _finite_vector(self.predictions_without_discrepancy, "predictions_without_discrepancy")
        discrepancy_mean = _finite_vector(self.discrepancy_mean, "discrepancy_mean")
        if len(predictions_without) != point_count:
            raise ValueError("predictions_without_discrepancy length must match observations.")
        if len(discrepancy_mean) != point_count:
            raise ValueError("discrepancy_mean length must match observations.")
        predictions_with = predictions_without if self.predictions_with_discrepancy is None else _finite_vector(
            self.predictions_with_discrepancy,
            "predictions_with_discrepancy",
        )
        if len(predictions_with) != point_count:
            raise ValueError("predictions_with_discrepancy length must match observations.")

        basis_tuple = None
        basis_rank = 0
        if self.basis is not None:
            basis = _coerce_matrix(self.basis, "basis", rows=point_count)
            if basis.shape[1] < 1:
                raise ValueError("basis must contain at least one column when provided.")
            basis_tuple = _as_tuple_matrix(basis)
            basis_rank = int(basis.shape[1])

        basis_names = None
        if self.basis_names is not None:
            basis_names = tuple(str(name).strip() for name in self.basis_names)
            if any(not name for name in basis_names):
                raise ValueError("basis_names must be non-empty.")
            if len(set(basis_names)) != len(basis_names):
                raise ValueError("basis_names must be unique.")
            if basis_rank and len(basis_names) != basis_rank:
                raise ValueError("basis_names count must match basis rank.")

        parameters_without = _finite_vector(
            self.parameters_without_discrepancy,
            "parameters_without_discrepancy",
            allow_empty=True,
        )
        parameters_with = _finite_vector(self.parameters_with_discrepancy, "parameters_with_discrepancy", allow_empty=True)
        if len(parameters_without) != len(parameters_with):
            raise ValueError("parameters_without_discrepancy and parameters_with_discrepancy lengths must match.")
        parameter_names = tuple(str(name).strip() for name in self.parameter_names)
        if not parameter_names and parameters_without:
            parameter_names = tuple(f"theta_{index}" for index in range(len(parameters_without)))
        if any(not name for name in parameter_names):
            raise ValueError("parameter_names must be non-empty.")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_names must be unique.")
        if len(parameter_names) != len(parameters_without):
            raise ValueError("parameter_names count must match parameter count.")

        posterior_sd = None
        if self.parameter_posterior_sd_without_discrepancy is not None:
            posterior_sd = tuple(_positive_float(value, "parameter_posterior_sd_without_discrepancy") for value in self.parameter_posterior_sd_without_discrepancy)
            if len(posterior_sd) != len(parameters_without):
                raise ValueError("parameter_posterior_sd_without_discrepancy count must match parameter count.")
        prior_sd = None
        if self.parameter_prior_sd is not None:
            prior_sd = tuple(_positive_float(value, "parameter_prior_sd") for value in self.parameter_prior_sd)
            if len(prior_sd) != len(parameters_without):
                raise ValueError("parameter_prior_sd count must match parameter count.")

        sensitivities = None
        if self.parameter_sensitivities is not None:
            sensitivities = {str(name): _finite_vector(values, f"parameter_sensitivities[{name}]") for name, values in self.parameter_sensitivities.items()}
            for name, values in sensitivities.items():
                if len(values) != point_count:
                    raise ValueError(f"parameter sensitivity '{name}' length must match observations.")

        theta_beta_correlation = None
        if self.theta_beta_correlation is not None:
            correlation = _coerce_matrix(self.theta_beta_correlation, "theta_beta_correlation")
            if parameters_without and correlation.shape[0] != len(parameters_without):
                raise ValueError("theta_beta_correlation row count must match parameter count.")
            if basis_rank and correlation.shape[1] != basis_rank:
                raise ValueError("theta_beta_correlation column count must match basis rank.")
            if np.max(np.abs(correlation)) > 1.0 + 1e-10:
                raise ValueError("theta_beta_correlation values must be within [-1, 1].")
            theta_beta_correlation = _as_tuple_matrix(correlation)

        coefficient_count = basis_rank
        coefficient_names = None
        if self.coefficient_names is not None:
            coefficient_names = tuple(str(name).strip() for name in self.coefficient_names)
            if any(not name for name in coefficient_names):
                raise ValueError("coefficient_names must be non-empty.")
            if len(set(coefficient_names)) != len(coefficient_names):
                raise ValueError("coefficient_names must be unique.")
            coefficient_count = len(coefficient_names)
        elif basis_names is not None:
            coefficient_names = basis_names
            coefficient_count = len(basis_names)
        elif coefficient_count:
            coefficient_names = tuple(f"beta_{index}" for index in range(coefficient_count))

        coefficient_prior_sd = None
        if self.coefficient_prior_sd is not None:
            coefficient_prior_sd = tuple(_nonnegative_float(value, "coefficient_prior_sd") for value in self.coefficient_prior_sd)
            coefficient_count = len(coefficient_prior_sd) if coefficient_count == 0 else coefficient_count
            if len(coefficient_prior_sd) != coefficient_count:
                raise ValueError("coefficient_prior_sd count must match coefficient count.")
        coefficient_posterior_mean = None
        if self.coefficient_posterior_mean is not None:
            coefficient_posterior_mean = _finite_vector(self.coefficient_posterior_mean, "coefficient_posterior_mean", allow_empty=True)
            coefficient_count = len(coefficient_posterior_mean) if coefficient_count == 0 else coefficient_count
            if len(coefficient_posterior_mean) != coefficient_count:
                raise ValueError("coefficient_posterior_mean count must match coefficient count.")
        coefficient_posterior_sd = None
        if self.coefficient_posterior_sd is not None:
            coefficient_posterior_sd = tuple(_nonnegative_float(value, "coefficient_posterior_sd") for value in self.coefficient_posterior_sd)
            coefficient_count = len(coefficient_posterior_sd) if coefficient_count == 0 else coefficient_count
            if len(coefficient_posterior_sd) != coefficient_count:
                raise ValueError("coefficient_posterior_sd count must match coefficient count.")
        if coefficient_names is None and coefficient_count:
            coefficient_names = tuple(f"beta_{index}" for index in range(coefficient_count))

        covariance_components = None
        if self.covariance_components is not None:
            covariance_components = {
                str(name): _as_tuple_matrix(_coerce_square_matrix(matrix, f"covariance_components[{name}]", size=point_count))
                for name, matrix in self.covariance_components.items()
            }
            if not covariance_components:
                raise ValueError("covariance_components must not be empty when provided.")
        total_covariance = None
        if self.total_covariance is not None:
            total_covariance = _as_tuple_matrix(_coerce_square_matrix(self.total_covariance, "total_covariance", size=point_count))

        required_covariance_groups = tuple(str(group).strip() for group in self.required_covariance_groups)
        if any(not group for group in required_covariance_groups):
            raise ValueError("required_covariance_groups must be non-empty.")

        fixture_id = str(self.fixture_id).strip()
        if not fixture_id:
            raise ValueError("fixture_id must be non-empty.")

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "predictions_without_discrepancy", predictions_without)
        object.__setattr__(self, "discrepancy_mean", discrepancy_mean)
        object.__setattr__(self, "predictions_with_discrepancy", predictions_with)
        object.__setattr__(self, "basis", basis_tuple)
        object.__setattr__(self, "basis_names", basis_names)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "parameters_without_discrepancy", parameters_without)
        object.__setattr__(self, "parameters_with_discrepancy", parameters_with)
        object.__setattr__(self, "parameter_posterior_sd_without_discrepancy", posterior_sd)
        object.__setattr__(self, "parameter_prior_sd", prior_sd)
        object.__setattr__(self, "parameter_sensitivities", sensitivities)
        object.__setattr__(self, "theta_beta_correlation", theta_beta_correlation)
        object.__setattr__(self, "coefficient_names", coefficient_names)
        object.__setattr__(self, "coefficient_prior_sd", coefficient_prior_sd)
        object.__setattr__(self, "coefficient_posterior_mean", coefficient_posterior_mean)
        object.__setattr__(self, "coefficient_posterior_sd", coefficient_posterior_sd)
        object.__setattr__(self, "covariance_components", covariance_components)
        object.__setattr__(self, "total_covariance", total_covariance)
        object.__setattr__(self, "discrepancy_enabled", bool(self.discrepancy_enabled))
        object.__setattr__(self, "discrepancy_opt_in", bool(self.discrepancy_opt_in))
        object.__setattr__(self, "expect_discrepancy", bool(self.expect_discrepancy))
        object.__setattr__(self, "negative_control", bool(self.negative_control))
        object.__setattr__(self, "fixture_id", fixture_id)
        object.__setattr__(self, "required_covariance_groups", required_covariance_groups)


@dataclass(frozen=True)
class DiscrepancyIdentifiabilityResult:
    fixture_id: str
    gate_status: str
    metrics: Mapping[str, float | int | bool | None]
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    parameter_shift: Mapping[str, Mapping[str, float | None]]
    coefficient_shrinkage: Mapping[str, Mapping[str, float | int | None]]
    covariance_component_trace_shares: Mapping[str, float | None]
    covariance_group_trace_shares: Mapping[str, float | None]
    theta_beta_correlation: tuple[tuple[float, ...], ...]
    summary: Mapping[str, Any]


def evaluate_discrepancy_identifiability(
    inputs: DiscrepancyIdentifiabilityInputs,
    thresholds: DiscrepancyIdentifiabilityThresholds | None = None,
) -> DiscrepancyIdentifiabilityResult:
    thresholds = DiscrepancyIdentifiabilityThresholds() if thresholds is None else thresholds
    observations = np.asarray(inputs.observations, dtype=float)
    predictions_without = np.asarray(inputs.predictions_without_discrepancy, dtype=float)
    predictions_with = np.asarray(inputs.predictions_with_discrepancy, dtype=float)
    discrepancy = np.asarray(inputs.discrepancy_mean, dtype=float)
    residual_without = observations - predictions_without
    residual_after = observations - predictions_with - discrepancy
    response_rms = _rms(observations)
    residual_without_rms = _rms(residual_without)
    residual_after_rms = _rms(residual_after)
    discrepancy_rms = _rms(discrepancy)
    residual_sum_without = float(np.dot(residual_without, residual_without))
    residual_sum_after = float(np.dot(residual_after, residual_after))
    explained_share = 0.0 if residual_sum_without <= _EPSILON else float(1.0 - residual_sum_after / residual_sum_without)

    parameter_shift = _parameter_shift_summary(inputs)
    coefficient_shrinkage = _coefficient_shrinkage_summary(inputs)
    covariance_summary = _covariance_summary(inputs)
    correlation, canonical_corr, correlation_names = _theta_beta_correlation_summary(inputs)

    max_shift_sigma = _max_optional(parameter["shift_sigma"] for parameter in parameter_shift.values())
    max_shift_relative = _max_optional(parameter["shift_relative"] for parameter in parameter_shift.values())
    max_abs_corr = float(np.max(np.abs(correlation))) if correlation.size else 0.0
    active_count = int(sum(1 for coefficient in coefficient_shrinkage.values() if coefficient.get("active") == 1))
    model_discrepancy_trace_share = covariance_summary["group_trace_shares"].get("model_discrepancy")

    total_sigma_stats = _discrepancy_total_sigma_stats(discrepancy, covariance_summary.get("total_covariance"))
    basis_metrics = _basis_metrics(inputs)
    warnings: list[str] = []
    failures: list[str] = []

    discrepancy_rms_over_response = _safe_ratio(discrepancy_rms, response_rms)
    discrepancy_rms_over_residual_without = _safe_ratio(discrepancy_rms, residual_without_rms)
    if inputs.discrepancy_enabled and not inputs.discrepancy_opt_in:
        failures.append("discrepancy is enabled without explicit opt-in.")
    if not inputs.discrepancy_enabled and discrepancy_rms > _EPSILON:
        failures.append("discrepancy is marked disabled but supplies a nonzero discrepancy mean.")
    if discrepancy_rms_over_response > thresholds.fail_discrepancy_rms_over_response:
        failures.append("discrepancy RMS exceeds the response-scale fail threshold.")
    elif discrepancy_rms_over_response > thresholds.warning_discrepancy_rms_over_response:
        warnings.append("discrepancy RMS exceeds the response-scale warning threshold.")
    if total_sigma_stats["mean_abs_over_sigma"] is not None:
        if total_sigma_stats["mean_abs_over_sigma"] > thresholds.fail_discrepancy_abs_over_total_sigma_mean:
            failures.append("discrepancy magnitude dominates total predictive standard deviation.")
        elif total_sigma_stats["mean_abs_over_sigma"] > thresholds.warning_discrepancy_abs_over_total_sigma_mean:
            warnings.append("discrepancy magnitude is large relative to total predictive standard deviation.")
    if explained_share > thresholds.fail_explained_residual_share and inputs.negative_control:
        failures.append("negative-control discrepancy explains too much residual structure.")
    elif explained_share > thresholds.warning_explained_residual_share:
        warnings.append("discrepancy explains a large share of residual variance.")
    if max_shift_sigma is not None:
        if (
            max_shift_sigma > thresholds.fail_parameter_shift_sigma
            and discrepancy_rms_over_response > thresholds.fail_parameter_shift_requires_discrepancy_rms_over_response
        ):
            failures.append("physical parameter shift exceeds the posterior-SD fail threshold with nontrivial discrepancy.")
        elif max_shift_sigma > thresholds.warning_parameter_shift_sigma:
            warnings.append("physical parameter shift exceeds the posterior-SD warning threshold.")
    if max_shift_relative is not None:
        if (
            max_shift_relative > thresholds.fail_parameter_shift_relative
            and discrepancy_rms_over_response > thresholds.fail_parameter_shift_requires_discrepancy_rms_over_response
        ):
            failures.append("physical parameter shift exceeds the relative fail threshold with nontrivial discrepancy.")
        elif max_shift_relative > thresholds.warning_parameter_shift_relative:
            warnings.append("physical parameter shift exceeds the relative warning threshold.")
    if max_abs_corr > thresholds.fail_max_abs_theta_beta_correlation:
        failures.append("physical/discrepancy correlation exceeds the fail threshold.")
    elif max_abs_corr > thresholds.warning_max_abs_theta_beta_correlation:
        warnings.append("physical/discrepancy correlation exceeds the warning threshold.")
    if canonical_corr is not None:
        if canonical_corr > thresholds.fail_theta_beta_canonical_correlation:
            failures.append("physical/discrepancy canonical correlation exceeds the fail threshold.")
        elif canonical_corr > thresholds.warning_theta_beta_canonical_correlation:
            warnings.append("physical/discrepancy canonical correlation exceeds the warning threshold.")
    if model_discrepancy_trace_share is not None:
        if model_discrepancy_trace_share > thresholds.fail_model_discrepancy_trace_share and inputs.negative_control:
            failures.append("negative-control model-discrepancy covariance trace share is too large.")
        elif model_discrepancy_trace_share > thresholds.warning_model_discrepancy_trace_share:
            warnings.append("model-discrepancy covariance trace share exceeds the warning threshold.")
    if not inputs.expect_discrepancy:
        if discrepancy_rms_over_response > thresholds.null_fail_discrepancy_rms_over_response:
            failures.append("no-discrepancy fixture has nonzero discrepancy above the null fail threshold.")
        if active_count > 0:
            failures.append("no-discrepancy fixture has active discrepancy coefficients.")
        if explained_share > thresholds.null_fail_explained_residual_share:
            failures.append("no-discrepancy fixture lets discrepancy explain residual variance.")
    if active_count > thresholds.warning_active_coefficient_count and not inputs.expect_discrepancy:
        warnings.append("too many active discrepancy coefficients in a no-discrepancy fixture.")
    if basis_metrics["basis_rank_deficient"]:
        warnings.append("discrepancy basis is rank deficient.")
    if basis_metrics["basis_condition_number"] is not None and basis_metrics["basis_condition_number"] > thresholds.warning_basis_condition_number:
        warnings.append("discrepancy basis is near singular.")
    if basis_metrics["basis_rank_fraction"] is not None and basis_metrics["basis_rank_fraction"] > thresholds.warning_basis_rank_fraction:
        warnings.append("discrepancy rank is large relative to observation count.")
    _check_covariance_group_coverage(inputs, covariance_summary, failures)
    if inputs.negative_control and not warnings and not failures:
        failures.append("negative-control fixture passed without identifiability warning or failure.")

    gate_status = "fail" if failures else "warn" if warnings else "pass"
    metrics: dict[str, float | int | bool | None] = {
        "response_rms": response_rms,
        "residual_rms_without_discrepancy": residual_without_rms,
        "residual_rms_after_discrepancy": residual_after_rms,
        "discrepancy_rms": discrepancy_rms,
        "discrepancy_rms_over_response": discrepancy_rms_over_response,
        "discrepancy_rms_over_residual_without": discrepancy_rms_over_residual_without,
        "discrepancy_explained_residual_share": explained_share,
        "max_parameter_shift_sigma": max_shift_sigma,
        "max_parameter_shift_relative": max_shift_relative,
        "max_abs_theta_beta_correlation": max_abs_corr,
        "theta_beta_canonical_correlation": canonical_corr,
        "active_discrepancy_coefficients": active_count,
        "model_discrepancy_trace_share": model_discrepancy_trace_share,
        "discrepancy_abs_over_total_sigma_mean": total_sigma_stats["mean_abs_over_sigma"],
        "discrepancy_abs_over_total_sigma_max": total_sigma_stats["max_abs_over_sigma"],
        "basis_rank": basis_metrics["basis_rank"],
        "basis_columns": basis_metrics["basis_columns"],
        "basis_rank_fraction": basis_metrics["basis_rank_fraction"],
        "basis_condition_number": basis_metrics["basis_condition_number"],
        "discrepancy_enabled": inputs.discrepancy_enabled,
        "discrepancy_opt_in": inputs.discrepancy_opt_in,
        "expect_discrepancy": inputs.expect_discrepancy,
        "negative_control": inputs.negative_control,
    }
    summary = {
        "schema_version": 1,
        "fixture_id": inputs.fixture_id,
        "gate_status": gate_status,
        "metrics": metrics,
        "warnings": tuple(warnings),
        "failures": tuple(failures),
        "thresholds": thresholds.as_dict(),
        "parameter_shift": parameter_shift,
        "coefficient_shrinkage": coefficient_shrinkage,
        "theta_beta_correlation": _as_tuple_matrix(correlation),
        "theta_beta_correlation_names": correlation_names,
        "covariance_component_trace_shares": covariance_summary["component_trace_shares"],
        "covariance_group_trace_shares": covariance_summary["group_trace_shares"],
        "covariance_component_mean_diagonal_shares": covariance_summary["component_mean_diagonal_shares"],
        "required_covariance_groups": inputs.required_covariance_groups,
    }
    return DiscrepancyIdentifiabilityResult(
        fixture_id=inputs.fixture_id,
        gate_status=gate_status,
        metrics=metrics,
        warnings=tuple(warnings),
        failures=tuple(failures),
        parameter_shift=parameter_shift,
        coefficient_shrinkage=coefficient_shrinkage,
        covariance_component_trace_shares=covariance_summary["component_trace_shares"],
        covariance_group_trace_shares=covariance_summary["group_trace_shares"],
        theta_beta_correlation=_as_tuple_matrix(correlation),
        summary=summary,
    )


def _parameter_shift_summary(inputs: DiscrepancyIdentifiabilityInputs) -> dict[str, dict[str, float | None]]:
    names = inputs.parameter_names
    without = np.asarray(inputs.parameters_without_discrepancy, dtype=float)
    with_discrepancy = np.asarray(inputs.parameters_with_discrepancy, dtype=float)
    posterior_sd = None if inputs.parameter_posterior_sd_without_discrepancy is None else np.asarray(inputs.parameter_posterior_sd_without_discrepancy, dtype=float)
    prior_sd = None if inputs.parameter_prior_sd is None else np.asarray(inputs.parameter_prior_sd, dtype=float)
    summary: dict[str, dict[str, float | None]] = {}
    for index, name in enumerate(names):
        shift = abs(float(with_discrepancy[index] - without[index]))
        relative = _safe_ratio(shift, max(abs(float(without[index])), _EPSILON))
        summary[name] = {
            "without_discrepancy": float(without[index]),
            "with_discrepancy": float(with_discrepancy[index]),
            "absolute_shift": shift,
            "shift_sigma": None if posterior_sd is None else _safe_ratio(shift, float(posterior_sd[index])),
            "shift_prior_sd": None if prior_sd is None else _safe_ratio(shift, float(prior_sd[index])),
            "shift_relative": relative,
        }
    return summary


def _coefficient_shrinkage_summary(inputs: DiscrepancyIdentifiabilityInputs) -> dict[str, dict[str, float | int | None]]:
    names = inputs.coefficient_names or ()
    prior_sd = None if inputs.coefficient_prior_sd is None else np.asarray(inputs.coefficient_prior_sd, dtype=float)
    posterior_mean = None if inputs.coefficient_posterior_mean is None else np.asarray(inputs.coefficient_posterior_mean, dtype=float)
    posterior_sd = None if inputs.coefficient_posterior_sd is None else np.asarray(inputs.coefficient_posterior_sd, dtype=float)
    summary: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(names):
        mean = None if posterior_mean is None else float(posterior_mean[index])
        sd = None if posterior_sd is None else float(posterior_sd[index])
        prior = None if prior_sd is None else float(prior_sd[index])
        active = 0
        if mean is not None and sd is not None:
            active = int(abs(mean) > 2.0 * max(sd, _EPSILON))
        summary[name] = {
            "posterior_mean": mean,
            "posterior_sd": sd,
            "prior_sd": prior,
            "posterior_sd_over_prior_sd": None if prior is None or sd is None else _safe_ratio(sd, prior),
            "active": active,
        }
    return summary


def _covariance_summary(inputs: DiscrepancyIdentifiabilityInputs) -> dict[str, Any]:
    components = {}
    if inputs.covariance_components is not None:
        components = {name: np.asarray(matrix, dtype=float) for name, matrix in inputs.covariance_components.items()}
    if inputs.total_covariance is not None:
        total = np.asarray(inputs.total_covariance, dtype=float)
    elif components:
        total = sum(components.values(), np.zeros_like(next(iter(components.values()))))
    else:
        total = None
    component_trace_shares: dict[str, float | None] = {}
    component_mean_diagonal_shares: dict[str, float | None] = {}
    group_traces: dict[str, float] = {}
    trace_total = None if total is None else float(np.trace(total))
    diagonal_total = None if total is None else np.diag(total)
    for name, matrix in components.items():
        trace = float(np.trace(matrix))
        component_trace_shares[name] = None if trace_total is None else _safe_ratio(trace, trace_total)
        if diagonal_total is None:
            component_mean_diagonal_shares[name] = None
        else:
            diagonal = np.diag(matrix)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.divide(diagonal, diagonal_total, out=np.zeros_like(diagonal), where=np.abs(diagonal_total) > _EPSILON)
            component_mean_diagonal_shares[name] = float(np.mean(share))
        group = _component_group(name)
        if group is not None:
            group_traces[group] = group_traces.get(group, 0.0) + trace
    group_trace_shares = {group: None if trace_total is None else _safe_ratio(trace, trace_total) for group, trace in group_traces.items()}
    return {
        "total_covariance": total,
        "component_trace_shares": component_trace_shares,
        "component_mean_diagonal_shares": component_mean_diagonal_shares,
        "group_trace_shares": group_trace_shares,
    }


def _theta_beta_correlation_summary(inputs: DiscrepancyIdentifiabilityInputs) -> tuple[np.ndarray, float | None, dict[str, tuple[str, ...]]]:
    if inputs.theta_beta_correlation is not None:
        correlation = np.asarray(inputs.theta_beta_correlation, dtype=float)
        canonical = _clip_unit(float(np.linalg.svd(correlation, compute_uv=False)[0])) if correlation.size else 0.0
        return correlation, canonical, {
            "parameter_names": inputs.parameter_names,
            "coefficient_names": inputs.coefficient_names or tuple(f"beta_{index}" for index in range(correlation.shape[1])),
            "source": ("explicit",),
        }
    if inputs.parameter_sensitivities is None or inputs.basis is None:
        return np.zeros((0, 0), dtype=float), None, {"parameter_names": (), "coefficient_names": (), "source": ("missing",)}
    parameter_names = tuple(inputs.parameter_sensitivities.keys())
    sensitivity = np.column_stack([np.asarray(inputs.parameter_sensitivities[name], dtype=float) for name in parameter_names])
    basis = np.asarray(inputs.basis, dtype=float)
    correlation = _column_correlations(sensitivity, basis)
    canonical = _first_canonical_correlation(sensitivity, basis)
    return correlation, canonical, {
        "parameter_names": parameter_names,
        "coefficient_names": inputs.coefficient_names or tuple(f"beta_{index}" for index in range(basis.shape[1])),
        "source": ("sensitivity_basis",),
    }


def _discrepancy_total_sigma_stats(discrepancy: np.ndarray, total_covariance: Any) -> dict[str, float | None]:
    if total_covariance is None:
        return {"mean_abs_over_sigma": None, "max_abs_over_sigma": None}
    diagonal = np.diag(np.asarray(total_covariance, dtype=float))
    sigma = np.sqrt(np.maximum(diagonal, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(np.abs(discrepancy), sigma, out=np.full_like(discrepancy, np.inf), where=sigma > _EPSILON)
    return {"mean_abs_over_sigma": float(np.mean(ratios)), "max_abs_over_sigma": float(np.max(ratios))}


def _basis_metrics(inputs: DiscrepancyIdentifiabilityInputs) -> dict[str, float | int | bool | None]:
    if inputs.basis is None:
        return {
            "basis_rank": 0,
            "basis_columns": 0,
            "basis_rank_fraction": None,
            "basis_condition_number": None,
            "basis_rank_deficient": False,
        }
    basis = np.asarray(inputs.basis, dtype=float)
    rank = int(np.linalg.matrix_rank(basis))
    condition_number = None if rank == 0 else _finite_or_none(np.linalg.cond(basis))
    return {
        "basis_rank": rank,
        "basis_columns": int(basis.shape[1]),
        "basis_rank_fraction": float(basis.shape[1] / basis.shape[0]),
        "basis_condition_number": condition_number,
        "basis_rank_deficient": rank < basis.shape[1],
    }


def _check_covariance_group_coverage(inputs: DiscrepancyIdentifiabilityInputs, covariance_summary: Mapping[str, Any], failures: list[str]) -> None:
    if not inputs.required_covariance_groups:
        return
    if inputs.covariance_components is None:
        failures.append("required covariance component report is missing.")
        return
    present = set(covariance_summary["group_trace_shares"].keys())
    missing = [group for group in inputs.required_covariance_groups if group not in present]
    if missing:
        failures.append(f"required covariance component groups are missing: {', '.join(missing)}.")


def _max_optional(values: Sequence[float | None]) -> float | None:
    finite_values = [float(value) for value in values if value is not None]
    return max(finite_values) if finite_values else None


__all__ = [
    "DiscrepancyIdentifiabilityInputs",
    "DiscrepancyIdentifiabilityResult",
    "DiscrepancyIdentifiabilityThresholds",
    "evaluate_discrepancy_identifiability",
]
