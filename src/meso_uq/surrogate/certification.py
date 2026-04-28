from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_ACCEPTANCE_UPPER_BOUND = 0.10
DEFAULT_BOOTSTRAP_STATISTIC = "mean"


@dataclass(frozen=True)
class BootstrapInterval:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_resamples: int
    random_seed: int
    sample_size: int
    statistic: str


@dataclass(frozen=True)
class AcceptanceDecision:
    ci_contains_zero: bool
    upper_bound_pass: bool
    passed: bool


@dataclass(frozen=True)
class CertificationSummary:
    per_metric: pd.DataFrame
    per_diameter: pd.DataFrame
    overall: dict[str, object]


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}.")


def _resolve_bootstrap_statistic(statistic: str) -> str:
    resolved = statistic.strip().lower()
    if resolved not in {"mean", "median"}:
        raise ValueError("bootstrap statistic must be 'mean' or 'median'.")
    return resolved


def _apply_bootstrap_statistic(values: np.ndarray, statistic: str, *, axis: int) -> np.ndarray:
    if statistic == "mean":
        return np.mean(values, axis=axis)
    return np.median(values, axis=axis)


def compute_relative_deltas(
    dnn_values: Sequence[float] | np.ndarray,
    bnn_values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    dnn = np.asarray(dnn_values, dtype=np.float64)
    bnn = np.asarray(bnn_values, dtype=np.float64)
    if dnn.shape != bnn.shape:
        raise ValueError("dnn_values and bnn_values must have the same shape.")
    if dnn.ndim != 1:
        raise ValueError("dnn_values and bnn_values must be one-dimensional.")
    if not np.isfinite(dnn).all() or not np.isfinite(bnn).all():
        raise ValueError("Relative deltas require finite DNN and BNN values.")

    zero_mask = dnn == 0.0
    mismatched_zero = zero_mask & (bnn != 0.0)
    if np.any(mismatched_zero):
        raise ValueError("Cannot compute relative delta when a DNN metric is zero and the paired BNN metric is non-zero.")

    deltas = np.empty_like(dnn, dtype=np.float64)
    deltas[zero_mask] = 0.0
    non_zero_mask = ~zero_mask
    deltas[non_zero_mask] = (bnn[non_zero_mask] - dnn[non_zero_mask]) / dnn[non_zero_mask]
    return deltas


def paired_bootstrap_ci(
    paired_deltas: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = 0,
    statistic: str = DEFAULT_BOOTSTRAP_STATISTIC,
) -> BootstrapInterval:
    deltas = np.asarray(paired_deltas, dtype=np.float64)
    if deltas.ndim != 1:
        raise ValueError("paired_deltas must be one-dimensional.")
    if deltas.size == 0:
        raise ValueError("paired_deltas must be non-empty.")
    if not np.isfinite(deltas).all():
        raise ValueError("paired_deltas must be finite.")
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")

    resolved_statistic = _resolve_bootstrap_statistic(statistic)
    point_estimate = float(_apply_bootstrap_statistic(deltas, resolved_statistic, axis=0))

    rng = np.random.default_rng(int(random_seed))
    sample_indices = rng.integers(0, deltas.size, size=(int(n_resamples), deltas.size))
    resampled = deltas[sample_indices]
    bootstrap_stats = _apply_bootstrap_statistic(resampled, resolved_statistic, axis=1)

    alpha = 1.0 - float(confidence_level)
    ci_lower, ci_upper = np.quantile(bootstrap_stats, [alpha / 2.0, 1.0 - (alpha / 2.0)])
    return BootstrapInterval(
        point_estimate=point_estimate,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=float(confidence_level),
        n_resamples=int(n_resamples),
        random_seed=int(random_seed),
        sample_size=int(deltas.size),
        statistic=resolved_statistic,
    )


def evaluate_acceptance_rule(
    ci_lower: float,
    ci_upper: float,
    *,
    acceptance_upper_bound: float = DEFAULT_ACCEPTANCE_UPPER_BOUND,
) -> AcceptanceDecision:
    lower = float(ci_lower)
    upper = float(ci_upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Acceptance rule requires finite confidence interval bounds.")
    if lower > upper:
        raise ValueError("Confidence interval lower bound cannot exceed upper bound.")

    ci_contains_zero = bool(lower <= 0.0 <= upper)
    upper_bound_pass = bool(upper <= float(acceptance_upper_bound))
    return AcceptanceDecision(
        ci_contains_zero=ci_contains_zero,
        upper_bound_pass=upper_bound_pass,
        passed=bool(ci_contains_zero and upper_bound_pass),
    )


def build_paired_metric_table(
    metrics_df: pd.DataFrame,
    *,
    metric_cols: Sequence[str],
    group_cols: Sequence[str] = ("diameter_um", "seed"),
    family_col: str = "surrogate_family",
    dnn_label: str = "dnn",
    bnn_label: str = "bnn",
) -> pd.DataFrame:
    metric_names = [str(column) for column in metric_cols]
    if not metric_names:
        raise ValueError("metric_cols must be non-empty.")

    _require_columns(metrics_df, [*group_cols, family_col, *metric_names], label="metrics_df")
    if metrics_df.duplicated([*group_cols, family_col]).any():
        raise ValueError("metrics_df must contain at most one row per group/family pair.")

    long_df = metrics_df[[*group_cols, family_col, *metric_names]].melt(
        id_vars=[*group_cols, family_col],
        value_vars=metric_names,
        var_name="metric_name",
        value_name="metric_value",
    )
    paired = (
        long_df.pivot(index=[*group_cols, "metric_name"], columns=family_col, values="metric_value")
        .reset_index()
    )
    paired.columns.name = None

    missing_family_columns = [label for label in (dnn_label, bnn_label) if label not in paired.columns]
    if missing_family_columns:
        raise ValueError(f"metrics_df is missing required surrogate families: {missing_family_columns}.")
    if paired[[dnn_label, bnn_label]].isna().any().any():
        raise ValueError("metrics_df must contain complete paired DNN/BNN metric rows for every group.")

    paired = paired.rename(columns={dnn_label: "dnn_value", bnn_label: "bnn_value"})
    paired["delta_rel"] = compute_relative_deltas(
        paired["dnn_value"].to_numpy(dtype=np.float64),
        paired["bnn_value"].to_numpy(dtype=np.float64),
    )
    return paired.sort_values([*group_cols, "metric_name"]).reset_index(drop=True)


def certify_paired_metrics(
    paired_metrics: pd.DataFrame,
    *,
    diameter_col: str = "diameter_um",
    metric_col: str = "metric_name",
    delta_col: str = "delta_rel",
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: int = 0,
    statistic: str = DEFAULT_BOOTSTRAP_STATISTIC,
    acceptance_upper_bound: float = DEFAULT_ACCEPTANCE_UPPER_BOUND,
) -> pd.DataFrame:
    _require_columns(paired_metrics, [diameter_col, metric_col, delta_col], label="paired_metrics")

    rows: list[dict[str, object]] = []
    for (diameter, metric_name), group_df in paired_metrics.groupby([diameter_col, metric_col], sort=True):
        interval = paired_bootstrap_ci(
            group_df[delta_col].to_numpy(dtype=np.float64),
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_seed=random_seed,
            statistic=statistic,
        )
        decision = evaluate_acceptance_rule(
            interval.ci_lower,
            interval.ci_upper,
            acceptance_upper_bound=acceptance_upper_bound,
        )
        rows.append(
            {
                diameter_col: diameter,
                metric_col: metric_name,
                "n_pairs": int(len(group_df)),
                "delta_rel_point_estimate": interval.point_estimate,
                "ci_lower": interval.ci_lower,
                "ci_upper": interval.ci_upper,
                "confidence_level": interval.confidence_level,
                "bootstrap_resamples": interval.n_resamples,
                "bootstrap_seed": interval.random_seed,
                "bootstrap_statistic": interval.statistic,
                "ci_contains_zero": decision.ci_contains_zero,
                "upper_bound_pass": decision.upper_bound_pass,
                "passed": decision.passed,
                "acceptance_upper_bound": float(acceptance_upper_bound),
            }
        )

    columns = [
        diameter_col,
        metric_col,
        "n_pairs",
        "delta_rel_point_estimate",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "bootstrap_resamples",
        "bootstrap_seed",
        "bootstrap_statistic",
        "ci_contains_zero",
        "upper_bound_pass",
        "passed",
        "acceptance_upper_bound",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values([diameter_col, metric_col]).reset_index(drop=True)


def summarize_certification_by_diameter(
    certification_df: pd.DataFrame,
    *,
    diameter_col: str = "diameter_um",
    metric_col: str = "metric_name",
    passed_col: str = "passed",
) -> pd.DataFrame:
    _require_columns(certification_df, [diameter_col, metric_col, passed_col], label="certification_df")

    rows: list[dict[str, object]] = []
    for diameter, group_df in certification_df.groupby(diameter_col, sort=True):
        passed_mask = group_df[passed_col].astype(bool)
        rows.append(
            {
                diameter_col: diameter,
                "metric_count": int(len(group_df)),
                "passed_metric_count": int(passed_mask.sum()),
                "failed_metric_count": int((~passed_mask).sum()),
                "all_metrics_passed": bool(passed_mask.all()),
                "passed_metrics": tuple(group_df.loc[passed_mask, metric_col].astype(str).tolist()),
                "failed_metrics": tuple(group_df.loc[~passed_mask, metric_col].astype(str).tolist()),
            }
        )

    columns = [
        diameter_col,
        "metric_count",
        "passed_metric_count",
        "failed_metric_count",
        "all_metrics_passed",
        "passed_metrics",
        "failed_metrics",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(diameter_col).reset_index(drop=True)


def build_certification_summary(
    certification_df: pd.DataFrame,
    *,
    diameter_col: str = "diameter_um",
    metric_col: str = "metric_name",
    passed_col: str = "passed",
) -> CertificationSummary:
    per_metric = certification_df.sort_values([diameter_col, metric_col]).reset_index(drop=True).copy()
    per_diameter = summarize_certification_by_diameter(
        certification_df,
        diameter_col=diameter_col,
        metric_col=metric_col,
        passed_col=passed_col,
    )

    metric_count = int(len(per_metric))
    passed_metric_count = int(per_metric[passed_col].astype(bool).sum()) if metric_count else 0
    failed_metric_count = int(metric_count - passed_metric_count)
    diameter_count = int(len(per_diameter))
    passing_diameter_count = int(per_diameter["all_metrics_passed"].astype(bool).sum()) if diameter_count else 0
    failing_diameter_count = int(diameter_count - passing_diameter_count)
    failing_diameters = tuple(per_diameter.loc[~per_diameter["all_metrics_passed"], diameter_col].tolist())

    overall = {
        "diameter_count": diameter_count,
        "passing_diameter_count": passing_diameter_count,
        "failing_diameter_count": failing_diameter_count,
        "metric_count": metric_count,
        "passed_metric_count": passed_metric_count,
        "failed_metric_count": failed_metric_count,
        "all_diameters_passed": bool(diameter_count > 0 and failing_diameter_count == 0),
        "failing_diameters": failing_diameters,
    }
    return CertificationSummary(per_metric=per_metric, per_diameter=per_diameter, overall=overall)
