from __future__ import annotations

import pandas as pd
import pytest

from meso_uq.surrogate.certification import (
    build_certification_summary,
    build_paired_metric_table,
    certify_paired_metrics,
    compute_relative_deltas,
    evaluate_acceptance_rule,
    paired_bootstrap_ci,
    summarize_certification_by_diameter,
)


def _synthetic_metric_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_deltas = {
        ("2.1", "val_rmse"): [-0.02, 0.00, 0.03, 0.04, -0.01],
        ("2.1", "median_curve_rel_l2_pct"): [-0.05, 0.01, 0.04, 0.02, -0.02],
        ("2.9", "val_rmse"): [0.12, 0.14, 0.15, 0.13, 0.16],
        ("2.9", "median_curve_rel_l2_pct"): [-0.03, 0.02, 0.04, 0.01, -0.02],
    }
    dnn_baselines = {
        "val_rmse": 2.0,
        "median_curve_rel_l2_pct": 8.0,
    }
    for diameter in ("2.1", "2.9"):
        for seed in range(5):
            dnn_row: dict[str, object] = {
                "diameter_um": diameter,
                "seed": seed,
                "surrogate_family": "dnn",
            }
            bnn_row: dict[str, object] = {
                "diameter_um": diameter,
                "seed": seed,
                "surrogate_family": "bnn",
            }
            for metric_name, dnn_value in dnn_baselines.items():
                delta = metric_deltas[(diameter, metric_name)][seed]
                dnn_row[metric_name] = dnn_value
                bnn_row[metric_name] = dnn_value * (1.0 + delta)
            rows.extend([dnn_row, bnn_row])
    return pd.DataFrame(rows)


def test_compute_relative_deltas_handles_equal_zero_pairs() -> None:
    deltas = compute_relative_deltas([0.0, 10.0], [0.0, 11.0])
    assert deltas.tolist() == [0.0, 0.1]


def test_paired_bootstrap_ci_is_deterministic() -> None:
    deltas = [-0.02, 0.00, 0.03, 0.04, -0.01]
    interval_a = paired_bootstrap_ci(deltas, n_resamples=4000, confidence_level=0.95, random_seed=17)
    interval_b = paired_bootstrap_ci(deltas, n_resamples=4000, confidence_level=0.95, random_seed=17)
    assert interval_a == interval_b
    assert interval_a.ci_lower <= interval_a.point_estimate <= interval_a.ci_upper


def test_evaluate_acceptance_rule_requires_zero_and_upper_bound() -> None:
    passing = evaluate_acceptance_rule(-0.02, 0.08)
    assert passing.ci_contains_zero is True
    assert passing.upper_bound_pass is True
    assert passing.passed is True

    misses_zero = evaluate_acceptance_rule(0.01, 0.08)
    assert misses_zero.ci_contains_zero is False
    assert misses_zero.upper_bound_pass is True
    assert misses_zero.passed is False

    exceeds_upper = evaluate_acceptance_rule(-0.03, 0.11)
    assert exceeds_upper.ci_contains_zero is True
    assert exceeds_upper.upper_bound_pass is False
    assert exceeds_upper.passed is False


def test_certify_paired_metrics_applies_vault_rule() -> None:
    metrics_df = _synthetic_metric_rows()
    paired = build_paired_metric_table(
        metrics_df,
        metric_cols=["val_rmse", "median_curve_rel_l2_pct"],
    )

    certification = certify_paired_metrics(
        paired,
        n_resamples=6000,
        confidence_level=0.95,
        random_seed=23,
    )

    assert set(certification.columns) == {
        "diameter_um",
        "metric_name",
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
    }
    assert certification["n_pairs"].tolist() == [5, 5, 5, 5]

    passed_row = certification[
        (certification["diameter_um"] == "2.1") & (certification["metric_name"] == "val_rmse")
    ].iloc[0]
    assert bool(passed_row["ci_contains_zero"]) is True
    assert bool(passed_row["upper_bound_pass"]) is True
    assert bool(passed_row["passed"]) is True

    failed_row = certification[
        (certification["diameter_um"] == "2.9") & (certification["metric_name"] == "val_rmse")
    ].iloc[0]
    assert failed_row["ci_lower"] > 0.0
    assert failed_row["ci_upper"] > 0.10
    assert bool(failed_row["ci_contains_zero"]) is False
    assert bool(failed_row["upper_bound_pass"]) is False
    assert bool(failed_row["passed"]) is False


def test_build_certification_summary_composes_per_diameter_and_overall() -> None:
    paired = build_paired_metric_table(
        _synthetic_metric_rows(),
        metric_cols=["val_rmse", "median_curve_rel_l2_pct"],
    )
    certification = certify_paired_metrics(paired, n_resamples=6000, random_seed=23)

    per_diameter = summarize_certification_by_diameter(certification)
    row_21 = per_diameter[per_diameter["diameter_um"] == "2.1"].iloc[0]
    assert bool(row_21["all_metrics_passed"]) is True
    assert row_21["failed_metrics"] == ()

    row_29 = per_diameter[per_diameter["diameter_um"] == "2.9"].iloc[0]
    assert bool(row_29["all_metrics_passed"]) is False
    assert row_29["failed_metrics"] == ("val_rmse",)

    summary = build_certification_summary(certification)
    assert summary.overall == {
        "diameter_count": 2,
        "passing_diameter_count": 1,
        "failing_diameter_count": 1,
        "metric_count": 4,
        "passed_metric_count": 3,
        "failed_metric_count": 1,
        "all_diameters_passed": False,
        "failing_diameters": ("2.9",),
    }


def test_build_paired_metric_table_rejects_incomplete_pairs() -> None:
    metrics_df = _synthetic_metric_rows()
    incomplete = metrics_df[~(
        (metrics_df["diameter_um"] == "2.9")
        & (metrics_df["seed"] == 4)
        & (metrics_df["surrogate_family"] == "bnn")
    )]
    with pytest.raises(ValueError, match="complete paired DNN/BNN"):
        build_paired_metric_table(incomplete, metric_cols=["val_rmse", "median_curve_rel_l2_pct"])


def test_compute_relative_deltas_rejects_unpaired_zero_baseline() -> None:
    with pytest.raises(ValueError, match="zero"):
        compute_relative_deltas([0.0, 10.0], [1.0, 11.0])
