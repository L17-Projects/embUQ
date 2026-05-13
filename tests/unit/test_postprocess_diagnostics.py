from __future__ import annotations

import math

import pandas as pd
import pytest

from meso_uq.postprocess.diagnostics import (
    PHASE1_POSTERIOR_FIGURE_POLICY,
    duplicate_mass_comparison,
    duplicate_particle_metrics,
    mean_or_nan,
    posterior_parameter_columns,
)


def test_posterior_parameter_columns_excludes_log_columns() -> None:
    frame = pd.DataFrame([{"Yt": 1.0, "kb": 2.0, "logLikelihood": -1.0, "logPosterior": -2.0}])
    assert posterior_parameter_columns(frame) == ["Yt", "kb"]


def test_duplicate_particle_metrics_computes_top_duplicate_masses() -> None:
    frame = pd.DataFrame(
        {
            "Yt": [1.0, 1.0, 1.0, 2.0, 3.0, 3.0],
            "kb": [5.0, 5.0, 5.0, 8.0, 9.0, 9.0],
            "logLikelihood": [-1.0, -1.1, -1.2, -2.0, -3.0, -3.1],
        }
    )

    metrics = duplicate_particle_metrics(frame)

    assert metrics["sample_count"] == 6
    assert metrics["unique_particle_count"] == 3
    assert metrics["duplicate_particle_count"] == 2
    assert metrics["top_duplicate_count"] == 3
    assert metrics["top_duplicate_mass"] == pytest.approx(3.0 / 6.0)
    assert metrics["top_10_duplicate_mass"] == pytest.approx(5.0 / 6.0)


def test_duplicate_particle_metrics_reports_zero_masses_when_no_duplicates() -> None:
    frame = pd.DataFrame({"Yt": [1.0, 2.0], "kb": [5.0, 6.0]})
    metrics = duplicate_particle_metrics(frame)
    assert metrics["duplicate_particle_count"] == 0
    assert metrics["top_duplicate_mass"] == pytest.approx(0.0)
    assert metrics["top_10_duplicate_mass"] == pytest.approx(0.0)


def test_duplicate_particle_metrics_handles_empty_samples_and_rejects_empty_parameters() -> None:
    metrics = duplicate_particle_metrics(pd.DataFrame({"Yt": []}))
    assert metrics["sample_count"] == 0
    assert metrics["duplicate_mass_total"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="No parameter columns"):
        duplicate_particle_metrics(pd.DataFrame({"logLikelihood": [-1.0]}))


def test_duplicate_mass_comparison_handles_missing_chain_metrics() -> None:
    posterior = {"top_duplicate_mass": 0.2, "top_10_duplicate_mass": 0.3, "duplicate_mass_total": 0.4}
    comparison = duplicate_mass_comparison(posterior, None)
    assert comparison["top_duplicate_mass_delta"] is None
    assert comparison["top_10_duplicate_mass_delta"] is None
    assert comparison["duplicate_mass_total_delta"] is None


def test_duplicate_mass_comparison_computes_deltas() -> None:
    posterior = {"top_duplicate_mass": 0.2, "top_10_duplicate_mass": 0.4, "duplicate_mass_total": 0.6}
    chain = {"top_duplicate_mass": 0.3, "top_10_duplicate_mass": 0.5, "duplicate_mass_total": 0.9}
    comparison = duplicate_mass_comparison(posterior, chain)
    assert comparison["top_duplicate_mass_delta"] == pytest.approx(0.1)
    assert comparison["top_10_duplicate_mass_delta"] == pytest.approx(0.1)
    assert comparison["duplicate_mass_total_delta"] == pytest.approx(0.3)


def test_mean_or_nan_handles_empty_iterables() -> None:
    assert math.isnan(mean_or_nan([]))
    assert mean_or_nan([0.2, 0.4]) == pytest.approx(0.3)


def test_phase1_posterior_policy_disables_filtering() -> None:
    assert PHASE1_POSTERIOR_FIGURE_POLICY["filtering"] == "disabled"
    assert PHASE1_POSTERIOR_FIGURE_POLICY["figure_source"] == "raw_posterior_samples"
