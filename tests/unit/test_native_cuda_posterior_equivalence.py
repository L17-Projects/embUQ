from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from meso_uq.inference.posterior_equivalence import (
    PosteriorEquivalenceThresholds,
    compare_posterior_samples,
    load_phase2_posterior_samples,
    summarize_posterior_samples,
)


def _samples(offset: float = 0.0) -> dict[str, list[float]]:
    return {
        "Yt": [1.0 + offset, 1.1 + offset, 0.9 + offset, 1.05 + offset],
        "kb": [2.0, 2.1, 1.9, 2.05],
        "sigma": [0.2, 0.21, 0.19, 0.205],
        "logLikelihood": [-10.0, -9.5, -10.3, -9.7],
        "logPrior": [-1.0, -1.1, -1.0, -1.2],
        "logPosterior": [-11.0, -10.6, -11.3, -10.9],
    }


def test_load_phase2_posterior_samples_uses_shared_korali_reader(tmp_path: Path) -> None:
    state = {
        "Variables": [
            {"Name": "Yt"},
            {"Name": "kb"},
            {"Name": "[Sigma]"},
        ],
        "Results": {
            "Posterior Sample Database": [
                [1.0, 2.0, 0.2],
                [1.1, 2.1, 0.21],
            ],
            "Posterior Sample LogLikelihood Database": [-10.0, -9.5],
            "Posterior Sample LogPrior Database": [-1.0, -1.1],
        },
    }
    (tmp_path / "latest").write_text(json.dumps(state), encoding="utf-8")

    loaded = load_phase2_posterior_samples(tmp_path)

    assert list(loaded.columns) == ["Yt", "kb", "sigma", "logLikelihood", "logPrior", "logPosterior"]
    assert loaded["logPosterior"].tolist() == [-11.0, -10.6]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("Posterior Sample LogLikelihood Database", -10.0),
        ("Posterior Sample LogPrior Database", [0.0]),
    ],
)
def test_load_phase2_posterior_samples_rejects_broadcast_raw_log_evidence(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    state = {
        "Variables": [
            {"Name": "Yt"},
            {"Name": "kb"},
        ],
        "Results": {
            "Posterior Sample Database": [
                [1.0, 2.0],
                [1.1, 2.1],
            ],
            "Posterior Sample LogLikelihood Database": [-10.0, -9.5],
            "Posterior Sample LogPrior Database": [-1.0, -1.1],
        },
    }
    state["Results"][field] = bad_value
    (tmp_path / "latest").write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_phase2_posterior_samples(tmp_path)


def test_summarize_posterior_samples_computes_backend_neutral_statistics() -> None:
    summary = summarize_posterior_samples(_samples(), backend="cpu-mpi", source="results_phase_2/latest")

    assert summary.backend == "cpu-mpi"
    assert summary.variables == ("Yt", "kb", "sigma")
    assert summary.sample_count == 4
    assert summary.finite_logposterior_ratio == 1.0
    assert np.isclose(summary.mean["Yt"], 1.0125)
    assert "q0.5" in summary.quantiles["Yt"]
    assert summary.logposterior_max == -10.6
    assert summary.to_manifest()["source"] == "results_phase_2/latest"


def test_compare_posterior_samples_passes_for_close_stochastic_outputs() -> None:
    report = compare_posterior_samples(
        _samples(),
        _samples(offset=0.01),
        thresholds=PosteriorEquivalenceThresholds(
            max_mean_abs_delta=0.02,
            max_quantile_abs_delta=0.03,
            max_std_scaled_delta=0.05,
        ),
    )

    assert report.passed
    assert report.mismatches == ()
    assert report.reference_backend == "cpu-mpi"
    assert report.candidate_backend == "native-cuda"


def test_compare_posterior_samples_allows_reordered_parameter_columns() -> None:
    samples = _samples()
    reordered = {
        "kb": samples["kb"],
        "Yt": samples["Yt"],
        "sigma": samples["sigma"],
        "logLikelihood": samples["logLikelihood"],
        "logPrior": samples["logPrior"],
        "logPosterior": samples["logPosterior"],
    }

    report = compare_posterior_samples(_samples(), reordered)

    assert report.passed
    assert report.mismatches == ()


def test_compare_posterior_samples_reports_mismatched_statistic() -> None:
    report = compare_posterior_samples(
        _samples(),
        _samples(offset=0.5),
        thresholds=PosteriorEquivalenceThresholds(
            max_mean_abs_delta=0.05,
            max_quantile_abs_delta=0.05,
        ),
    )

    assert not report.passed
    assert any("Yt: mean delta" in mismatch for mismatch in report.mismatches)
    assert any("Yt: quantile q0.5 delta" in mismatch for mismatch in report.mismatches)
    manifest = report.to_manifest()
    assert manifest["reference_backend"] == "cpu-mpi"
    assert manifest["candidate_backend"] == "native-cuda"


def test_compare_posterior_samples_reports_nonfinite_logposterior() -> None:
    candidate = _samples()
    candidate["logPosterior"] = [-11.0, float("nan"), -11.3, -10.9]

    report = compare_posterior_samples(candidate, candidate)

    assert not report.passed
    assert any("finite_logposterior_ratio" in mismatch for mismatch in report.mismatches)


def test_summarize_posterior_samples_sums_likelihood_and_prior_when_posterior_missing() -> None:
    samples = _samples()
    samples.pop("logPosterior")
    samples["logPrior"][1] = float("-inf")

    summary = summarize_posterior_samples(samples, backend="native-cuda")

    assert summary.finite_logposterior_ratio == 0.75
    assert summary.logposterior_max == pytest.approx(-10.9)


def test_summarize_posterior_samples_rejects_missing_logposterior_evidence() -> None:
    samples = _samples()
    samples.pop("logPosterior")
    samples.pop("logLikelihood")

    with pytest.raises(ValueError, match="logPosterior or both logLikelihood and logPrior"):
        summarize_posterior_samples(samples, backend="native-cuda")


def test_summarize_posterior_samples_rejects_broadcast_logprior_evidence() -> None:
    samples = _samples()
    samples.pop("logPosterior")
    samples["logPrior"] = [0.0]

    with pytest.raises(ValueError, match="logPrior column length"):
        summarize_posterior_samples(samples, backend="native-cuda")


def test_compare_posterior_samples_reports_sample_count_mismatch() -> None:
    candidate = {name: values[:-1] for name, values in _samples().items()}

    report = compare_posterior_samples(
        _samples(),
        candidate,
        thresholds=PosteriorEquivalenceThresholds(
            max_mean_abs_delta=10.0,
            max_quantile_abs_delta=10.0,
            max_std_scaled_delta=10.0,
            max_logposterior_max_abs_delta=10.0,
        ),
    )

    assert not report.passed
    assert any("sample_count differs" in mismatch for mismatch in report.mismatches)


def test_compare_posterior_samples_reports_variable_mismatch() -> None:
    candidate = _samples()
    candidate.pop("kb")

    report = compare_posterior_samples(_samples(), candidate, parameter_columns=None)

    assert not report.passed
    assert any("variables differ" in mismatch for mismatch in report.mismatches)
