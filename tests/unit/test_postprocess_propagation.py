from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import meso_uq.postprocess.propagation as propagation


def test_evaluate_posterior_samples_uses_parameter_columns_and_std() -> None:
    samples = pd.DataFrame(
        [
            {
                "Yt": 1.0,
                "kb": 2.0,
                "d0": 0.1,
                "sigma": 0.01,
                "logLikelihood": -1.0,
                "logPrior": -2.0,
                "logPosterior": -3.0,
            },
            {
                "Yt": 1.5,
                "kb": 2.5,
                "d0": 0.2,
                "sigma": 0.02,
                "logLikelihood": -1.1,
                "logPrior": -2.1,
                "logPosterior": -3.1,
            },
        ]
    )

    def _evaluate(sample: dict, reference_points: list[float]) -> None:
        total = sum(sample["Parameters"])
        sample["Reference Evaluations"] = [total + x for x in reference_points]
        sample["Standard Deviation"] = [0.5 for _ in reference_points]

    values, stds = propagation._evaluate_posterior_samples(
        samples, evaluate_sample=_evaluate, reference_points=[0.0, 1.0]
    )

    assert values.shape == (2, 2)
    assert stds.shape == (2, 2)
    assert np.allclose(stds, 0.5)


def test_summarize_propagation_computes_expected_statistics() -> None:
    values = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=float)
    summary = propagation.summarize_propagation([0.0, 1.0], values)

    assert list(summary.columns) == ["x", "mean", "median", "q05", "q95"]
    assert summary["mean"].tolist() == [1.0, 3.0]
    assert summary["median"].tolist() == [1.0, 3.0]


def test_summarize_propagation_predictive_wider_than_param_only() -> None:
    """Noise-inclusive CI must be at least as wide as parameter-only CI."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=1.0, scale=0.05, size=(50, 5))
    stds = np.full_like(values, 0.3)  # substantial noise

    param_only = propagation.summarize_propagation([0.0, 1.0, 2.0, 3.0, 4.0], values)
    predictive = propagation.summarize_propagation_predictive([0.0, 1.0, 2.0, 3.0, 4.0], values, stds)

    assert list(predictive.columns) == ["x", "mean", "median", "q05", "q95"]
    # Predictive CI must be wider at every point
    param_width = (param_only["q95"] - param_only["q05"]).values
    pred_width = (predictive["q95"] - predictive["q05"]).values
    assert np.all(pred_width > param_width), "Predictive CI should be wider than parameter-only CI"


def test_summarize_propagation_predictive_zero_noise_matches_param_only() -> None:
    """With zero noise, predictive CI collapses to raw parameter quantiles."""
    rng = np.random.default_rng(1)
    values = rng.normal(loc=2.0, scale=0.1, size=(200, 3))
    stds = np.zeros_like(values)

    param_only = propagation.summarize_propagation([0.0, 1.0, 2.0], values)
    predictive = propagation.summarize_propagation_predictive([0.0, 1.0, 2.0], values, stds)

    # Medians must agree
    assert np.allclose(param_only["median"].values, predictive["median"].values, atol=1e-6)


def test_propagate_run_directory_writes_summary_csv(tmp_path: Path, monkeypatch) -> None:
    samples = pd.DataFrame(
        [
            {"Yt": 1.0, "kb": 2.0, "d0": 0.1, "sigma": 0.01},
            {"Yt": 1.2, "kb": 2.2, "d0": 0.2, "sigma": 0.02},
        ]
    )
    monkeypatch.setattr(propagation, "load_posterior_samples", lambda run_dir: samples)

    def _evaluate(sample: dict, reference_points: list[float]) -> None:
        base = sum(sample["Parameters"])
        sample["Reference Evaluations"] = [base + x for x in reference_points]
        sample["Standard Deviation"] = [0.1 * (base + x) for x in reference_points]

    output_csv = tmp_path / "prop" / "summary.csv"
    result = propagation.propagate_run_directory(
        tmp_path / "run",
        reference_points=[0.0, 1.0, 2.0],
        evaluate_sample=_evaluate,
        output_csv=output_csv,
    )

    # summary.csv (parameter-only) must exist with correct columns
    assert output_csv.exists()
    frame = pd.read_csv(output_csv)
    assert list(frame.columns) == ["x", "mean", "median", "q05", "q95"]

    # summary_predictive.csv (noise-inclusive) must also be written
    predictive_csv = tmp_path / "prop" / "summary_predictive.csv"
    assert predictive_csv.exists()
    pred_frame = pd.read_csv(predictive_csv)
    assert list(pred_frame.columns) == ["x", "mean", "median", "q05", "q95"]

    # Return dict must include both paths
    assert result["summary_csv"] == str(output_csv)
    assert result["predictive_csv"] == str(predictive_csv)
    assert result["num_samples"] == 2
    assert result["num_points"] == 3
