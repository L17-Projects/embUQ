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

    output_csv = tmp_path / "prop" / "summary.csv"
    result = propagation.propagate_run_directory(
        tmp_path / "run",
        reference_points=[0.0, 1.0, 2.0],
        evaluate_sample=_evaluate,
        output_csv=output_csv,
    )

    assert output_csv.exists()
    frame = pd.read_csv(output_csv)
    assert list(frame.columns) == ["x", "mean", "median", "q05", "q95"]
    assert result == {
        "summary_csv": str(output_csv),
        "num_samples": 2,
        "num_points": 3,
    }
