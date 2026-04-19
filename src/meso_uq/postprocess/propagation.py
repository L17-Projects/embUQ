from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .maps import load_posterior_samples
from ..predictive_statistics import compute_interval_statistics


def _evaluate_posterior_samples(samples_df: pd.DataFrame, *, evaluate_sample: Callable, reference_points: list[float]):
    value_samples = []
    std_samples = []
    parameter_columns = [
        c for c in samples_df.columns if c not in {"logLikelihood", "logPrior", "logPosterior"}
    ]

    for _, row in samples_df.iterrows():
        params = [float(row[c]) for c in parameter_columns]
        sample = {"Parameters": params}
        evaluate_sample(sample, reference_points)
        value_samples.append(np.asarray(sample["Reference Evaluations"], dtype=float))
        std_samples.append(np.asarray(sample.get("Standard Deviation", []), dtype=float))

    values = np.vstack(value_samples)
    stds = np.vstack(std_samples) if std_samples and len(std_samples[0]) else np.zeros_like(values)
    return values, stds


def summarize_propagation(reference_points: list[float], values: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "x": np.asarray(reference_points, dtype=float),
            "mean": np.mean(arr, axis=0),
            "median": np.median(arr, axis=0),
            "q05": np.quantile(arr, 0.05, axis=0),
            "q95": np.quantile(arr, 0.95, axis=0),
        }
    )


def summarize_propagation_predictive(
    reference_points: list[float], values: np.ndarray, stds: np.ndarray
) -> pd.DataFrame:
    """Posterior-predictive summary including observation noise (parameter uncertainty + sigma noise)."""
    stats = compute_interval_statistics(values, stds, percentiles=(90,), include_observation_noise=True)
    return pd.DataFrame(
        {
            "x": np.asarray(reference_points, dtype=float),
            "mean": np.mean(values, axis=0),
            "median": np.median(values, axis=0),
            "q05": stats["ci_90_lower"],
            "q95": stats["ci_90_upper"],
        }
    )


def propagate_run_directory(
    run_dir: str | Path,
    *,
    reference_points: list[float],
    evaluate_sample: Callable,
    output_csv: str | Path,
) -> dict:
    samples_df = load_posterior_samples(run_dir)
    values, stds = _evaluate_posterior_samples(
        samples_df,
        evaluate_sample=evaluate_sample,
        reference_points=reference_points,
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Parameter-uncertainty-only summary (legacy, kept for traceability)
    summary = summarize_propagation(reference_points, values)
    summary.to_csv(output_csv, index=False)

    # Posterior-predictive summary: parameter uncertainty + observation noise
    predictive_csv = output_csv.with_name(output_csv.stem + "_predictive.csv")
    predictive = summarize_propagation_predictive(reference_points, values, stds)
    predictive.to_csv(predictive_csv, index=False)

    return {
        "summary_csv": str(output_csv),
        "predictive_csv": str(predictive_csv),
        "num_samples": int(values.shape[0]),
        "num_points": int(values.shape[1]),
    }
