#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from meso_uq.noise import (
    AdditiveRelativeObservationNoiseConfig,
    CorrelatedCurveNoiseConfig,
    CurveGrid,
    LikelihoodInputs,
    RobustLikelihoodConfig,
    build_additive_relative_observation_noise,
    build_correlated_curve_covariance,
    evaluate_observation_likelihood,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_observation_table(output_root: Path) -> Path:
    predictions = np.linspace(-5.0, 5.0, 41)
    config = AdditiveRelativeObservationNoiseConfig(additive_sigma=0.2, relative_sigma=0.1)
    result = build_additive_relative_observation_noise(predictions.tolist(), config)
    table_path = output_root / "observation_sigma.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prediction", "additive_sigma", "relative_sigma", "total_sigma"],
        )
        writer.writeheader()
        for prediction, additive, relative, total in zip(
            result.predictions,
            result.additive_variance,
            result.relative_variance,
            result.total_variance,
        ):
            writer.writerow(
                {
                    "prediction": prediction,
                    "additive_sigma": additive ** 0.5,
                    "relative_sigma": relative ** 0.5,
                    "total_sigma": total ** 0.5,
                }
            )
    return table_path


def _write_plots(output_root: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = np.linspace(-5.0, 5.0, 41)
    observation = build_additive_relative_observation_noise(
        predictions.tolist(),
        AdditiveRelativeObservationNoiseConfig(additive_sigma=0.2, relative_sigma=0.1),
    )
    observation_png = output_root / "observation_sigma.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(predictions, np.sqrt(observation.additive_variance), label="additive")
    ax.plot(predictions, np.sqrt(observation.relative_variance), label="relative")
    ax.plot(predictions, np.sqrt(observation.total_variance), label="total", linewidth=2.0)
    ax.set_xlabel("prediction")
    ax.set_ylabel("sigma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(observation_png, dpi=150)
    plt.close(fig)

    covariance = build_correlated_curve_covariance(
        CurveGrid(tuple(np.linspace(0.0, 1.0, 25)), "diagnostic"),
        CorrelatedCurveNoiseConfig(amplitude=0.3, length_scale=0.2),
    )
    covariance_png = output_root / "correlated_covariance.png"
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(covariance.covariance, origin="lower", cmap="viridis")
    ax.set_title("correlated covariance")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(covariance_png, dpi=150)
    plt.close(fig)

    influence_png = output_root / "student_t_influence.png"
    residuals = np.linspace(0.0, 8.0, 161)
    gaussian_weights = []
    student_weights = []
    for residual in residuals:
        inputs = LikelihoodInputs(observed=(float(residual),), predicted=(0.0,), standard_deviation=(1.0,))
        gaussian_weights.append(evaluate_observation_likelihood(inputs, RobustLikelihoodConfig()).influence_weights[0])
        student_weights.append(
            evaluate_observation_likelihood(
                inputs,
                RobustLikelihoodConfig(kind="student_t", degrees_of_freedom=4.0),
            ).influence_weights[0]
        )
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(residuals, gaussian_weights, label="gaussian")
    ax.plot(residuals, student_weights, label="student_t df=4")
    ax.set_xlabel("standardized residual")
    ax.set_ylabel("influence weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(influence_png, dpi=150)
    plt.close(fig)

    _write_json(output_root / "correlated_covariance_summary.json", dict(covariance.summary))
    return {
        "observation_plot": observation_png.as_posix(),
        "covariance_plot": covariance_png.as_posix(),
        "student_t_influence_plot": influence_png.as_posix(),
        "covariance_summary": (output_root / "correlated_covariance_summary.json").as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M2 noise primitive diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("_runs/noise/m2_diagnostics"))
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    table = _write_observation_table(output_root)
    artifacts = _write_plots(output_root)
    manifest = {
        "schema_version": 1,
        "artifacts": {"observation_table": table.as_posix(), **artifacts},
        "description": "M2 additive/relative, correlated covariance, and Student-t robust likelihood diagnostics.",
    }
    _write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
