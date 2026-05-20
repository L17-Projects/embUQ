#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from meso_uq.noise import (
    AdditiveRelativeObservationNoiseConfig,
    ContactAlignmentInputs,
    ContactAlignmentUncertaintyConfig,
    GeometryParameterUncertainty,
    GeometrySensitivityInputs,
    GeometryUncertaintyConfig,
    SurrogateCovarianceConfig,
    SurrogateCovarianceInputs,
    build_additive_relative_observation_noise,
    build_contact_alignment_covariance,
    build_geometry_uncertainty_covariance,
    build_surrogate_covariance,
    legacy_compression_surrogate_likelihood,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _matrix_summary(covariance: np.ndarray) -> dict[str, Any]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return {
        "shape": list(covariance.shape),
        "diagonal_min": _finite_or_none(np.min(np.diag(covariance))),
        "diagonal_max": _finite_or_none(np.max(np.diag(covariance))),
        "min_eigenvalue": _finite_or_none(np.min(eigenvalues)),
        "max_eigenvalue": _finite_or_none(np.max(eigenvalues)),
        "condition_number": _finite_or_none(np.linalg.cond(covariance)) if not np.allclose(covariance, 0.0) else None,
    }


def _diagnostic_inputs() -> dict[str, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 50)
    predictions = 2.0 + 1.5 * grid + 0.25 * np.sin(2.0 * np.pi * grid)
    surrogate_std = 0.12 + 0.08 * grid
    full_correlation = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / 0.18) ** 2)
    full_covariance = np.outer(surrogate_std, surrogate_std) * full_correlation
    low_rank_factors = np.column_stack((0.16 * np.ones_like(grid), 0.12 * (grid - grid.mean())))
    radius_sensitivity = 0.4 + 0.15 * grid
    height_sensitivity = -0.08 + 0.15 * grid**2
    return {
        "grid": grid,
        "predictions": predictions,
        "surrogate_std": surrogate_std,
        "full_covariance": full_covariance,
        "low_rank_factors": low_rank_factors,
        "radius_sensitivity": radius_sensitivity,
        "height_sensitivity": height_sensitivity,
    }


def _build_measurement_covariance(inputs: dict[str, np.ndarray]) -> np.ndarray:
    contact = build_contact_alignment_covariance(
        ContactAlignmentInputs(
            controls=tuple(float(value) for value in inputs["grid"]),
            predictions=tuple(float(value) for value in inputs["predictions"]),
            curve_id="m4/contact_alignment_fixture",
        ),
        ContactAlignmentUncertaintyConfig(contact_offset_sigma=0.015, alignment_slope_sigma=0.01, minimum_variance=1e-5),
    )
    geometry = build_geometry_uncertainty_covariance(
        GeometrySensitivityInputs(
            predictions=tuple(float(value) for value in inputs["predictions"]),
            sensitivities={
                "radius_um": tuple(float(value) for value in inputs["radius_sensitivity"]),
                "height_um": tuple(float(value) for value in inputs["height_sensitivity"]),
            },
            curve_id="m4/geometry_fixture",
        ),
        GeometryUncertaintyConfig(
            parameters=(
                GeometryParameterUncertainty("radius_um", sigma=0.01, units="micrometer", nominal=2.0),
                GeometryParameterUncertainty("height_um", sigma=0.03, units="micrometer", nominal=14.28),
            )
        ),
    )
    return contact.covariance.covariance + geometry.covariance.covariance


def _build_surrogate_results(inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    base = SurrogateCovarianceInputs(
        predictions=tuple(float(value) for value in inputs["predictions"]),
        predictive_standard_deviation=tuple(float(value) for value in inputs["surrogate_std"]),
        predictive_covariance=tuple(tuple(float(value) for value in row) for row in inputs["full_covariance"]),
        low_rank_factors=tuple(tuple(float(value) for value in row) for row in inputs["low_rank_factors"]),
        curve_grid=tuple(float(value) for value in inputs["grid"]),
        curve_id="m4/surrogate_covariance_fixture",
    )
    disabled = build_surrogate_covariance(base, SurrogateCovarianceConfig(enabled=False, kind="diagonal"))
    diagonal = build_surrogate_covariance(base, SurrogateCovarianceConfig(kind="diagonal", jitter=1e-12, max_jitter=1e-8))
    full = build_surrogate_covariance(base, SurrogateCovarianceConfig(kind="full", jitter=1e-12, max_jitter=1e-8))
    low_rank = build_surrogate_covariance(base, SurrogateCovarianceConfig(kind="low_rank", jitter=1e-12, max_jitter=1e-8))
    return {"disabled": disabled, "diagonal": diagonal, "full": full, "low_rank": low_rank}


def _write_comparison_table(output_root: Path, inputs: dict[str, np.ndarray], results: dict[str, Any]) -> Path:
    path = output_root / "surrogate_covariance_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["index", "grid", "prediction", "disabled_sigma", "diagonal_sigma", "full_sigma", "low_rank_sigma"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (grid, prediction) in enumerate(zip(inputs["grid"], inputs["predictions"])):
            writer.writerow(
                {
                    "index": index,
                    "grid": float(grid),
                    "prediction": float(prediction),
                    "disabled_sigma": results["disabled"].standard_deviation[index],
                    "diagonal_sigma": results["diagonal"].standard_deviation[index],
                    "full_sigma": results["full"].standard_deviation[index],
                    "low_rank_sigma": results["low_rank"].standard_deviation[index],
                }
            )
    return path


def _legacy_quadrature_metric(inputs: dict[str, np.ndarray], diagonal: Any) -> dict[str, float]:
    sample_predictions = tuple(float(value) for value in inputs["predictions"][:8])
    sample_std = tuple(float(value) for value in inputs["surrogate_std"][:8])
    sigma = 0.07
    legacy = legacy_compression_surrogate_likelihood(
        sample_predictions,
        sigma,
        surrogate_standard_deviation=sample_std,
    )
    observation = build_additive_relative_observation_noise(
        sample_predictions,
        AdditiveRelativeObservationNoiseConfig(relative_sigma=sigma),
    )
    total_variance = np.asarray(observation.total_variance) + np.diag(diagonal.covariance.covariance[:8, :8])
    total_std = np.sqrt(total_variance)
    delta = np.asarray(legacy.standard_deviation) - total_std
    return {
        "legacy_bnn_quadrature_max_abs_delta": float(np.max(np.abs(delta))),
        "legacy_bnn_quadrature_points": float(len(sample_predictions)),
    }


def _write_plots(
    output_root: Path,
    inputs: dict[str, np.ndarray],
    measurement_covariance: np.ndarray,
    results: dict[str, Any],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap_path = output_root / "surrogate_covariance_heatmaps.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for ax, key in zip(axes, ("diagonal", "full", "low_rank")):
        image = ax.imshow(results[key].covariance.covariance, origin="lower", cmap="viridis")
        ax.set_title(key)
        ax.set_xlabel("point")
        ax.set_ylabel("point")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    band_path = output_root / "surrogate_covariance_sigma_band.png"
    grid = inputs["grid"]
    predictions = inputs["predictions"]
    observation = build_additive_relative_observation_noise(
        tuple(float(value) for value in predictions),
        AdditiveRelativeObservationNoiseConfig(additive_sigma=0.05, relative_sigma=0.04),
    )
    observation_cov = np.diag(observation.total_variance)
    base_sigma = np.sqrt(np.maximum(np.diag(observation_cov + measurement_covariance), 0.0))
    with_surrogate_sigma = np.sqrt(
        np.maximum(np.diag(observation_cov + measurement_covariance + results["diagonal"].covariance.covariance), 0.0)
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(grid, predictions, color="black", linewidth=2.0, label="prediction")
    ax.fill_between(grid, predictions - base_sigma, predictions + base_sigma, alpha=0.22, label="M2+M3 sigma")
    ax.fill_between(
        grid,
        predictions - with_surrogate_sigma,
        predictions + with_surrogate_sigma,
        alpha=0.18,
        label="M2+M3+M4 diagonal sigma",
    )
    ax.set_xlabel("normalized control")
    ax.set_ylabel("observable")
    ax.legend()
    fig.tight_layout()
    fig.savefig(band_path, dpi=150)
    plt.close(fig)

    return {
        "heatmap_plot": heatmap_path.as_posix(),
        "sigma_band_plot": band_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M4 surrogate covariance diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("_runs/noise/m4_surrogate_covariance_diagnostics"))
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = _diagnostic_inputs()
    measurement_covariance = _build_measurement_covariance(inputs)
    results = _build_surrogate_results(inputs)
    table_path = _write_comparison_table(output_root, inputs, results)
    plots = _write_plots(output_root, inputs, measurement_covariance, results)
    summary_path = output_root / "surrogate_covariance_summary.json"
    summary = {
        "schema_version": 1,
        "measurement_covariance": _matrix_summary(measurement_covariance),
        "representations": {name: dict(result.summary) for name, result in results.items()},
        "matrix_summaries": {name: _matrix_summary(result.covariance.covariance) for name, result in results.items()},
        **_legacy_quadrature_metric(inputs, results["diagonal"]),
    }
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "description": "M4 surrogate covariance representation, composition, and legacy BNN quadrature diagnostics.",
        "artifacts": {
            "comparison_table": table_path.as_posix(),
            "covariance_summary": summary_path.as_posix(),
            **plots,
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
