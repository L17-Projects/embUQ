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
    CorrelatedCurveNoiseConfig,
    CovarianceTerm,
    CurveGrid,
    GeometryParameterUncertainty,
    GeometrySensitivityInputs,
    GeometryUncertaintyConfig,
    LowRankDiscrepancyConfig,
    LowRankDiscrepancyInputs,
    SurrogateCovarianceConfig,
    SurrogateCovarianceInputs,
    TotalCovarianceConfig,
    assemble_total_covariance,
    build_additive_relative_observation_noise,
    build_contact_alignment_covariance,
    build_correlated_curve_covariance,
    build_geometry_uncertainty_covariance,
    build_low_rank_model_discrepancy_covariance,
    build_polynomial_discrepancy_basis,
    build_surrogate_covariance,
    covariance_term_from_diagonal,
    fit_low_rank_discrepancy_coefficients,
    gaussian_log_likelihood_from_covariance,
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
        "matrix_rank": int(np.linalg.matrix_rank(covariance)) if covariance.size else 0,
        "condition_number": _finite_or_none(np.linalg.cond(covariance)) if not np.allclose(covariance, 0.0) else None,
    }


def _diagnostic_inputs() -> dict[str, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 60)
    predictions = 1.5 + 0.8 * grid + 0.15 * np.sin(2.0 * np.pi * grid)
    surrogate_std = 0.055 + 0.025 * grid
    basis = np.asarray(build_polynomial_discrepancy_basis(grid, degree=2, include_intercept=False))
    true_discrepancy_coefficients = np.asarray((0.08, -0.045))
    true_discrepancy = basis @ true_discrepancy_coefficients
    observations = predictions + true_discrepancy + 0.01 * np.cos(4.0 * np.pi * grid)
    return {
        "grid": grid,
        "predictions": predictions,
        "surrogate_std": surrogate_std,
        "basis": basis,
        "true_discrepancy_coefficients": true_discrepancy_coefficients,
        "true_discrepancy": true_discrepancy,
        "observations": observations,
        "radius_sensitivity": 0.25 + 0.08 * grid,
        "height_sensitivity": -0.06 + 0.1 * grid**2,
    }


def _children(result: Any, total_name: str) -> dict[str, np.ndarray]:
    return {name: matrix for name, matrix in result.covariance_components.items() if name != total_name}


def _build_terms(inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    predictions = tuple(float(value) for value in inputs["predictions"])
    grid = tuple(float(value) for value in inputs["grid"])
    observation = build_additive_relative_observation_noise(
        predictions,
        AdditiveRelativeObservationNoiseConfig(additive_sigma=0.035, relative_sigma=0.035, minimum_total_variance=1e-5),
    )
    correlated = build_correlated_curve_covariance(
        CurveGrid(grid, "m5/correlated_curve"),
        CorrelatedCurveNoiseConfig(amplitude=0.012, length_scale=0.22, jitter=1e-12, max_jitter=1e-8),
    )
    contact = build_contact_alignment_covariance(
        ContactAlignmentInputs(controls=grid, predictions=predictions, curve_id="m5/contact_alignment"),
        ContactAlignmentUncertaintyConfig(contact_offset_sigma=0.01, alignment_slope_sigma=0.008, minimum_variance=5e-6),
    )
    geometry = build_geometry_uncertainty_covariance(
        GeometrySensitivityInputs(
            predictions=predictions,
            sensitivities={
                "radius_um": tuple(float(value) for value in inputs["radius_sensitivity"]),
                "height_um": tuple(float(value) for value in inputs["height_sensitivity"]),
            },
            curve_id="m5/geometry",
        ),
        GeometryUncertaintyConfig(
            parameters=(
                GeometryParameterUncertainty("radius_um", sigma=0.015, units="micrometer", nominal=2.0),
                GeometryParameterUncertainty("height_um", sigma=0.025, units="micrometer", nominal=14.28),
            ),
            covariance=((0.000225, 0.000035), (0.000035, 0.000625)),
        ),
    )
    surrogate = build_surrogate_covariance(
        SurrogateCovarianceInputs(
            predictions=predictions,
            predictive_standard_deviation=tuple(float(value) for value in inputs["surrogate_std"]),
            curve_grid=grid,
            curve_id="m5/surrogate_predictive",
        ),
        SurrogateCovarianceConfig(kind="diagonal", jitter=1e-12, max_jitter=1e-8),
    )
    discrepancy_inputs = LowRankDiscrepancyInputs(
        predictions=predictions,
        basis=tuple(tuple(float(value) for value in row) for row in inputs["basis"]),
        basis_names=("linear", "curvature"),
        curve_id="m5/model_discrepancy",
    )
    discrepancy = build_low_rank_model_discrepancy_covariance(
        discrepancy_inputs,
        LowRankDiscrepancyConfig(
            enabled=True,
            coefficient_covariance=((0.0064, 0.0012), (0.0012, 0.0025)),
            minimum_variance=5e-6,
            jitter=1e-12,
            max_jitter=1e-8,
        ),
    )
    disabled_discrepancy = build_low_rank_model_discrepancy_covariance(
        discrepancy_inputs,
        LowRankDiscrepancyConfig(enabled=False, coefficient_scale=1.0, minimum_variance=5e-6),
    )

    observation_children = {
        f"observation:{name}": np.diag(values)
        for name, values in observation.variance_components.items()
    }
    base_terms = (
        covariance_term_from_diagonal(
            "observation:additive_relative",
            observation.total_variance,
            children=observation_children,
            summary={"source": "M2 additive-relative observation noise"},
        ),
        CovarianceTerm("observation:correlated_curve", correlated.covariance, summary=correlated.summary),
        CovarianceTerm("measurement:contact_alignment", contact.covariance.covariance, children=_children(contact, "contact_alignment_total"), summary=contact.summary),
        CovarianceTerm("measurement:geometry", geometry.covariance.covariance, children=_children(geometry, "geometry_jacobian"), summary=geometry.summary),
        CovarianceTerm("surrogate:predictive", surrogate.covariance.covariance, children=_children(surrogate, "surrogate_covariance_total"), summary=surrogate.summary),
    )
    enabled_discrepancy_term = CovarianceTerm(
        "discrepancy:low_rank",
        discrepancy.covariance.covariance,
        children=_children(discrepancy, "model_discrepancy_total"),
        summary=discrepancy.summary,
    )
    disabled_discrepancy_term = CovarianceTerm(
        "discrepancy:low_rank",
        disabled_discrepancy.covariance.covariance,
        included=False,
        children=_children(disabled_discrepancy, "model_discrepancy_total"),
        summary=disabled_discrepancy.summary,
    )
    return {
        "observation": observation,
        "correlated": correlated,
        "contact": contact,
        "geometry": geometry,
        "surrogate": surrogate,
        "discrepancy": discrepancy,
        "disabled_discrepancy": disabled_discrepancy,
        "base_terms": base_terms,
        "enabled_discrepancy_term": enabled_discrepancy_term,
        "disabled_discrepancy_term": disabled_discrepancy_term,
    }


def _write_components_table(output_root: Path, inputs: dict[str, np.ndarray], total: Any) -> Path:
    path = output_root / "total_covariance_components.csv"
    term_names = list(total.included_term_names)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["index", "grid", "prediction", *[f"{name}_variance" for name in term_names], "total_variance", "total_sigma"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (grid, prediction) in enumerate(zip(inputs["grid"], inputs["predictions"])):
            row = {"index": index, "grid": float(grid), "prediction": float(prediction)}
            for name in term_names:
                row[f"{name}_variance"] = float(np.diag(total.covariance_components[name])[index])
            row["total_variance"] = float(np.diag(total.covariance.covariance)[index])
            row["total_sigma"] = float(total.standard_deviation[index])
            writer.writerow(row)
    return path


def _write_plots(output_root: Path, inputs: dict[str, np.ndarray], no_discrepancy: Any, full: Any, discrepancy: Any, fit: Any) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap_path = output_root / "total_covariance_heatmaps.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    for ax, title, matrix in (
        (axes[0], "M2-M4 total", no_discrepancy.covariance.covariance),
        (axes[1], "discrepancy", discrepancy.covariance.covariance),
        (axes[2], "M2-M5 total", full.covariance.covariance),
    ):
        image = ax.imshow(matrix, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("point")
        ax.set_ylabel("point")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    basis_path = output_root / "model_discrepancy_basis.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for index in range(inputs["basis"].shape[1]):
        ax.plot(inputs["grid"], inputs["basis"][:, index], label=f"basis {index}")
    ax.set_xlabel("normalized control")
    ax.set_ylabel("basis value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(basis_path, dpi=150)
    plt.close(fig)

    recovery_path = output_root / "model_discrepancy_recovery.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(inputs["grid"], inputs["true_discrepancy"], color="black", linewidth=2.0, label="true discrepancy")
    ax.plot(inputs["grid"], fit.fitted_discrepancy, linestyle="--", label="ridge fit")
    sigma = np.sqrt(np.maximum(np.diag(full.covariance.covariance), 0.0))
    ax.fill_between(inputs["grid"], inputs["true_discrepancy"] - sigma, inputs["true_discrepancy"] + sigma, alpha=0.18, label="M5 total sigma")
    ax.set_xlabel("normalized control")
    ax.set_ylabel("discrepancy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(recovery_path, dpi=150)
    plt.close(fig)

    return {
        "heatmap_plot": heatmap_path.as_posix(),
        "basis_plot": basis_path.as_posix(),
        "recovery_plot": recovery_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M5 full hierarchy covariance diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("_runs/noise/m5_full_hierarchy_diagnostics"))
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = _diagnostic_inputs()
    built = _build_terms(inputs)
    config = TotalCovarianceConfig(jitter=1e-10, max_jitter=1e-6)
    no_discrepancy = assemble_total_covariance(built["base_terms"], config)
    disabled_discrepancy = assemble_total_covariance((*built["base_terms"], built["disabled_discrepancy_term"]), config)
    full = assemble_total_covariance((*built["base_terms"], built["enabled_discrepancy_term"]), config)

    duplicate_guard_exercised = False
    try:
        assemble_total_covariance((built["base_terms"][0], built["base_terms"][0]), config)
    except ValueError:
        duplicate_guard_exercised = True

    fit = fit_low_rank_discrepancy_coefficients(
        tuple(float(value) for value in inputs["true_discrepancy"]),
        tuple(tuple(float(value) for value in row) for row in inputs["basis"]),
        regularization=1e-10,
        protected_sensitivities={"offset": tuple(1.0 for _ in inputs["grid"])},
    )
    table_path = _write_components_table(output_root, inputs, full)
    plots = _write_plots(output_root, inputs, no_discrepancy, full, built["discrepancy"], fit)

    component_sum = sum(full.covariance_components[name] for name in full.included_term_names)
    residuals = tuple(float(obs - pred) for obs, pred in zip(inputs["observations"], inputs["predictions"]))
    full_log_likelihood = gaussian_log_likelihood_from_covariance(residuals, full.covariance)
    summary = {
        "schema_version": 1,
        "total_covariance": dict(full.summary),
        "no_discrepancy_total": dict(no_discrepancy.summary),
        "disabled_discrepancy_total": dict(disabled_discrepancy.summary),
        "discrepancy": dict(built["discrepancy"].summary),
        "matrix_summaries": {
            "no_discrepancy": _matrix_summary(no_discrepancy.covariance.covariance),
            "enabled_discrepancy": _matrix_summary(built["discrepancy"].covariance.covariance),
            "full": _matrix_summary(full.covariance.covariance),
        },
        "metrics": {
            "included_component_sum_max_abs_delta": float(np.max(np.abs(component_sum - full.covariance.covariance))),
            "disabled_discrepancy_max_abs_delta": float(np.max(np.abs(disabled_discrepancy.covariance.covariance - no_discrepancy.covariance.covariance))),
            "duplicate_guard_exercised": duplicate_guard_exercised,
            "full_gaussian_log_likelihood": float(full_log_likelihood),
            "full_gaussian_log_likelihood_is_finite": bool(math.isfinite(full_log_likelihood)),
            "recovered_discrepancy_rmse": float(fit.residual_rmse),
            "protected_offset_drift_abs": abs(float(fit.protected_parameter_drift["offset"])),
        },
        "synthetic_recovery": {
            "true_coefficients": [float(value) for value in inputs["true_discrepancy_coefficients"]],
            "fit_coefficients": list(fit.coefficients),
            "residual_rmse": float(fit.residual_rmse),
            "protected_parameter_drift": dict(fit.protected_parameter_drift),
        },
    }
    summary_path = output_root / "total_covariance_summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "description": "M5 full hierarchy covariance assembly and low-rank model discrepancy diagnostics.",
        "artifacts": {
            "covariance_summary": summary_path.as_posix(),
            "components_table": table_path.as_posix(),
            **plots,
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
