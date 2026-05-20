#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from meso_uq.noise import (
    ContactAlignmentInputs,
    ContactAlignmentUncertaintyConfig,
    GeometryParameterUncertainty,
    GeometrySensitivityInputs,
    GeometryUncertaintyConfig,
    build_contact_alignment_covariance,
    build_geometry_uncertainty_covariance,
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


def _diagnostic_inputs() -> dict[str, Any]:
    controls = np.linspace(0.0, 1.0, 60)
    predictions = 0.35 + 0.25 * controls + 0.08 * np.sin(2.0 * np.pi * controls)
    force_sensitivity = 0.2 + 0.15 * controls
    radius_sensitivity = 0.6 + 0.2 * controls
    height_sensitivity = -0.1 + 0.25 * controls**2
    return {
        "controls": controls,
        "predictions": predictions,
        "force_sensitivity": force_sensitivity,
        "geometry_sensitivities": {
            "radius_um": radius_sensitivity,
            "height_um": height_sensitivity,
        },
    }


def _build_diagnostics() -> tuple[dict[str, Any], Any, Any, np.ndarray]:
    inputs = _diagnostic_inputs()
    contact = build_contact_alignment_covariance(
        ContactAlignmentInputs(
            controls=tuple(float(value) for value in inputs["controls"]),
            predictions=tuple(float(value) for value in inputs["predictions"]),
            force_sensitivity=tuple(float(value) for value in inputs["force_sensitivity"]),
            curve_id="m3/contact_alignment_diagnostic",
        ),
        ContactAlignmentUncertaintyConfig(
            contact_offset_sigma=0.015,
            alignment_slope_sigma=0.02,
            displacement_scale_sigma=0.03,
            force_scale_sigma=0.01,
            minimum_variance=2.5e-5,
            jitter=1.0e-12,
            max_jitter=1.0e-8,
        ),
    )
    geometry = build_geometry_uncertainty_covariance(
        GeometrySensitivityInputs(
            predictions=tuple(float(value) for value in inputs["predictions"]),
            sensitivities={
                name: tuple(float(value) for value in values)
                for name, values in inputs["geometry_sensitivities"].items()
            },
            curve_id="m3/geometry_diagnostic",
        ),
        GeometryUncertaintyConfig(
            parameters=(
                GeometryParameterUncertainty("radius_um", sigma=0.012, units="micrometer", nominal=2.0),
                GeometryParameterUncertainty("height_um", sigma=0.04, units="micrometer", nominal=14.28),
            ),
            covariance=((1.44e-4, 1.2e-5), (1.2e-5, 1.6e-3)),
            jitter=1.0e-12,
            max_jitter=1.0e-8,
        ),
    )
    combined = contact.covariance.covariance + geometry.covariance.covariance
    return inputs, contact, geometry, combined


def _write_component_table(output_root: Path, inputs: dict[str, Any], contact: Any, geometry: Any, combined: np.ndarray) -> Path:
    path = output_root / "measurement_variance_components.csv"
    fieldnames = [
        "index",
        "control",
        "prediction",
        "contact_offset_variance",
        "alignment_tilt_variance",
        "displacement_scale_calibration_variance",
        "force_scale_calibration_variance",
        "minimum_variance_floor",
        "contact_alignment_total_variance",
        "geometry_radius_um_variance",
        "geometry_height_um_variance",
        "geometry_jacobian_variance",
        "combined_measurement_sigma",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (control, prediction) in enumerate(zip(inputs["controls"], inputs["predictions"])):
            writer.writerow(
                {
                    "index": index,
                    "control": float(control),
                    "prediction": float(prediction),
                    "contact_offset_variance": contact.variance_components["contact_offset"][index],
                    "alignment_tilt_variance": contact.variance_components["alignment_tilt"][index],
                    "displacement_scale_calibration_variance": contact.variance_components[
                        "displacement_scale_calibration"
                    ][index],
                    "force_scale_calibration_variance": contact.variance_components["force_scale_calibration"][index],
                    "minimum_variance_floor": contact.variance_components["minimum_variance_floor"][index],
                    "contact_alignment_total_variance": contact.variance_components["contact_alignment_total"][index],
                    "geometry_radius_um_variance": geometry.variance_components["geometry:radius_um"][index],
                    "geometry_height_um_variance": geometry.variance_components["geometry:height_um"][index],
                    "geometry_jacobian_variance": geometry.variance_components["geometry_jacobian"][index],
                    "combined_measurement_sigma": float(np.sqrt(max(combined[index, index], 0.0))),
                }
            )
    return path


def _write_plots(output_root: Path, inputs: dict[str, Any], contact: Any, geometry: Any, combined: np.ndarray) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controls = inputs["controls"]
    predictions = inputs["predictions"]

    heatmap_path = output_root / "measurement_covariance_components.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for ax, title, matrix in (
        (axes[0], "contact/alignment", contact.covariance.covariance),
        (axes[1], "geometry", geometry.covariance.covariance),
        (axes[2], "combined", combined),
    ):
        image = ax.imshow(matrix, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("point")
        ax.set_ylabel("point")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    band_path = output_root / "measurement_sigma_band.png"
    contact_sigma = np.asarray(contact.standard_deviation, dtype=float)
    geometry_sigma = np.asarray(geometry.standard_deviation, dtype=float)
    combined_sigma = np.sqrt(np.maximum(np.diag(combined), 0.0))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(controls, predictions, color="black", linewidth=2.0, label="prediction")
    ax.fill_between(
        controls,
        predictions - contact_sigma,
        predictions + contact_sigma,
        alpha=0.25,
        label="contact/alignment sigma",
    )
    ax.fill_between(
        controls,
        predictions - combined_sigma,
        predictions + combined_sigma,
        alpha=0.18,
        label="combined measurement sigma",
    )
    ax.plot(controls, geometry_sigma, linestyle="--", label="geometry sigma")
    ax.set_xlabel("normalized control")
    ax.set_ylabel("observable")
    ax.legend()
    fig.tight_layout()
    fig.savefig(band_path, dpi=150)
    plt.close(fig)

    return {
        "covariance_components_plot": heatmap_path.as_posix(),
        "measurement_sigma_band_plot": band_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M3 measurement hierarchy diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("_runs/noise/m3_measurement_diagnostics"))
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    inputs, contact, geometry, combined = _build_diagnostics()
    table_path = _write_component_table(output_root, inputs, contact, geometry, combined)
    plot_artifacts = _write_plots(output_root, inputs, contact, geometry, combined)
    summary_path = output_root / "measurement_covariance_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "contact_alignment": dict(contact.summary),
            "geometry": dict(geometry.summary),
            "combined_measurement_covariance": _matrix_summary(combined),
        },
    )
    manifest = {
        "schema_version": 1,
        "description": "M3 contact/alignment and geometry measurement-uncertainty diagnostics.",
        "artifacts": {
            "component_table": table_path.as_posix(),
            "covariance_summary": summary_path.as_posix(),
            **plot_artifacts,
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
