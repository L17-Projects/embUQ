#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from meso_uq.noise import (
    DiscrepancyIdentifiabilityInputs,
    DiscrepancyIdentifiabilityThresholds,
    evaluate_discrepancy_identifiability,
)


MappingLike = dict[str, Any]


def _tuple_vector(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _tuple_matrix(values: Any) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(values, dtype=float)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")



def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _provenance() -> MappingLike:
    status = _git_output("status", "--short") or ""
    return {
        "argv": list(sys.argv),
        "command_line": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status_clean": status == "",
        "git_status_short": status or "",
    }


def _component_matrices(
    point_count: int,
    *,
    observation_variance: float,
    surrogate_covariance: np.ndarray | None = None,
    measurement_variance: float = 1.5e-4,
    discrepancy_covariance: np.ndarray | None = None,
) -> tuple[dict[str, tuple[tuple[float, ...], ...]], tuple[tuple[float, ...], ...]]:
    observation = observation_variance * np.eye(point_count)
    surrogate = 5.0e-4 * np.eye(point_count) if surrogate_covariance is None else np.asarray(surrogate_covariance, dtype=float)
    measurement = measurement_variance * np.eye(point_count)
    discrepancy = np.zeros((point_count, point_count), dtype=float) if discrepancy_covariance is None else np.asarray(discrepancy_covariance, dtype=float)
    total = observation + surrogate + measurement + discrepancy
    return {
        "observation:additive_relative": _tuple_matrix(observation),
        "surrogate:predictive": _tuple_matrix(surrogate),
        "measurement:geometry": _tuple_matrix(measurement),
        "discrepancy:low_rank": _tuple_matrix(discrepancy),
    }, _tuple_matrix(total)


def _fixture_records() -> list[MappingLike]:
    grid = np.linspace(0.0, 1.0, 80)
    centered = grid - float(np.mean(grid))
    base = 1.75 + 0.42 * grid
    wave_basis = np.column_stack((np.sin(2.0 * np.pi * grid), np.cos(2.0 * np.pi * grid)))
    curvature = centered * centered - float(np.mean(centered * centered))
    records: list[MappingLike] = []

    discrepancy = 0.010 * wave_basis[:, 0]
    noise = 0.040 * np.cos(4.0 * np.pi * grid)
    components, total = _component_matrices(
        grid.size,
        observation_variance=1.6e-3,
        discrepancy_covariance=1.0e-5 * np.eye(grid.size),
    )
    records.append({
        "fixture_id": "identifiable_residual",
        "description": "Small residual discrepancy weakly aligned with protected physical sensitivities.",
        "expected_gate_status": "pass",
        "grid": grid,
        "predictions_without": base,
        "predictions_with": base,
        "observations": base + discrepancy + noise,
        "discrepancy": discrepancy,
        "components": components,
        "inputs": DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(base + discrepancy + noise),
            predictions_without_discrepancy=_tuple_vector(base),
            discrepancy_mean=_tuple_vector(discrepancy),
            basis=_tuple_matrix(wave_basis),
            basis_names=("wave_sin", "wave_cos"),
            parameter_names=("offset",),
            parameters_without_discrepancy=(1.75,),
            parameters_with_discrepancy=(1.755,),
            parameter_posterior_sd_without_discrepancy=(0.080,),
            parameter_sensitivities={"offset": _tuple_vector(np.ones_like(grid))},
            coefficient_names=("wave_sin", "wave_cos"),
            coefficient_prior_sd=(0.10, 0.10),
            coefficient_posterior_mean=(0.010, 0.000),
            coefficient_posterior_sd=(0.020, 0.020),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
            fixture_id="identifiable_residual",
        ),
    })

    null_noise = 0.020 * np.sin(6.0 * np.pi * grid)
    components, total = _component_matrices(grid.size, observation_variance=6.0e-4)
    records.append({
        "fixture_id": "null_control",
        "description": "Physical model plus declared noise; discrepancy should shrink to zero.",
        "expected_gate_status": "pass",
        "grid": grid,
        "predictions_without": base,
        "predictions_with": base,
        "observations": base + null_noise,
        "discrepancy": np.zeros_like(grid),
        "components": components,
        "inputs": DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(base + null_noise),
            predictions_without_discrepancy=_tuple_vector(base),
            discrepancy_mean=_tuple_vector(np.zeros_like(grid)),
            basis=_tuple_matrix(wave_basis),
            basis_names=("wave_sin", "wave_cos"),
            parameter_names=("offset",),
            parameters_without_discrepancy=(1.75,),
            parameters_with_discrepancy=(1.75,),
            parameter_posterior_sd_without_discrepancy=(0.080,),
            parameter_sensitivities={"offset": _tuple_vector(np.ones_like(grid))},
            coefficient_names=("wave_sin", "wave_cos"),
            coefficient_prior_sd=(0.10, 0.10),
            coefficient_posterior_mean=(0.0, 0.0),
            coefficient_posterior_sd=(0.004, 0.004),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
            expect_discrepancy=False,
            fixture_id="null_control",
        ),
    })

    high_noise = 0.060 * np.cos(8.0 * np.pi * grid)
    components, total = _component_matrices(grid.size, observation_variance=4.0e-3)
    records.append({
        "fixture_id": "noise_only_control",
        "description": "Inflated observation noise without structural discrepancy.",
        "expected_gate_status": "pass",
        "grid": grid,
        "predictions_without": base,
        "predictions_with": base,
        "observations": base + high_noise,
        "discrepancy": np.zeros_like(grid),
        "components": components,
        "inputs": DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(base + high_noise),
            predictions_without_discrepancy=_tuple_vector(base),
            discrepancy_mean=_tuple_vector(np.zeros_like(grid)),
            basis=_tuple_matrix(wave_basis),
            basis_names=("wave_sin", "wave_cos"),
            coefficient_names=("wave_sin", "wave_cos"),
            coefficient_prior_sd=(0.10, 0.10),
            coefficient_posterior_mean=(0.0, 0.0),
            coefficient_posterior_sd=(0.004, 0.004),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
            expect_discrepancy=False,
            fixture_id="noise_only_control",
        ),
    })

    surrogate_shape = np.sin(3.0 * np.pi * grid)
    surrogate_covariance = 6.0e-4 * np.eye(grid.size) + 8.0e-4 * np.outer(surrogate_shape, surrogate_shape) / float(grid.size)
    surrogate_residual = 0.030 * surrogate_shape
    components, total = _component_matrices(grid.size, observation_variance=8.0e-4, surrogate_covariance=surrogate_covariance)
    records.append({
        "fixture_id": "surrogate_covariance_control",
        "description": "Structured residual assigned to surrogate covariance rather than discrepancy.",
        "expected_gate_status": "pass",
        "grid": grid,
        "predictions_without": base,
        "predictions_with": base,
        "observations": base + surrogate_residual,
        "discrepancy": np.zeros_like(grid),
        "components": components,
        "inputs": DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(base + surrogate_residual),
            predictions_without_discrepancy=_tuple_vector(base),
            discrepancy_mean=_tuple_vector(np.zeros_like(grid)),
            basis=_tuple_matrix(wave_basis),
            basis_names=("wave_sin", "wave_cos"),
            coefficient_names=("wave_sin", "wave_cos"),
            coefficient_prior_sd=(0.10, 0.10),
            coefficient_posterior_mean=(0.0, 0.0),
            coefficient_posterior_sd=(0.004, 0.004),
            covariance_components=components,
            total_covariance=total,
            discrepancy_opt_in=True,
            expect_discrepancy=False,
            fixture_id="surrogate_covariance_control",
        ),
    })

    slope_grid = np.linspace(-1.0, 1.0, 80)
    slope_base = 1.50 + 0.40 * slope_grid
    slope_discrepancy = 0.12 * slope_grid
    slope_components, slope_total = _component_matrices(
        slope_grid.size,
        observation_variance=1.6e-3,
        discrepancy_covariance=2.5e-3 * np.eye(slope_grid.size),
    )
    records.append({
        "fixture_id": "parameter_absorption_negative",
        "description": "Discrepancy basis reproduces the protected physical slope sensitivity.",
        "expected_gate_status": "fail",
        "grid": slope_grid,
        "predictions_without": slope_base,
        "predictions_with": slope_base,
        "observations": slope_base + slope_discrepancy,
        "discrepancy": slope_discrepancy,
        "components": slope_components,
        "inputs": DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(slope_base + slope_discrepancy),
            predictions_without_discrepancy=_tuple_vector(slope_base),
            discrepancy_mean=_tuple_vector(slope_discrepancy),
            basis=_tuple_matrix(slope_grid.reshape(-1, 1)),
            basis_names=("slope_like",),
            parameter_names=("slope",),
            parameters_without_discrepancy=(0.40,),
            parameters_with_discrepancy=(0.40,),
            parameter_posterior_sd_without_discrepancy=(0.050,),
            parameter_sensitivities={"slope": _tuple_vector(slope_grid)},
            coefficient_names=("slope_like",),
            coefficient_prior_sd=(0.20,),
            coefficient_posterior_mean=(0.12,),
            coefficient_posterior_sd=(0.010,),
            covariance_components=slope_components,
            total_covariance=slope_total,
            discrepancy_opt_in=True,
            negative_control=True,
            fixture_id="parameter_absorption_negative",
        ),
    })

    misspecified = 0.24 * curvature
    miss_components, miss_total = _component_matrices(
        grid.size,
        observation_variance=1.6e-3,
        discrepancy_covariance=1.4e-3 * np.eye(grid.size),
    )
    records.append({
        "fixture_id": "masked_curvature_negative",
        "description": "Missing curvature physics is hidden by a flexible discrepancy term.",
        "expected_gate_status": "fail",
        "grid": grid,
        "predictions_without": base,
        "predictions_with": base,
        "observations": base + misspecified,
        "discrepancy": misspecified,
        "components": miss_components,
        "inputs": DiscrepancyIdentifiabilityInputs(
            observations=_tuple_vector(base + misspecified),
            predictions_without_discrepancy=_tuple_vector(base),
            discrepancy_mean=_tuple_vector(misspecified),
            basis=_tuple_matrix(curvature.reshape(-1, 1)),
            basis_names=("curvature_like",),
            parameter_names=("slope",),
            parameters_without_discrepancy=(0.42,),
            parameters_with_discrepancy=(0.42,),
            parameter_posterior_sd_without_discrepancy=(0.050,),
            parameter_sensitivities={"slope": _tuple_vector(centered)},
            coefficient_names=("curvature_like",),
            coefficient_prior_sd=(0.20,),
            coefficient_posterior_mean=(0.24,),
            coefficient_posterior_sd=(0.012,),
            covariance_components=miss_components,
            total_covariance=miss_total,
            discrepancy_opt_in=True,
            negative_control=True,
            fixture_id="masked_curvature_negative",
        ),
    })
    return records


def _record_results(records: list[MappingLike], thresholds: DiscrepancyIdentifiabilityThresholds) -> list[MappingLike]:
    evaluated = []
    for record in records:
        result = evaluate_discrepancy_identifiability(record["inputs"], thresholds)
        evaluated.append({**record, "result": result, "summary": dict(result.summary)})
    return evaluated


def _write_fixture_csv(path: Path, records: list[MappingLike]) -> None:
    fieldnames = [
        "fixture_id",
        "expected_gate_status",
        "actual_gate_status",
        "discrepancy_rms_over_response",
        "discrepancy_explained_residual_share",
        "max_parameter_shift_sigma",
        "max_abs_theta_beta_correlation",
        "model_discrepancy_trace_share",
        "warnings",
        "failures",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            result = record["result"]
            writer.writerow({
                "fixture_id": record["fixture_id"],
                "expected_gate_status": record["expected_gate_status"],
                "actual_gate_status": result.gate_status,
                "discrepancy_rms_over_response": result.metrics["discrepancy_rms_over_response"],
                "discrepancy_explained_residual_share": result.metrics["discrepancy_explained_residual_share"],
                "max_parameter_shift_sigma": result.metrics["max_parameter_shift_sigma"],
                "max_abs_theta_beta_correlation": result.metrics["max_abs_theta_beta_correlation"],
                "model_discrepancy_trace_share": result.metrics["model_discrepancy_trace_share"],
                "warnings": " | ".join(result.warnings),
                "failures": " | ".join(result.failures),
            })


def _group_sigma(record: MappingLike, group: str) -> np.ndarray:
    components = record["components"]
    total = None
    for name, matrix in components.items():
        if _group_name(name) != group:
            continue
        values = np.asarray(matrix, dtype=float)
        total = values if total is None else total + values
    if total is None:
        return np.zeros_like(record["grid"], dtype=float)
    return np.sqrt(np.maximum(np.diag(total), 0.0))


def _group_name(name: str) -> str | None:
    if name.startswith("observation:"):
        return "observation"
    if name.startswith("surrogate:"):
        return "surrogate"
    if name.startswith("measurement:"):
        return "measurement"
    if name.startswith("discrepancy:") or name.startswith("model_discrepancy"):
        return "model_discrepancy"
    return None


def _write_plots(output_root: Path, records: list[MappingLike]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.0), constrained_layout=True)
    for ax, record in zip(axes.ravel(), records):
        grid = np.asarray(record["grid"], dtype=float)
        raw = np.asarray(record["observations"], dtype=float) - np.asarray(record["predictions_without"], dtype=float)
        discrepancy = np.asarray(record["discrepancy"], dtype=float)
        final = np.asarray(record["observations"], dtype=float) - np.asarray(record["predictions_with"], dtype=float) - discrepancy
        obs_sigma = _group_sigma(record, "observation")
        surrogate_sigma = _group_sigma(record, "surrogate")
        ax.fill_between(grid, -obs_sigma, obs_sigma, color="tab:blue", alpha=0.12, label="observation sigma")
        ax.fill_between(grid, -surrogate_sigma, surrogate_sigma, color="tab:green", alpha=0.12, label="surrogate sigma")
        ax.plot(grid, raw, color="black", linewidth=1.5, label="raw residual")
        ax.plot(grid, discrepancy, color="tab:red", linestyle="--", label="discrepancy mean")
        ax.plot(grid, final, color="tab:purple", linewidth=1.2, label="final residual")
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.set_title(f"{record['fixture_id']} ({record['result'].gate_status})", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    residual_path = output_root / "residual_decomposition.png"
    fig.savefig(residual_path, dpi=150)
    plt.close(fig)
    paths["residual_decomposition_plot"] = residual_path.as_posix()

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    labels = [record["fixture_id"] for record in records]
    values = [record["result"].metrics["max_parameter_shift_sigma"] or 0.0 for record in records]
    ax.bar(labels, values, color="tab:orange")
    ax.axhline(0.5, color="0.25", linestyle="--", linewidth=1.0, label="warn")
    ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1.0, label="fail")
    ax.set_ylabel("max parameter shift (posterior SD)")
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.legend(fontsize=8)
    path = output_root / "parameter_shift_gate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["parameter_shift_plot"] = path.as_posix()

    absorption = next(record for record in records if record["fixture_id"] == "parameter_absorption_negative")
    correlation = np.asarray(absorption["result"].theta_beta_correlation, dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
    image = ax.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(correlation.shape[1]), absorption["inputs"].coefficient_names)
    ax.set_yticks(range(correlation.shape[0]), absorption["inputs"].parameter_names)
    ax.set_title("theta-beta correlation")
    fig.colorbar(image, ax=ax, shrink=0.82)
    path = output_root / "theta_beta_correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["theta_beta_correlation_heatmap"] = path.as_posix()

    groups = ("observation", "surrogate", "measurement", "model_discrepancy")
    fig, ax = plt.subplots(figsize=(8.4, 4.2), constrained_layout=True)
    bottom = np.zeros(len(records), dtype=float)
    for group in groups:
        shares = [record["result"].covariance_group_trace_shares.get(group) or 0.0 for record in records]
        ax.bar(labels, shares, bottom=bottom, label=group)
        bottom += np.asarray(shares, dtype=float)
    ax.set_ylabel("trace share")
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.legend(fontsize=8, ncols=2)
    path = output_root / "covariance_trace_shares.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["covariance_trace_share_plot"] = path.as_posix()

    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    shrinkage_labels = []
    shrinkage_values = []
    active_values = []
    for record in records:
        for name, payload in record["result"].coefficient_shrinkage.items():
            shrinkage_labels.append(f"{record['fixture_id']}:{name}")
            shrinkage_values.append(payload["posterior_sd_over_prior_sd"] or 0.0)
            active_values.append(payload["active"])
    colors = ["tab:red" if active else "tab:blue" for active in active_values]
    ax.bar(range(len(shrinkage_values)), shrinkage_values, color=colors)
    ax.set_xticks(range(len(shrinkage_labels)), shrinkage_labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("posterior SD / prior SD")
    ax.set_title("coefficient shrinkage")
    path = output_root / "coefficient_shrinkage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["coefficient_shrinkage_plot"] = path.as_posix()

    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    ax.axis("off")
    table_rows = []
    for record in records:
        result = record["result"]
        table_rows.append([
            record["fixture_id"],
            record["expected_gate_status"],
            result.gate_status,
            f"{result.metrics['discrepancy_rms_over_response']:.3f}",
            f"{result.metrics['discrepancy_explained_residual_share']:.3f}",
            f"{result.metrics['max_abs_theta_beta_correlation']:.3f}",
        ])
    table = ax.table(
        cellText=table_rows,
        colLabels=("fixture", "expected", "actual", "disc/y", "resid share", "max corr"),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    path = output_root / "fixture_gate_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["fixture_gate_summary_plot"] = path.as_posix()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M5 discrepancy identifiability diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("_runs/noise/m5_discrepancy_identifiability_diagnostics"))
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    thresholds = DiscrepancyIdentifiabilityThresholds()
    records = _record_results(_fixture_records(), thresholds)
    summary_path = output_root / "identifiability_metrics.json"
    table_path = output_root / "fixture_summary.csv"
    plots = _write_plots(output_root, records)
    _write_fixture_csv(table_path, records)

    fixture_summaries = {record["fixture_id"]: record["summary"] for record in records}
    expected_matches = {
        record["fixture_id"]: record["expected_gate_status"] == record["result"].gate_status
        for record in records
    }
    negative_controls_failed = all(
        record["result"].gate_status == "fail"
        for record in records
        if record["inputs"].negative_control
    )
    passing_controls_passed = all(
        record["result"].gate_status == "pass"
        for record in records
        if not record["inputs"].negative_control
    )
    metrics = {
        "schema_version": 1,
        "thresholds": thresholds.as_dict(),
        "fixture_gate_matches_expected": expected_matches,
        "all_expected_gate_statuses_matched": all(expected_matches.values()),
        "negative_controls_failed": negative_controls_failed,
        "passing_controls_passed": passing_controls_passed,
        "fixture_gate_statuses": {record["fixture_id"]: record["result"].gate_status for record in records},
        "fixture_descriptions": {record["fixture_id"]: record["description"] for record in records},
        "fixtures": fixture_summaries,
    }
    _write_json(summary_path, metrics)
    residual_risk_notes = (
        "Synthetic controls cover default-disabled safety, shrinkage, covariance decomposition, "
        "parameter absorption, and masked-curvature misspecification. Production EMB thresholds "
        "remain interpretive until calibrated on real posterior samples in later milestones."
    )
    manifest = {
        "schema_version": 1,
        "description": "M5 discrepancy identifiability and safety-gate diagnostics for MES-36/MES-45.",
        "commands": {
            "regenerate": "python scripts/qa/noise_m5_discrepancy_identifiability_diagnostics.py --output-root <output-root>",
            "executed_argv": list(sys.argv),
        },
        "provenance": _provenance(),
        "thresholds": thresholds.as_dict(),
        "fixture_gate_statuses": metrics["fixture_gate_statuses"],
        "fixture_gate_matches_expected": expected_matches,
        "residual_risk_notes": residual_risk_notes,
        "artifacts": {
            "metrics": summary_path.as_posix(),
            "fixture_summary": table_path.as_posix(),
            **plots,
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
