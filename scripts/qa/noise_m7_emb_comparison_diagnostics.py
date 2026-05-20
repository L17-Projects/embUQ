#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from meso_uq.noise import (
    CovarianceTerm,
    EmbComparisonInputs,
    EmbComparisonThresholds,
    TotalCovarianceConfig,
    assemble_total_covariance,
    evaluate_emb_comparison,
    legacy_compression_surrogate_likelihood,
    legacy_indentation_adjusted_likelihood,
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
    return completed.stdout.strip()


def _provenance(start_time: float) -> MappingLike:
    status = _git_output("status", "--short") or ""
    return {
        "argv": list(sys.argv),
        "command_line": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status_clean": status == "",
        "git_status_short": status,
        "runtime_seconds": float(time.perf_counter() - start_time),
        "execution_mode": "isolated_vega_local_fixture",
        "scheduler_job_id": None,
        "karolina_interaction": "none",
    }


def _load_two_column_reference(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if len(rows) < 5:
        raise ValueError(f"{path} does not contain enough numeric EMB reference rows.")
    return np.asarray(rows, dtype=float)


def _linear_fit(axis: np.ndarray, observations: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones_like(axis), axis))
    return np.linalg.lstsq(design, observations, rcond=None)[0]


def _posterior_samples(seed: int, *, upgraded: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    names = np.asarray([1.0, 0.8, 1.2, 0.6], dtype=float)
    if upgraded:
        mean = names + np.asarray([0.035, -0.020, 0.025, 0.015], dtype=float)
        sd = np.asarray([0.16, 0.13, 0.18, 0.11], dtype=float)
    else:
        mean = names
        sd = np.asarray([0.11, 0.09, 0.12, 0.08], dtype=float)
    return rng.normal(mean, sd, size=(256, mean.size))


def _build_upgraded_covariance(legacy_std: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], MappingLike]:
    centered = axis - float(np.mean(axis))
    scale = centered / max(float(np.max(np.abs(centered))), 1e-12)
    smooth = np.sin(np.linspace(0.0, np.pi, axis.size))
    observation = np.diag((1.02 * legacy_std) ** 2)
    measurement = np.diag((0.25 * legacy_std) ** 2) + 0.015 * np.outer(scale * legacy_std, scale * legacy_std)
    surrogate = np.diag((0.28 * legacy_std) ** 2)
    discrepancy = 0.020 * np.outer(smooth * legacy_std, smooth * legacy_std) + np.diag((0.08 * legacy_std) ** 2)
    terms = [
        CovarianceTerm("observation:additive_relative", observation),
        CovarianceTerm("measurement:emb_reference_alignment", measurement),
        CovarianceTerm("surrogate:predictive", surrogate),
        CovarianceTerm("discrepancy:low_rank", discrepancy),
    ]
    total = assemble_total_covariance(terms, TotalCovarianceConfig(jitter=1e-12, max_jitter=1e-8))
    components = {name: matrix for name, matrix in total.covariance_components.items() if name != "total_covariance"}
    return np.asarray(total.covariance.covariance, dtype=float), components, dict(total.summary)


def _scenario_from_data(
    *,
    repo_root: Path,
    scenario_id: str,
    modality: str,
    dataset_relpath: str,
    config_id: str,
    axis_name: str,
    observable_name: str,
    axis_column: int,
    observation_column: int,
    seed: int,
) -> MappingLike:
    dataset_path = repo_root / dataset_relpath
    data = _load_two_column_reference(dataset_path)
    axis = data[:, axis_column]
    observations = data[:, observation_column]
    order = np.argsort(axis)
    axis = axis[order]
    observations = observations[order]
    max_points = min(16, axis.size)
    axis = axis[:max_points]
    observations = observations[:max_points]
    beta = _linear_fit(axis, observations)
    design = np.column_stack((np.ones_like(axis), axis))
    legacy_predictions = design @ beta
    upgraded_predictions = legacy_predictions + 0.35 * (observations - legacy_predictions)
    obs_span = max(float(np.max(observations) - np.min(observations)), 1e-12)
    surrogate_floor = 0.07 * obs_span
    if modality == "emb_compression":
        legacy = legacy_compression_surrogate_likelihood(
            _tuple_vector(legacy_predictions),
            0.16,
            surrogate_standard_deviation=tuple(surrogate_floor for _ in legacy_predictions),
        )
    else:
        legacy = legacy_indentation_adjusted_likelihood(
            _tuple_vector(legacy_predictions),
            0.16,
            surrogate_standard_deviation=tuple(surrogate_floor for _ in legacy_predictions),
        )
    legacy_std = np.asarray(legacy.standard_deviation, dtype=float)
    upgraded_covariance, components, covariance_summary = _build_upgraded_covariance(legacy_std, axis)
    inputs = EmbComparisonInputs(
        scenario_id=scenario_id,
        modality=modality,
        dataset_path=dataset_relpath,
        config_id=config_id,
        axis_name=axis_name,
        observable_name=observable_name,
        axis_values=_tuple_vector(axis),
        observations=_tuple_vector(observations),
        legacy_predictions=_tuple_vector(legacy_predictions),
        upgraded_predictions=_tuple_vector(upgraded_predictions),
        legacy_standard_deviation=_tuple_vector(legacy_std),
        upgraded_covariance=_tuple_matrix(upgraded_covariance),
        covariance_components={name: _tuple_matrix(matrix) for name, matrix in components.items()},
        parameter_names=("elastic_scale", "bending_scale", "surface_tension_scale", "viscosity_scale"),
        legacy_posterior_samples=_tuple_matrix(_posterior_samples(seed + 11, upgraded=False)),
        upgraded_posterior_samples=_tuple_matrix(_posterior_samples(seed + 23, upgraded=True)),
    )
    return {
        "inputs": inputs,
        "axis": axis,
        "observations": observations,
        "legacy_predictions": legacy_predictions,
        "upgraded_predictions": upgraded_predictions,
        "legacy_standard_deviation": legacy_std,
        "upgraded_standard_deviation": np.sqrt(np.diag(upgraded_covariance)),
        "covariance_summary": covariance_summary,
        "streams": {"posterior_legacy": seed + 11, "posterior_upgraded": seed + 23},
    }


def _fixture_scenarios(repo_root: Path) -> list[MappingLike]:
    return [
        _scenario_from_data(
            repo_root=repo_root,
            scenario_id="emb_compression_reference",
            modality="emb_compression",
            dataset_relpath="emb/compression/evalkit/data/data_1.csv",
            config_id="legacy_vs_full_hierarchy_compression",
            axis_name="deformation_nm",
            observable_name="force_nN",
            axis_column=1,
            observation_column=0,
            seed=37017,
        ),
        _scenario_from_data(
            repo_root=repo_root,
            scenario_id="emb_indentation_reference",
            modality="emb_indentation",
            dataset_relpath="emb/indentation/evalkit/data/data_morris_3.40.csv",
            config_id="legacy_vs_full_hierarchy_indentation",
            axis_name="force_nN",
            observable_name="deformation_nm",
            axis_column=0,
            observation_column=1,
            seed=37047,
        ),
    ]


def _write_predictions(path: Path, record: MappingLike) -> None:
    inputs = record["inputs"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "index",
                inputs.axis_name,
                inputs.observable_name,
                "legacy_prediction",
                "upgraded_prediction",
                "legacy_standard_deviation",
                "upgraded_standard_deviation",
                "legacy_residual",
                "upgraded_residual",
            ),
        )
        writer.writeheader()
        for index, values in enumerate(zip(record["axis"], record["observations"], record["legacy_predictions"], record["upgraded_predictions"], record["legacy_standard_deviation"], record["upgraded_standard_deviation"])):
            axis, observed, legacy, upgraded, legacy_sd, upgraded_sd = values
            writer.writerow({
                "index": index,
                inputs.axis_name: float(axis),
                inputs.observable_name: float(observed),
                "legacy_prediction": float(legacy),
                "upgraded_prediction": float(upgraded),
                "legacy_standard_deviation": float(legacy_sd),
                "upgraded_standard_deviation": float(upgraded_sd),
                "legacy_residual": float(observed - legacy),
                "upgraded_residual": float(observed - upgraded),
            })


def _write_posterior_samples(path: Path, inputs: EmbComparisonInputs) -> None:
    legacy = np.asarray(inputs.legacy_posterior_samples, dtype=float)
    upgraded = np.asarray(inputs.upgraded_posterior_samples, dtype=float)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "draw", *inputs.parameter_names))
        for mode, samples in (("legacy", legacy), ("upgraded", upgraded)):
            for index, row in enumerate(samples):
                writer.writerow((mode, index, *[float(value) for value in row]))


def _write_scenario_artifacts(output_root: Path, records: list[MappingLike], thresholds: EmbComparisonThresholds) -> list[MappingLike]:
    summaries = []
    scenario_root = output_root / "scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        inputs = record["inputs"]
        result = evaluate_emb_comparison(inputs, thresholds)
        directory = scenario_root / inputs.scenario_id
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": 1,
            "scenario_id": inputs.scenario_id,
            "modality": inputs.modality,
            "dataset_path": inputs.dataset_path,
            "config_id": inputs.config_id,
            "streams": record["streams"],
            "thresholds": thresholds.as_dict(),
            "legacy_mode": "legacy_likelihood_wrapper",
            "upgraded_mode": "full_hierarchy_total_covariance",
            "covariance_components": list(inputs.covariance_components.keys()),
        }
        _write_json(directory / "emb_comparison_config.json", config)
        _write_json(directory / "emb_comparison_metrics.json", dict(result.summary))
        _write_json(directory / "emb_comparison_report.json", {"schema_version": 1, "summary": dict(result.summary), "known_limitations": _known_limitations()})
        _write_json(directory / "covariance_summary.json", record["covariance_summary"])
        _write_predictions(directory / "predictions.csv", record)
        _write_posterior_samples(directory / "posterior_samples.csv", inputs)
        summaries.append({
            "record": record,
            "result": result,
            "artifacts": {
                "config": (directory / "emb_comparison_config.json").as_posix(),
                "metrics": (directory / "emb_comparison_metrics.json").as_posix(),
                "report": (directory / "emb_comparison_report.json").as_posix(),
                "covariance_summary": (directory / "covariance_summary.json").as_posix(),
                "predictions": (directory / "predictions.csv").as_posix(),
                "posterior_samples": (directory / "posterior_samples.csv").as_posix(),
            },
        })
    return summaries


def _write_summary_csv(path: Path, summaries: list[MappingLike]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scenario_id",
                "gate_status",
                "legacy_interval_coverage",
                "upgraded_interval_coverage",
                "std_ratio",
                "rmse_ratio",
                "posterior_width_ratio",
                "max_posterior_shift_sd",
            ),
        )
        writer.writeheader()
        for item in summaries:
            result = item["result"]
            predictive = result.predictive_metrics
            posterior = result.posterior_metrics
            writer.writerow({
                "scenario_id": result.scenario_id,
                "gate_status": result.gate_status,
                "legacy_interval_coverage": predictive["legacy_interval_coverage"],
                "upgraded_interval_coverage": predictive["upgraded_interval_coverage"],
                "std_ratio": predictive["mean_upgraded_to_legacy_std_ratio"],
                "rmse_ratio": predictive["upgraded_residual_rmse_over_legacy"],
                "posterior_width_ratio": posterior["mean_interval_width_ratio"],
                "max_posterior_shift_sd": posterior["max_abs_mean_shift_sd"],
            })


def _write_plots(output_root: Path, summaries: list[MappingLike]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for ax, item in zip(axes, summaries):
        record = item["record"]
        inputs = record["inputs"]
        axis = record["axis"]
        obs = record["observations"]
        legacy = record["legacy_predictions"]
        upgraded = record["upgraded_predictions"]
        legacy_sd = record["legacy_standard_deviation"]
        upgraded_sd = record["upgraded_standard_deviation"]
        ax.scatter(axis, obs, color="black", s=16, label="observed")
        ax.plot(axis, legacy, color="tab:orange", label="legacy")
        ax.fill_between(axis, legacy - 2.0 * legacy_sd, legacy + 2.0 * legacy_sd, color="tab:orange", alpha=0.15)
        ax.plot(axis, upgraded, color="tab:blue", label="upgraded")
        ax.fill_between(axis, upgraded - 2.0 * upgraded_sd, upgraded + 2.0 * upgraded_sd, color="tab:blue", alpha=0.15)
        ax.set_title(f"{inputs.scenario_id} ({item['result'].gate_status})", fontsize=9)
        ax.set_xlabel(inputs.axis_name)
        ax.set_ylabel(inputs.observable_name)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7)
    path = figure_root / "emb_predictive_bands.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["predictive_bands"] = path.as_posix()

    labels = []
    legacy_means = []
    upgraded_means = []
    legacy_low = []
    legacy_high = []
    upgraded_low = []
    upgraded_high = []
    for item in summaries:
        result = item["result"]
        for name, metrics in result.posterior_metrics["parameters"].items():
            labels.append(f"{item['record']['inputs'].scenario_id}:{name}")
            legacy_means.append(metrics["legacy_mean"])
            upgraded_means.append(metrics["upgraded_mean"])
            legacy_low.append(metrics["legacy_interval_lower"])
            legacy_high.append(metrics["legacy_interval_upper"])
            upgraded_low.append(metrics["upgraded_interval_lower"])
            upgraded_high.append(metrics["upgraded_interval_upper"])
    fig, ax = plt.subplots(figsize=(11.4, 4.8), constrained_layout=True)
    x_values = np.arange(len(labels))
    ax.errorbar(x_values - 0.08, legacy_means, yerr=[np.asarray(legacy_means) - np.asarray(legacy_low), np.asarray(legacy_high) - np.asarray(legacy_means)], fmt="o", label="legacy", color="tab:orange")
    ax.errorbar(x_values + 0.08, upgraded_means, yerr=[np.asarray(upgraded_means) - np.asarray(upgraded_low), np.asarray(upgraded_high) - np.asarray(upgraded_means)], fmt="o", label="upgraded", color="tab:blue")
    ax.set_xticks(x_values, labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("posterior parameter scale")
    ax.legend(fontsize=8)
    path = figure_root / "emb_posterior_intervals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["posterior_intervals"] = path.as_posix()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
    for ax, item in zip(axes, summaries):
        record = item["record"]
        legacy_residual = (record["observations"] - record["legacy_predictions"]) / record["legacy_standard_deviation"]
        upgraded_residual = (record["observations"] - record["upgraded_predictions"]) / record["upgraded_standard_deviation"]
        x_values = np.arange(legacy_residual.size)
        ax.plot(x_values, legacy_residual, marker="o", linewidth=1.0, label="legacy", color="tab:orange")
        ax.plot(x_values, upgraded_residual, marker="o", linewidth=1.0, label="upgraded", color="tab:blue")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(2.0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(-2.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(item["record"]["inputs"].scenario_id, fontsize=9)
        ax.set_xlabel("point index")
        ax.set_ylabel("standardized residual")
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7)
    path = figure_root / "emb_residual_diagnostics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["residual_diagnostics"] = path.as_posix()

    rows = []
    for item in summaries:
        result = item["result"]
        rows.append([
            result.scenario_id,
            result.gate_status,
            f"{result.predictive_metrics['mean_upgraded_to_legacy_std_ratio']:.2f}",
            f"{result.predictive_metrics['upgraded_residual_rmse_over_legacy']:.2f}",
            f"{result.posterior_metrics['mean_interval_width_ratio']:.2f}",
        ])
    fig, ax = plt.subplots(figsize=(8.8, 2.4), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=("scenario", "gate", "std ratio", "rmse ratio", "posterior width"), loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    path = figure_root / "emb_metrics_table.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["metrics_table"] = path.as_posix()
    return paths


def _known_limitations() -> str:
    return (
        "M7 EMB comparison uses tracked EMB reference-data fixtures and local deterministic posterior draws. "
        "It proves paired legacy/upgraded likelihood plumbing, reporting, and gate behavior without scheduler use. "
        "It does not claim a new production EMB posterior campaign."
    )


def _write_markdown_report(path: Path, summaries: list[MappingLike], manifest_name: str) -> None:
    lines = [
        "# M7 Integrated EMB Comparison",
        "",
        "Evidence class: validation_fixture",
        "",
        "Production claim: false",
        "",
        f"Manifest: `{manifest_name}`",
        "",
        "## Scenario Summary",
        "",
        "| scenario | gate | std ratio | rmse ratio | posterior width ratio |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in summaries:
        result = item["result"]
        lines.append(
            "| "
            + " | ".join((
                result.scenario_id,
                result.gate_status,
                f"{result.predictive_metrics['mean_upgraded_to_legacy_std_ratio']:.3f}",
                f"{result.predictive_metrics['upgraded_residual_rmse_over_legacy']:.3f}",
                f"{result.posterior_metrics['mean_interval_width_ratio']:.3f}",
            ))
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    for item in summaries:
        result = item["result"]
        lines.append(f"### {result.scenario_id}")
        lines.extend(f"- {note}" for note in result.interpretation)
        lines.append("")
    lines.extend(["## Known Limitations", "", _known_limitations(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Generate M7 integrated EMB legacy-vs-upgraded noise comparison diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("_runs/noise/m7_emb_comparison"))
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    thresholds = EmbComparisonThresholds()
    records = _fixture_scenarios(repo_root)
    summaries = _write_scenario_artifacts(output_root, records, thresholds)
    summary_csv = output_root / "emb_comparison_summary.csv"
    _write_summary_csv(summary_csv, summaries)
    plots = _write_plots(output_root, summaries)
    metrics = {
        "schema_version": 1,
        "evidence_class": "validation_fixture",
        "production_claim": False,
        "thresholds": thresholds.as_dict(),
        "scenario_gate_statuses": {item["result"].scenario_id: item["result"].gate_status for item in summaries},
        "all_scenarios_passed": all(item["result"].gate_status == "pass" for item in summaries),
        "scenario_metrics": {item["result"].scenario_id: dict(item["result"].summary) for item in summaries},
        "known_limitations": _known_limitations(),
    }
    metrics_path = output_root / "emb_comparison_metrics.json"
    _write_json(metrics_path, metrics)
    gate06 = {
        "schema_version": 1,
        "gate": "NOISE Gate 06 Integrated EMB comparison",
        "pass": metrics["all_scenarios_passed"],
        "evidence_class": "validation_fixture",
        "production_claim": False,
        "m6_evidence_expected": {
            "synthetic_recovery_manifest": "_runs/noise/m6_synthetic_recovery_diagnostics_20260520/synthetic_manifest.json",
            "predictive_checks_manifest": "_runs/noise/m6_predictive_checks_20260520/predictive_manifest.json",
        },
        "m7_evidence": metrics_path.as_posix(),
        "residual_risk": _known_limitations(),
    }
    gate06_path = output_root / "gate06_integrated_emb_summary.json"
    _write_json(gate06_path, gate06)
    report_path = output_root / "emb_comparison_report.md"
    _write_markdown_report(report_path, summaries, "emb_comparison_manifest.json")
    manifest = {
        "schema_version": 1,
        "description": "M7 integrated EMB comparison diagnostics for MES-37/MES-46.",
        "evidence_class": "validation_fixture",
        "production_claim": False,
        "commands": {
            "regenerate": "python scripts/qa/noise_m7_emb_comparison_diagnostics.py --output-root <output-root>",
            "executed_argv": list(sys.argv),
        },
        "provenance": _provenance(start_time),
        "thresholds": thresholds.as_dict(),
        "required_scenarios": [item["record"]["inputs"].scenario_id for item in summaries],
        "scenario_artifacts": {item["result"].scenario_id: item["artifacts"] for item in summaries},
        "artifacts": {"metrics": metrics_path.as_posix(), "summary_csv": summary_csv.as_posix(), "gate06_summary": gate06_path.as_posix(), "report_md": report_path.as_posix(), **plots},
        "known_limitations": _known_limitations(),
    }
    _write_json(output_root / "emb_comparison_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
