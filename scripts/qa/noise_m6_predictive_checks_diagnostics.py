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
    PredictiveCheckInputs,
    PredictiveCheckThresholds,
    SbcRankRecord,
    evaluate_predictive_checks,
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
    }


def _squared_exponential(points: np.ndarray, amplitude: float, length_scale: float) -> np.ndarray:
    distance = points[:, None] - points[None, :]
    return amplitude * amplitude * np.exp(-0.5 * (distance / length_scale) ** 2)


def _sbc_records(prefix: str, *, seed: int, record_count: int = 40, draw_count: int = 31) -> tuple[SbcRankRecord, ...]:
    rng = np.random.default_rng(seed)
    quantiles = (np.arange(record_count, dtype=float) + 0.5) / record_count
    rng.shuffle(quantiles)
    records: list[SbcRankRecord] = []
    for index, quantile in enumerate(quantiles):
        samples = np.sort(rng.normal(loc=0.03 * np.sin(index), scale=1.0 + 0.01 * (index % 5), size=draw_count))
        true_value = float(np.quantile(samples, quantile))
        records.append(SbcRankRecord(f"{prefix}_theta_{index:03d}", true_value, _tuple_vector(samples)))
    return tuple(records)


def _draw_predictive_samples(design: np.ndarray, theta: np.ndarray, covariance: np.ndarray, *, seed: int, draw_count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parameter_covariance = np.diag([0.0016, 0.0009])
    theta_draws = rng.multivariate_normal(theta, parameter_covariance, size=draw_count)
    noise_draws = rng.multivariate_normal(np.zeros(design.shape[0]), covariance, size=draw_count)
    return theta_draws @ design.T + noise_draws


def _scenario(
    scenario_id: str,
    *,
    seed: int,
    x: np.ndarray,
    theta: np.ndarray,
    components: dict[str, np.ndarray],
    description: str,
    likelihood_stage: str,
) -> MappingLike:
    design = np.column_stack((np.ones_like(x), x))
    expected = design @ theta
    covariance = sum(components.values(), np.zeros((x.size, x.size), dtype=float))
    residual = 0.007 * np.cos(np.pi * x) + 0.004 * np.sin(2.0 * np.pi * x)
    observations = expected + residual
    predictive_samples = _draw_predictive_samples(design, theta, covariance, seed=seed + 201, draw_count=256)
    inputs = PredictiveCheckInputs(
        scenario_id=scenario_id,
        seed=seed,
        observed=_tuple_vector(observations),
        predictive_samples=_tuple_matrix(predictive_samples),
        sbc_rank_records=_sbc_records(scenario_id, seed=seed + 301),
    )
    return {
        "scenario_id": scenario_id,
        "description": description,
        "likelihood_stage": likelihood_stage,
        "seed": seed,
        "streams": {"truth": seed, "predictive_samples": seed + 201, "sbc_rank_records": seed + 301},
        "x": x,
        "design": design,
        "theta": theta,
        "expected": expected,
        "observations": observations,
        "predictive_samples": predictive_samples,
        "components": components,
        "total_covariance": covariance,
        "inputs": inputs,
    }


def _fixture_scenarios() -> list[MappingLike]:
    x = np.linspace(-1.0, 1.0, 48)
    theta = np.asarray((1.2, 0.42), dtype=float)
    eye = np.eye(x.size, dtype=float)
    centered = x - float(np.mean(x))
    geometry_shape = centered * centered - float(np.mean(centered * centered))
    discrepancy_shape = np.sin(2.0 * np.pi * (x - np.min(x)) / (np.max(x) - np.min(x)))
    return [
        _scenario(
            "baseline_legacy",
            seed=35017,
            x=x,
            theta=theta,
            components={"observation:additive_relative": 0.0025 * eye},
            description="Baseline legacy-equivalent PPC/SBC fixture with diagonal observation noise.",
            likelihood_stage="M1/M2",
        ),
        _scenario(
            "full_hierarchy",
            seed=35047,
            x=x,
            theta=theta,
            components={
                "observation:additive_relative": 0.0012 * eye,
                "measurement:geometry": 0.00018 * np.outer(geometry_shape, geometry_shape) + 0.00018 * eye,
                "surrogate:predictive": _squared_exponential(x, amplitude=0.016, length_scale=0.42) + 0.00018 * eye,
                "discrepancy:low_rank": 0.00035 * np.outer(discrepancy_shape, discrepancy_shape) + 0.00010 * eye,
            },
            description="Full M5 hierarchy PPC/SBC fixture with measurement, surrogate, and discrepancy covariance.",
            likelihood_stage="M5",
        ),
    ]


def _write_observations(path: Path, record: MappingLike) -> None:
    samples = record["predictive_samples"]
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    mean = np.mean(samples, axis=0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "x", "truth", "observation", "predictive_mean", "predictive_lower", "predictive_upper"))
        writer.writeheader()
        for index, values in enumerate(zip(record["x"], record["expected"], record["observations"], mean, lower, upper)):
            x_value, truth, observed, predicted, low, high = values
            writer.writerow({
                "index": index,
                "x": float(x_value),
                "truth": float(truth),
                "observation": float(observed),
                "predictive_mean": float(predicted),
                "predictive_lower": float(low),
                "predictive_upper": float(high),
            })


def _write_rank_records(path: Path, records: tuple[SbcRankRecord, ...], thresholds: PredictiveCheckThresholds) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "true_value", "draw_count", "rank", "rank_quantile", "interval_covered"))
        writer.writeheader()
        for record in records:
            summary = record.as_dict(alpha=thresholds.interval_alpha)
            writer.writerow({key: summary[key] for key in writer.fieldnames})


def _write_scenario_artifacts(output_root: Path, records: list[MappingLike], thresholds: PredictiveCheckThresholds) -> list[MappingLike]:
    summaries = []
    scenario_root = output_root / "scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        result = evaluate_predictive_checks(record["inputs"], thresholds)
        directory = scenario_root / record["scenario_id"]
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": 1,
            "scenario_id": record["scenario_id"],
            "description": record["description"],
            "likelihood_stage": record["likelihood_stage"],
            "seed": record["seed"],
            "streams": record["streams"],
            "thresholds": thresholds.as_dict(),
            "covariance_components": list(record["components"].keys()),
        }
        _write_json(directory / "predictive_config.json", config)
        _write_json(directory / "predictive_metrics.json", dict(result.summary))
        _write_json(directory / "predictive_report.json", {"schema_version": 1, "summary": dict(result.summary), "known_limitations": _known_limitations()})
        _write_observations(directory / "observations.csv", record)
        _write_rank_records(directory / "sbc_rank_records.csv", record["inputs"].sbc_rank_records, thresholds)
        summaries.append({
            "record": record,
            "result": result,
            "artifacts": {
                "predictive_config": (directory / "predictive_config.json").as_posix(),
                "observations": (directory / "observations.csv").as_posix(),
                "sbc_rank_records": (directory / "sbc_rank_records.csv").as_posix(),
                "predictive_metrics": (directory / "predictive_metrics.json").as_posix(),
                "predictive_report": (directory / "predictive_report.json").as_posix(),
            },
        })
    return summaries


def _write_summary_csv(path: Path, summaries: list[MappingLike]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scenario_id", "gate_status", "finite_fraction", "pointwise_interval_coverage", "pointwise_rank_edge_fraction", "sbc_rank_histogram_l1", "sbc_interval_coverage"),
        )
        writer.writeheader()
        for item in summaries:
            result = item["result"]
            writer.writerow({
                "scenario_id": result.scenario_id,
                "gate_status": result.gate_status,
                "finite_fraction": result.ppc_metrics["finite_fraction"],
                "pointwise_interval_coverage": result.ppc_metrics["pointwise_metrics"]["interval_coverage"],
                "pointwise_rank_edge_fraction": result.ppc_metrics["pointwise_metrics"]["rank_edge_fraction"],
                "sbc_rank_histogram_l1": result.sbc_metrics["rank_histogram_l1"],
                "sbc_interval_coverage": result.sbc_metrics["interval_coverage"],
            })


def _write_plots(output_root: Path, summaries: list[MappingLike]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    for ax, item in zip(axes, summaries):
        record = item["record"]
        samples = record["predictive_samples"]
        lower = np.quantile(samples, 0.05, axis=0)
        upper = np.quantile(samples, 0.95, axis=0)
        mean = np.mean(samples, axis=0)
        ax.fill_between(record["x"], lower, upper, color="tab:blue", alpha=0.16, label="90% PPC interval")
        ax.plot(record["x"], mean, color="tab:blue", linewidth=1.4, label="predictive mean")
        ax.plot(record["x"], record["expected"], color="black", linewidth=1.2, label="truth")
        ax.scatter(record["x"], record["observations"], s=12, color="tab:orange", alpha=0.75, label="observed")
        ax.set_title(f"{record['scenario_id']} ({item['result'].gate_status})", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc="best")
    path = figure_root / "ppc_observable_overlay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["ppc_observable_overlay"] = path.as_posix()

    labels = []
    observed = []
    centers = []
    lows = []
    highs = []
    for item in summaries:
        for name, metrics in item["result"].ppc_metrics["summary_metrics"].items():
            labels.append(f"{item['record']['scenario_id']}:{name}")
            observed.append(metrics["observed"])
            centers.append(metrics["predictive_mean"])
            lows.append(metrics["interval_lower"])
            highs.append(metrics["interval_upper"])
    fig, ax = plt.subplots(figsize=(10.8, 4.2), constrained_layout=True)
    x_values = np.arange(len(labels))
    lower_error = np.asarray(centers) - np.asarray(lows)
    upper_error = np.asarray(highs) - np.asarray(centers)
    ax.errorbar(x_values, centers, yerr=[lower_error, upper_error], fmt="o", color="tab:blue", label="predictive 90% interval")
    ax.scatter(x_values, observed, marker="x", color="black", label="observed")
    ax.set_xticks(x_values, labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("summary value")
    ax.legend(fontsize=8)
    path = figure_root / "ppc_summary_intervals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["ppc_summary_intervals"] = path.as_posix()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for ax, item in zip(axes, summaries):
        records = item["result"].sbc_metrics["records"]
        rank_quantiles = [record["rank_quantile"] for record in records]
        ax.hist(rank_quantiles, bins=5, range=(0.0, 1.0), color="tab:green", alpha=0.72)
        ax.axhline(len(rank_quantiles) / 5.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(item["record"]["scenario_id"], fontsize=9)
        ax.set_xlabel("SBC rank quantile")
        ax.set_ylabel("count")
        ax.tick_params(labelsize=8)
    path = figure_root / "sbc_rank_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["sbc_rank_histogram"] = path.as_posix()

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    scenario_labels = [item["record"]["scenario_id"] for item in summaries]
    ppc_coverage = [item["result"].ppc_metrics["pointwise_metrics"]["interval_coverage"] for item in summaries]
    sbc_coverage = [item["result"].sbc_metrics["interval_coverage"] for item in summaries]
    x_values = np.arange(len(scenario_labels))
    width = 0.35
    ax.bar(x_values - width / 2.0, ppc_coverage, width, label="PPC pointwise coverage", color="tab:blue")
    ax.bar(x_values + width / 2.0, sbc_coverage, width, label="SBC interval coverage", color="tab:green")
    ax.set_xticks(x_values, scenario_labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    path = figure_root / "calibration_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["calibration_summary"] = path.as_posix()
    return paths


def _known_limitations() -> str:
    return (
        "M6 predictive checks use deterministic lightweight posterior predictive and SBC rank fixtures. "
        "They prove summary/reporting/calibration plumbing for baseline and full-hierarchy configurations, "
        "but do not replace production EMB posterior calibration or the MES-37 EMB comparison gate."
    )


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Generate M6 posterior predictive and SBC diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("_runs/noise/m6_predictive_checks"))
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    thresholds = PredictiveCheckThresholds()
    records = _fixture_scenarios()
    summaries = _write_scenario_artifacts(output_root, records, thresholds)
    summary_csv = output_root / "predictive_summary.csv"
    _write_summary_csv(summary_csv, summaries)
    plots = _write_plots(output_root, summaries)
    metrics = {
        "schema_version": 1,
        "thresholds": thresholds.as_dict(),
        "scenario_gate_statuses": {item["result"].scenario_id: item["result"].gate_status for item in summaries},
        "all_scenarios_passed": all(item["result"].gate_status == "pass" for item in summaries),
        "scenario_metrics": {item["result"].scenario_id: dict(item["result"].summary) for item in summaries},
        "known_limitations": _known_limitations(),
    }
    metrics_path = output_root / "predictive_check_metrics.json"
    _write_json(metrics_path, metrics)
    manifest = {
        "schema_version": 1,
        "description": "M6 posterior predictive and SBC diagnostics for MES-35.",
        "commands": {
            "regenerate": "python scripts/qa/noise_m6_predictive_checks_diagnostics.py --output-root <output-root>",
            "executed_argv": list(sys.argv),
        },
        "provenance": _provenance(start_time),
        "thresholds": thresholds.as_dict(),
        "required_scenarios": [record["scenario_id"] for record in records],
        "scenario_artifacts": {item["result"].scenario_id: item["artifacts"] for item in summaries},
        "artifacts": {"metrics": metrics_path.as_posix(), "summary_csv": summary_csv.as_posix(), **plots},
        "known_limitations": _known_limitations(),
    }
    _write_json(output_root / "predictive_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
