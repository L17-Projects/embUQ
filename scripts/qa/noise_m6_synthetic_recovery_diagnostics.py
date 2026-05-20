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

from meso_uq.noise import SyntheticRecoveryInputs, SyntheticRecoveryThresholds, evaluate_synthetic_recovery


MappingLike = dict[str, Any]


def _tuple_vector(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _tuple_matrix(values: Any) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(values, dtype=float)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_ref(output_root: Path, path: Path) -> str:
    return path.relative_to(output_root).as_posix()


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


def _provenance() -> MappingLike:
    status = _git_output("status", "--short") or ""
    return {
        "argv": list(sys.argv),
        "command_line": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status_clean": status == "",
        "git_status_short": status,
    }


def _squared_exponential(points: np.ndarray, amplitude: float, length_scale: float) -> np.ndarray:
    distance = points[:, None] - points[None, :]
    return amplitude * amplitude * np.exp(-0.5 * (distance / length_scale) ** 2)


def _orthogonal_residual(design: np.ndarray, covariance: np.ndarray, seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidate = rng.normal(size=design.shape[0])
    precision_design = np.linalg.solve(covariance, design)
    precision_candidate = np.linalg.solve(covariance, candidate)
    normal = design.T @ precision_design
    correction = np.linalg.solve(normal, design.T @ precision_candidate)
    residual = candidate - design @ correction
    rms = float(np.sqrt(np.mean(residual * residual)))
    if rms <= 0.0:
        raise RuntimeError("synthetic residual projection collapsed to zero")
    return residual * (scale / rms)


def _component_summary(components: MappingLike, total: np.ndarray) -> MappingLike:
    total_trace = float(np.trace(total))
    payload: MappingLike = {}
    for name, matrix in components.items():
        values = np.asarray(matrix, dtype=float)
        eigenvalues = np.linalg.eigvalsh(0.5 * values + 0.5 * values.T)
        payload[name] = {
            "trace": float(np.trace(values)),
            "trace_share": float(np.trace(values) / total_trace),
            "diagonal_min": float(np.min(np.diag(values))),
            "diagonal_max": float(np.max(np.diag(values))),
            "min_eigenvalue": float(np.min(eigenvalues)),
            "max_eigenvalue": float(np.max(eigenvalues)),
        }
    total_eigenvalues = np.linalg.eigvalsh(0.5 * total + 0.5 * total.T)
    payload["total"] = {
        "trace": total_trace,
        "diagonal_min": float(np.min(np.diag(total))),
        "diagonal_max": float(np.max(np.diag(total))),
        "min_eigenvalue": float(np.min(total_eigenvalues)),
        "max_eigenvalue": float(np.max(total_eigenvalues)),
        "condition_number": float(np.linalg.cond(total)),
    }
    return payload


def _scenario(
    scenario_id: str,
    *,
    seed: int,
    x: np.ndarray,
    theta: np.ndarray,
    components: dict[str, np.ndarray],
    residual_scale: float,
    description: str,
    stage: str,
) -> MappingLike:
    design = np.column_stack((np.ones_like(x), x))
    total = sum(components.values(), np.zeros((x.size, x.size), dtype=float))
    residual = _orthogonal_residual(design, total, seed + 101, residual_scale)
    expected = design @ theta
    observations = expected + residual
    inputs = SyntheticRecoveryInputs(
        scenario_id=scenario_id,
        seed=seed,
        design_matrix=_tuple_matrix(design),
        observations=_tuple_vector(observations),
        true_parameters=_tuple_vector(theta),
        parameter_names=("offset", "slope"),
        total_covariance=_tuple_matrix(total),
        covariance_components={name: _tuple_matrix(matrix) for name, matrix in components.items()},
        expected_observables=_tuple_vector(expected),
    )
    return {
        "scenario_id": scenario_id,
        "description": description,
        "stage": stage,
        "seed": seed,
        "streams": {
            "truth": seed,
            "measurement_noise": seed + 101,
            "surrogate_noise": seed + 102,
            "discrepancy": seed + 103,
            "posterior_fixture": seed + 104,
        },
        "x": x,
        "design": design,
        "truth": theta,
        "expected": expected,
        "observations": observations,
        "residual": residual,
        "components": components,
        "total_covariance": total,
        "inputs": inputs,
    }


def _fixture_scenarios() -> list[MappingLike]:
    x = np.linspace(-1.0, 1.0, 48)
    theta = np.asarray((1.20, 0.42), dtype=float)
    eye = np.eye(x.size, dtype=float)
    centered = x - float(np.mean(x))
    geometry_shape = centered * centered - float(np.mean(centered * centered))
    discrepancy_shape = np.sin(2.0 * np.pi * (x - np.min(x)) / (np.max(x) - np.min(x)))

    return [
        _scenario(
            "legacy_noise_only",
            seed=34017,
            x=x,
            theta=theta,
            components={"observation:additive_relative": 0.0016 * eye},
            residual_scale=0.010,
            description="Legacy-equivalent diagonal Gaussian observation noise only.",
            stage="M1/M2",
        ),
        _scenario(
            "measurement_uncertainty",
            seed=34027,
            x=x,
            theta=theta,
            components={
                "observation:additive_relative": 0.0010 * eye,
                "measurement:contact_alignment": 0.00010 * np.outer(np.ones_like(x), np.ones_like(x)) + 0.00020 * eye,
                "measurement:geometry": 0.00035 * np.outer(geometry_shape, geometry_shape) + 0.00015 * eye,
            },
            residual_scale=0.012,
            description="Observation plus contact/alignment and geometry measurement uncertainty.",
            stage="M3",
        ),
        _scenario(
            "surrogate_covariance",
            seed=34037,
            x=x,
            theta=theta,
            components={
                "observation:additive_relative": 0.0010 * eye,
                "surrogate:predictive": _squared_exponential(x, amplitude=0.018, length_scale=0.45) + 0.00025 * eye,
            },
            residual_scale=0.012,
            description="Observation plus correlated surrogate predictive covariance.",
            stage="M4",
        ),
        _scenario(
            "discrepancy_enabled",
            seed=34047,
            x=x,
            theta=theta,
            components={
                "observation:additive_relative": 0.0010 * eye,
                "surrogate:predictive": 0.00025 * eye,
                "discrepancy:low_rank": 0.00055 * np.outer(discrepancy_shape, discrepancy_shape) + 0.00010 * eye,
            },
            residual_scale=0.012,
            description="Full hierarchy fixture with opt-in low-rank discrepancy covariance.",
            stage="M5",
        ),
    ]


def _write_observations(path: Path, record: MappingLike, result: Any) -> None:
    estimates = np.asarray([result.parameter_estimates["offset"], result.parameter_estimates["slope"]], dtype=float)
    fitted = record["design"] @ estimates
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "x", "truth", "observation", "fitted", "residual"))
        writer.writeheader()
        for index, (x_value, truth, observed, fit) in enumerate(zip(record["x"], record["expected"], record["observations"], fitted)):
            writer.writerow({
                "index": index,
                "x": float(x_value),
                "truth": float(truth),
                "observation": float(observed),
                "fitted": float(fit),
                "residual": float(observed - fit),
            })


def _write_scenario_artifacts(output_root: Path, records: list[MappingLike], thresholds: SyntheticRecoveryThresholds) -> list[MappingLike]:
    summaries = []
    scenario_root = output_root / "scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        result = evaluate_synthetic_recovery(record["inputs"], thresholds)
        directory = scenario_root / record["scenario_id"]
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": 1,
            "scenario_id": record["scenario_id"],
            "description": record["description"],
            "likelihood_stage": record["stage"],
            "seed": record["seed"],
            "streams": record["streams"],
            "parameter_names": list(record["inputs"].parameter_names),
            "thresholds": thresholds.as_dict(),
            "covariance_components": list(record["components"].keys()),
        }
        truth = {
            "schema_version": 1,
            "scenario_id": record["scenario_id"],
            "parameters": {name: float(value) for name, value in zip(record["inputs"].parameter_names, record["truth"])},
        }
        covariance_summary = _component_summary(record["components"], record["total_covariance"])
        _write_json(directory / "synthetic_config.json", config)
        _write_json(directory / "truth.json", truth)
        _write_json(directory / "covariance_summary.json", covariance_summary)
        _write_json(directory / "recovery_metrics.json", dict(result.summary))
        _write_json(directory / "recovery_report.json", {"schema_version": 1, "summary": dict(result.summary), "residual_risk": _residual_risk_notes()})
        _write_observations(directory / "observations.csv", record, result)
        summaries.append({
            "record": record,
            "result": result,
            "artifacts": {
                "synthetic_config": (directory / "synthetic_config.json").as_posix(),
                "truth": (directory / "truth.json").as_posix(),
                "observations": (directory / "observations.csv").as_posix(),
                "covariance_summary": (directory / "covariance_summary.json").as_posix(),
                "recovery_metrics": (directory / "recovery_metrics.json").as_posix(),
                "recovery_report": (directory / "recovery_report.json").as_posix(),
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
                "max_abs_parameter_bias",
                "max_parameter_z_error",
                "parameter_interval_coverage",
                "residual_rmse_over_observation_rms",
                "finite_observables",
            ),
        )
        writer.writeheader()
        for item in summaries:
            result = item["result"]
            writer.writerow({
                "scenario_id": result.scenario_id,
                "gate_status": result.gate_status,
                "max_abs_parameter_bias": result.metrics["max_abs_parameter_bias"],
                "max_parameter_z_error": result.metrics["max_parameter_z_error"],
                "parameter_interval_coverage": result.metrics["parameter_interval_coverage"],
                "residual_rmse_over_observation_rms": result.metrics["residual_rmse_over_observation_rms"],
                "finite_observables": result.metrics["finite_observables"],
            })


def _whitened_residual(record: MappingLike, result: Any) -> np.ndarray:
    estimates = np.asarray([result.parameter_estimates["offset"], result.parameter_estimates["slope"]], dtype=float)
    residual = record["observations"] - record["design"] @ estimates
    cholesky = np.linalg.cholesky(record["total_covariance"])
    return np.linalg.solve(cholesky, residual)


def _write_plots(output_root: Path, summaries: list[MappingLike]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    for ax, item in zip(axes.ravel(), summaries):
        record = item["record"]
        result = item["result"]
        estimates = np.asarray([result.parameter_estimates["offset"], result.parameter_estimates["slope"]], dtype=float)
        fitted = record["design"] @ estimates
        sigma = np.sqrt(np.maximum(np.diag(record["total_covariance"]), 0.0))
        ax.fill_between(record["x"], fitted - 2.0 * sigma, fitted + 2.0 * sigma, color="tab:blue", alpha=0.14, label="2 sigma")
        ax.plot(record["x"], record["expected"], color="black", linewidth=1.7, label="truth")
        ax.plot(record["x"], fitted, color="tab:orange", linestyle="--", label="recovered")
        ax.scatter(record["x"], record["observations"], s=12, color="tab:blue", alpha=0.55, label="observed")
        ax.set_title(f"{record['scenario_id']} ({result.gate_status})", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7, loc="best")
    path = figure_root / "synthetic_observable_overlay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["synthetic_observable_overlay"] = path.as_posix()

    labels = []
    means = []
    lows = []
    highs = []
    truths = []
    for item in summaries:
        result = item["result"]
        for name in item["record"]["inputs"].parameter_names:
            label = f"{item['record']['scenario_id']}:{name}"
            mean = result.parameter_estimates[name]
            sd = result.parameter_standard_deviation[name]
            labels.append(label)
            means.append(mean)
            lows.append(mean - 2.0 * sd)
            highs.append(mean + 2.0 * sd)
            truths.append(dict(result.summary["true_parameters"])[name])
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    x_values = np.arange(len(labels))
    lower = np.asarray(means) - np.asarray(lows)
    upper = np.asarray(highs) - np.asarray(means)
    ax.errorbar(x_values, means, yerr=[lower, upper], fmt="o", color="tab:blue", label="estimate +/- 2 sd")
    ax.scatter(x_values, truths, marker="x", color="black", label="truth")
    ax.set_xticks(x_values, labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("parameter value")
    ax.legend(fontsize=8)
    path = figure_root / "recovery_parameter_intervals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["recovery_parameter_intervals"] = path.as_posix()

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for item in summaries:
        whitened = _whitened_residual(item["record"], item["result"])
        ax.hist(whitened, bins=12, alpha=0.42, label=item["record"]["scenario_id"])
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("whitened residual")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    path = figure_root / "residual_whitened_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["residual_whitened_hist"] = path.as_posix()

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.6), constrained_layout=True)
    for ax, item in zip(axes.ravel(), summaries):
        matrix = item["record"]["total_covariance"]
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(item["record"]["scenario_id"], fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=ax, shrink=0.76)
    path = figure_root / "covariance_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["covariance_heatmap"] = path.as_posix()
    return paths


def _residual_risk_notes() -> str:
    return (
        "M6 synthetic recovery fixtures validate deterministic linear-Gaussian plumbing, finite observables, "
        "named hierarchy covariance terms, and threshold reporting. They do not claim production EMB posterior "
        "calibration; MES-35/MES-37 add PPC/SBC and EMB comparison evidence."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M6 synthetic recovery diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("_runs/noise/m6_synthetic_recovery_diagnostics"))
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    thresholds = SyntheticRecoveryThresholds()
    records = _fixture_scenarios()
    summaries = _write_scenario_artifacts(output_root, records, thresholds)
    summary_csv = output_root / "recovery_summary.csv"
    _write_summary_csv(summary_csv, summaries)
    plots = _write_plots(output_root, summaries)

    metrics = {
        "schema_version": 1,
        "thresholds": thresholds.as_dict(),
        "scenario_gate_statuses": {item["result"].scenario_id: item["result"].gate_status for item in summaries},
        "all_scenarios_passed": all(item["result"].gate_status == "pass" for item in summaries),
        "scenario_metrics": {item["result"].scenario_id: dict(item["result"].summary) for item in summaries},
        "residual_risk_notes": _residual_risk_notes(),
    }
    metrics_path = output_root / "synthetic_recovery_metrics.json"
    _write_json(metrics_path, metrics)
    manifest = {
        "schema_version": 1,
        "description": "M6 synthetic recovery diagnostics for MES-34.",
        "commands": {
            "regenerate": "python scripts/qa/noise_m6_synthetic_recovery_diagnostics.py --output-root <output-root>",
            "executed_argv": list(sys.argv),
        },
        "provenance": _provenance(),
        "thresholds": thresholds.as_dict(),
        "required_scenarios": [record["scenario_id"] for record in records],
        "scenario_artifacts": {item["result"].scenario_id: item["artifacts"] for item in summaries},
        "configs": {"primary": "configs/noise/synthetic_recovery.example.yaml"},
        "artifacts": {
            "metrics": _artifact_ref(output_root, metrics_path),
            "summary_csv": _artifact_ref(output_root, summary_csv),
            **{name: _artifact_ref(output_root, Path(plot_path)) for name, plot_path in plots.items()},
        },
        "residual_risk": _residual_risk_notes(),
        "residual_risk_notes": _residual_risk_notes(),
    }
    _write_json(output_root / "synthetic_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
