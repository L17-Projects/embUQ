#!/usr/bin/env python3
"""Evaluate the MES-208 replicate-based Phase 2 posterior gate."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.inference.posterior_equivalence import load_phase2_posterior_samples  # noqa: E402

LOG_COLUMNS = frozenset({"logLikelihood", "logPrior", "logPosterior"})
NON_PARAMETER_COLUMNS = frozenset({"Sample Id", "Sample ID", "Generation", "Chain", *LOG_COLUMNS})
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class LoadedRun:
    run_id: str
    backend: str
    replicate_id: str
    output_dir: Path
    samples: Any
    variables: tuple[str, ...]
    sample_count: int
    finite_logposterior_ratio: float
    nonfinite_parameter_columns: tuple[str, ...]
    mean: Mapping[str, float]
    std: Mapping[str, float]
    quantiles: Mapping[str, Mapping[str, float]]


def numeric_column(samples: Any, name: str) -> np.ndarray:
    values = samples[name]
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return np.asarray(values, dtype=float)


def infer_parameter_columns(samples: Any) -> tuple[str, ...]:
    columns = tuple(str(name) for name in samples.columns)
    return tuple(name for name in columns if name not in NON_PARAMETER_COLUMNS)


def finite_logposterior_ratio(samples: Any) -> float:
    if "logPosterior" not in samples:
        return 0.0
    values = numeric_column(samples, "logPosterior")
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).sum() / values.size)


def summarize_run(run: Mapping[str, Any], quantiles: Sequence[float]) -> LoadedRun:
    output_dir = Path(str(run["output_dir"])).expanduser().resolve()
    phase2_dir = output_dir / "results_phase_2"
    samples = load_phase2_posterior_samples(phase2_dir)
    variables = infer_parameter_columns(samples)
    if not variables:
        raise ValueError(f"{run['id']} has no inferred parameter columns")

    arrays = {name: numeric_column(samples, name) for name in variables}
    sample_counts = {values.size for values in arrays.values()}
    if len(sample_counts) != 1:
        raise ValueError(f"{run['id']} has inconsistent parameter column lengths")
    sample_count = sample_counts.pop()
    nonfinite = tuple(name for name, values in arrays.items() if not np.all(np.isfinite(values)))
    quantile_values = tuple(float(value) for value in quantiles)

    return LoadedRun(
        run_id=str(run["id"]),
        backend=str(run["backend"]),
        replicate_id=str(run.get("replicate_id", run["id"])),
        output_dir=output_dir,
        samples=samples,
        variables=variables,
        sample_count=int(sample_count),
        finite_logposterior_ratio=finite_logposterior_ratio(samples),
        nonfinite_parameter_columns=nonfinite,
        mean={name: float(np.mean(values)) for name, values in arrays.items()},
        std={name: float(np.std(values, ddof=1)) if sample_count > 1 else 0.0 for name, values in arrays.items()},
        quantiles={
            name: {quantile_key(value): float(np.quantile(values, value)) for value in quantile_values}
            for name, values in arrays.items()
        },
    )


def quantile_key(value: float) -> str:
    return f"q{value:.6g}"


def pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    variance = (float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2.0
    return math.sqrt(max(variance, 0.0))


def normalize(value: float, scale: float) -> float:
    if scale <= 0.0 or not math.isfinite(scale):
        return 0.0 if abs(value) <= 0.0 else math.inf
    return float(abs(value) / scale)


def ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if left.size == 0 or right.size == 0:
        return math.inf
    values = np.concatenate([left, right])
    cdf_left = np.searchsorted(left, values, side="right") / left.size
    cdf_right = np.searchsorted(right, values, side="right") / right.size
    return float(np.max(np.abs(cdf_left - cdf_right)))


def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if left.size == 0 or right.size == 0:
        return math.inf
    if left.size == right.size:
        return float(np.mean(np.abs(left - right)))
    grid = np.linspace(0.0, 1.0, max(left.size, right.size), endpoint=True)
    return float(np.mean(np.abs(np.quantile(left, grid) - np.quantile(right, grid))))


def pair_diagnostics(
    left: LoadedRun,
    right: LoadedRun,
    *,
    pair_class: str,
    quantiles: Sequence[float],
) -> list[dict[str, Any]]:
    if left.variables != right.variables:
        raise ValueError(f"{left.run_id} and {right.run_id} expose different variables")

    rows: list[dict[str, Any]] = []
    for variable in left.variables:
        left_values = numeric_column(left.samples, variable)
        right_values = numeric_column(right.samples, variable)
        scale = pooled_std(left_values, right_values)
        row = {
            "pair_class": pair_class,
            "left_run": left.run_id,
            "right_run": right.run_id,
            "variable": variable,
            "ks_statistic": ks_statistic(left_values, right_values),
            "wasserstein_norm": normalize(wasserstein_1d(left_values, right_values), scale),
            "mean_shift_norm": normalize(float(np.mean(left_values) - np.mean(right_values)), scale),
            "std_shift_norm": normalize(float(np.std(left_values, ddof=1) - np.std(right_values, ddof=1)), scale),
        }
        for value in quantiles:
            key = quantile_key(float(value))
            delta = float(np.quantile(left_values, value) - np.quantile(right_values, value))
            row[f"quantile_shift_norm_{key}"] = normalize(delta, scale)
        rows.append(row)
    return rows


def percentile(values: Sequence[float], q: float) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=float), q))


def metric_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key.endswith("_norm") or key.startswith("quantile_shift_norm_") or key == "ks_statistic":
                if isinstance(value, (int, float)):
                    names.add(key)
    return tuple(sorted(names))


def aggregate_envelope(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_class: str,
    metrics: Sequence[str],
) -> dict[str, dict[str, float | None]]:
    selected = [row for row in rows if row["pair_class"] == pair_class]
    return {
        metric: {
            "median": percentile([row[metric] for row in selected], 50.0),
            "p95": percentile([row[metric] for row in selected], 95.0),
        }
        for metric in metrics
    }


def compare_envelopes(
    cpu_envelope: Mapping[str, Mapping[str, float | None]],
    cuda_envelope: Mapping[str, Mapping[str, float | None]],
    tolerances: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    for metric, cpu_values in cpu_envelope.items():
        cuda_values = cuda_envelope.get(metric, {})
        tolerance = float(tolerances.get(metric, 0.0))
        for field in ("median", "p95"):
            cpu_value = cpu_values.get(field)
            cuda_value = cuda_values.get(field)
            if cpu_value is None or cuda_value is None:
                failures.append(f"{metric}.{field} missing envelope value")
                continue
            if cuda_value > cpu_value + tolerance:
                failures.append(
                    f"{metric}.{field}: cuda_vs_cpu={cuda_value:.6g} exceeds "
                    f"cpu_vs_cpu={cpu_value:.6g} + tolerance={tolerance:.6g}"
                )
    return failures


def persistent_directional_shift_flags(
    cpu_runs: Sequence[LoadedRun],
    cuda_runs: Sequence[LoadedRun],
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if len(cpu_runs) < 2 or len(cuda_runs) < 2:
        return flags
    variables = cpu_runs[0].variables
    for variable in variables:
        cpu_means = np.asarray([run.mean[variable] for run in cpu_runs], dtype=float)
        cuda_means = np.asarray([run.mean[variable] for run in cuda_runs], dtype=float)
        cpu_center = float(np.median(cpu_means))
        signs = np.sign(cuda_means - cpu_center)
        nonzero_signs = signs[signs != 0]
        if nonzero_signs.size != signs.size or len(set(nonzero_signs.tolist())) != 1:
            continue
        cpu_pair_values = [
            float(row["mean_shift_norm"])
            for row in rows
            if row["pair_class"] == "cpu_vs_cpu" and row["variable"] == variable
        ]
        cpu_p95 = percentile(cpu_pair_values, 95.0) or 0.0
        scale = float(np.median([run.std[variable] for run in cpu_runs if run.std[variable] > 0.0]))
        cuda_shift_norm = normalize(float(np.median(cuda_means) - cpu_center), scale)
        if cuda_shift_norm > cpu_p95 + tolerance:
            flags.append(
                {
                    "variable": variable,
                    "direction": "positive" if nonzero_signs[0] > 0 else "negative",
                    "cuda_median_shift_norm": cuda_shift_norm,
                    "cpu_vs_cpu_p95_mean_shift_norm": cpu_p95,
                    "tolerance": tolerance,
                }
            )
    return flags


def existing_runs(runs: Iterable[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    ready: list[Mapping[str, Any]] = []
    missing: list[Mapping[str, Any]] = []
    for run in runs:
        latest = Path(str(run["output_dir"])).expanduser().resolve() / "results_phase_2" / "latest"
        if latest.exists():
            ready.append(run)
        else:
            missing.append(run)
    return ready, missing


def evaluate_plan(manifest: Mapping[str, Any], plan_name: str) -> dict[str, Any]:
    plan = manifest["plans"][plan_name]
    acceptance = manifest["acceptance_criteria"]
    quantiles = tuple(float(value) for value in acceptance.get("quantiles", DEFAULT_QUANTILES))
    ready_specs, missing_specs = existing_runs(plan["runs"])
    loaded = [summarize_run(run, quantiles) for run in ready_specs]

    cpu_runs = [run for run in loaded if run.backend == "cpu-mpi"]
    cuda_runs = [run for run in loaded if run.backend == "native-cuda"]
    quality_failures: list[str] = []
    for run in loaded:
        if run.sample_count != int(acceptance["sample_count"]):
            quality_failures.append(f"{run.run_id} sample_count={run.sample_count}")
        if run.finite_logposterior_ratio != float(acceptance["finite_logposterior_ratio"]):
            quality_failures.append(
                f"{run.run_id} finite_logposterior_ratio={run.finite_logposterior_ratio}"
            )
        if run.nonfinite_parameter_columns:
            quality_failures.append(
                f"{run.run_id} nonfinite_parameter_columns={list(run.nonfinite_parameter_columns)}"
            )

    pair_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(cpu_runs, 2):
        pair_rows.extend(pair_diagnostics(left, right, pair_class="cpu_vs_cpu", quantiles=quantiles))
    for cuda_run in cuda_runs:
        for cpu_run in cpu_runs:
            pair_rows.extend(
                pair_diagnostics(cuda_run, cpu_run, pair_class="cuda_vs_cpu", quantiles=quantiles)
            )
    for left, right in itertools.combinations(cuda_runs, 2):
        pair_rows.extend(pair_diagnostics(left, right, pair_class="cuda_vs_cuda", quantiles=quantiles))

    metrics = metric_names(pair_rows)
    cpu_envelope = aggregate_envelope(pair_rows, pair_class="cpu_vs_cpu", metrics=metrics)
    cuda_envelope = aggregate_envelope(pair_rows, pair_class="cuda_vs_cpu", metrics=metrics)
    tolerances = acceptance.get("envelope_tolerances", {})
    envelope_failures = compare_envelopes(cpu_envelope, cuda_envelope, tolerances)
    directional_flags = persistent_directional_shift_flags(
        cpu_runs,
        cuda_runs,
        pair_rows,
        tolerance=float(tolerances.get("mean_shift_norm", 0.0)),
    )

    count_failures = []
    if len(cpu_runs) < int(acceptance["min_cpu_replicates"]):
        count_failures.append(f"cpu-mpi replicates ready={len(cpu_runs)}")
    if len(cuda_runs) < int(acceptance["min_cuda_replicates"]):
        count_failures.append(f"native-cuda replicates ready={len(cuda_runs)}")

    passed = not (missing_specs or quality_failures or count_failures or envelope_failures or directional_flags)
    return {
        "schema_version": "mes208.phase2_gate_report.v1",
        "linear_issue": manifest.get("linear_issue", "MES-208"),
        "plan": plan_name,
        "passed": passed,
        "status": "passed" if passed else "incomplete_or_failed",
        "ready_run_ids": [run.run_id for run in loaded],
        "missing_run_ids": [str(run["id"]) for run in missing_specs],
        "replicate_counts": {"cpu-mpi": len(cpu_runs), "native-cuda": len(cuda_runs)},
        "quality_failures": quality_failures,
        "count_failures": count_failures,
        "envelope_failures": envelope_failures,
        "persistent_directional_shift_flags": directional_flags,
        "cpu_vs_cpu_envelope": cpu_envelope,
        "cuda_vs_cpu_envelope": cuda_envelope,
        "pair_diagnostics": pair_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plan", default="minimal_reuse")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = evaluate_plan(manifest, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "passed": report["passed"]}))
    if report["passed"] or args.allow_incomplete:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
