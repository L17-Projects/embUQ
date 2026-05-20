#!/usr/bin/env python3
"""
Run GPU-batched validation workflows and package the resulting artifacts.

This restored operator runner drives the public MesoUQ scripts end to end for
selected full/reduced workflows, extracts MAP samples from Phase 3b, generates
posterior plots, and produces UQ-vs-reference overlays from propagation outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from meso_uq.hpc_paths import default_runs_root  # noqa: E402
from meso_uq.platforms.site_selector import resolve_hpc_site  # noqa: E402
from meso_uq.vega_workflows import VegaWorkflowSelection, parse_selection, selection_key, selection_slug

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = None
DEFAULT_KORALI_PYTHONPATH = (
    PROJECT_ROOT.parent / "korali_gpu" / "local_install" / "usr" / "local" / "lib" / "python3.8" / "site-packages"
)
DEFAULT_WORKFLOWS = [
    "compression:reduced-model:validation",
    "compression:full-model:validation",
    "indentation:reduced-model:validation",
    "indentation:full-model:validation",
]
WORKFLOW_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compression:reduced-model:validation": {
        "experiment": "compression",
        "model_family": "reduced-model",
        "profile": "validation",
        "config": PROJECT_ROOT / "reduced" / "configs" / "validation" / "validation_config_compression.yaml",
    },
    "compression:full-model:validation": {
        "experiment": "compression",
        "model_family": "full-model",
        "profile": "validation",
        "config": PROJECT_ROOT / "inference" / "configs" / "validation" / "validation_config_compression.yaml",
    },
    "indentation:reduced-model:validation": {
        "experiment": "indentation",
        "model_family": "reduced-model",
        "profile": "validation",
        "config": PROJECT_ROOT / "reduced" / "configs" / "validation" / "validation_config_indentation.yaml",
    },
    "indentation:full-model:validation": {
        "experiment": "indentation",
        "model_family": "full-model",
        "profile": "validation",
        "config": PROJECT_ROOT / "inference" / "configs" / "validation" / "validation_config_indentation.yaml",
    },
}


def _run_step(step_name: str, command: list[str], cwd: Path, env: dict[str, str], timings: list[dict[str, object]]) -> None:
    print(f"\n=== {step_name} ===")
    print(f"cwd: {cwd}")
    print("cmd:", " ".join(shlex.quote(part) for part in command))
    start = time.perf_counter()
    subprocess.run(command, cwd=str(cwd), env=env, check=True)
    elapsed = time.perf_counter() - start
    timings.append({"step": step_name, "elapsed_seconds": elapsed})
    print(f"{step_name} finished in {elapsed:.2f}s")


def _set_population_settings(config: dict[str, Any], population_size: int | None) -> dict[str, Any]:
    updated = dict(config)
    if population_size is not None:
        for key in ("pop_size", "hbi_pop_size", "phase3b_pop_size"):
            if key in updated:
                updated[key] = int(population_size)
    return updated


def _write_derived_config(base_config: Path, output_file: Path, population_size: int | None) -> dict[str, Any]:
    with base_config.open("rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)
    config = _set_population_settings(config, population_size)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config


def _build_env(korali_pythonpath: str | None) -> dict[str, str]:
    env = os.environ.copy()
    extra_pythonpath = [str(PROJECT_ROOT)]
    if korali_pythonpath:
        extra_pythonpath.insert(0, korali_pythonpath)
    env["PYTHONPATH"] = ":".join(extra_pythonpath + [env.get("PYTHONPATH", "")]).rstrip(":")
    return env


def _resolve_validation_selection(value: str) -> str:
    selection = parse_selection(value)
    if selection.profile != "validation":
        raise ValueError(
            "The validation suite only supports validation-profile selections. "
            f"Got: {value}"
        )
    key = selection_key(selection)
    if key not in WORKFLOW_CONFIGS:
        raise ValueError(f"Unsupported validation workflow selection: {value}")
    return key


def _load_experiment_spec(config_path: Path, experiment_name: str):
    from meso_uq.experiments import load_experiments

    with config_path.open("rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)
    experiments = [exp for exp in load_experiments(config, PROJECT_ROOT) if exp.enabled]
    for experiment in experiments:
        if experiment.name == experiment_name:
            return experiment
    raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")


def _strict_jsonable(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_strict_jsonable(item) for item in value]
    return value


def _postprocess_workflow(
    workflow_name: str,
    experiment_name: str,
    config_path: Path,
    workflow_dir: Path,
) -> dict[str, str]:
    from meso_uq.postprocess.diagnostics import (
        PHASE1_POSTERIOR_FIGURE_POLICY,
        duplicate_mass_comparison,
        duplicate_particle_metrics,
        mean_or_nan,
        posterior_parameter_columns,
    )
    from meso_uq.postprocess.maps import (
        extract_map_from_directory,
        load_chain_leader_samples,
        load_posterior_samples,
    )
    from meso_uq.postprocess.plots import (
        plot_d0_correlations,
        plot_posterior_marginals,
        plot_validation_overlay,
    )

    results_dir = workflow_dir / "results"
    map_dir = workflow_dir / "map_phase3b"
    phase1_posterior_dir = workflow_dir / "posteriors_phase1"
    posterior_dir = workflow_dir / "posteriors_phase3b"
    diagnostics_dir = workflow_dir / "diagnostics_phase1"
    overlay_dir = workflow_dir / "overlay_uq_ref"
    phase1_samples_dir = workflow_dir / "samples_phase1"
    samples_dir = workflow_dir / "samples_phase3b"
    map_dir.mkdir(parents=True, exist_ok=True)
    phase1_posterior_dir.mkdir(parents=True, exist_ok=True)
    posterior_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    phase1_samples_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    experiment = _load_experiment_spec(config_path, experiment_name)
    combined_map: Dict[str, Any] = {}
    phase1_diagnostics: dict[str, Any] = {
        "workflow": workflow_name,
        "experiment": experiment_name,
        "posterior_figure_policy": dict(PHASE1_POSTERIOR_FIGURE_POLICY),
        "datasets": {},
    }

    for diameter_um in experiment.diameters:
        exp_name = experiment.dataset_name(diameter_um)
        phase1_dir = results_dir / "results_phase_1" / exp_name
        phase3b_dir = results_dir / "results_phase_3b" / exp_name
        map_csv = map_dir / f"{diameter_um}um_map.csv"
        map_df = extract_map_from_directory(phase3b_dir, output_csv=str(map_csv))
        combined_map[f"{diameter_um}um"] = map_df.iloc[0].to_dict()

        phase1_samples_df = load_posterior_samples(phase1_dir)
        phase1_samples_csv = phase1_samples_dir / f"{diameter_um}um_samples.csv"
        phase1_samples_df.to_csv(phase1_samples_csv, index=False)
        plot_posterior_marginals(
            str(phase1_samples_csv),
            str(phase1_posterior_dir / f"posterior_marginals_{diameter_um}um.png"),
        )
        if "d0" in phase1_samples_df.columns:
            plot_d0_correlations(
                str(phase1_samples_csv),
                str(phase1_posterior_dir / f"d0_correlations_{diameter_um}um.png"),
            )

        posterior_metrics = duplicate_particle_metrics(
            phase1_samples_df,
            parameter_columns=posterior_parameter_columns(phase1_samples_df),
        )
        chain_leader_metrics: dict[str, float | int] | None
        chain_leader_error: str | None = None
        try:
            chain_leader_df = load_chain_leader_samples(phase1_dir)
            chain_leader_metrics = duplicate_particle_metrics(
                chain_leader_df,
                parameter_columns=posterior_parameter_columns(chain_leader_df),
            )
        except ValueError as exc:
            chain_leader_metrics = None
            chain_leader_error = str(exc)

        phase1_diagnostics["datasets"][exp_name] = {
            "diameter_um": diameter_um,
            "posterior_sample_duplicates": posterior_metrics,
            "chain_leader_duplicates": chain_leader_metrics,
            "comparison_vs_posterior_samples": duplicate_mass_comparison(
                posterior_metrics, chain_leader_metrics
            ),
            "chain_leader_error": chain_leader_error,
        }

        samples_df = load_posterior_samples(phase3b_dir)
        samples_csv = samples_dir / f"{diameter_um}um_samples.csv"
        samples_df.to_csv(samples_csv, index=False)

        plot_posterior_marginals(str(samples_csv), str(posterior_dir / f"posterior_marginals_{diameter_um}um.png"))
        if "d0" in samples_df.columns:
            plot_d0_correlations(str(samples_csv), str(posterior_dir / f"d0_correlations_{diameter_um}um.png"))

        ref_overlay_csv = overlay_dir / f"reference_{diameter_um}um.csv"
        pd.DataFrame({
            "x": experiment.get_reference_points(diameter_um),
            "y": experiment.get_reference_data(diameter_um),
        }).to_csv(ref_overlay_csv, index=False)

        propagation_csv = results_dir / "propagation_phase3b" / exp_name / "summary.csv"
        overlay_png = overlay_dir / f"uq_overlay_{diameter_um}um.png"
        plot_validation_overlay(
            str(ref_overlay_csv),
            str(propagation_csv),
            str(overlay_png),
            x_col="x",
            y_ref_col="y",
            y_pred_col="mean",
            label_ref="reference",
            label_pred="posterior mean",
        )

    with (map_dir / "all_diameters_map.json").open("w") as handle:
        json.dump(combined_map, handle, indent=2)

    per_dataset = list(phase1_diagnostics["datasets"].values())
    chain_available = [
        item for item in per_dataset if item.get("chain_leader_duplicates") is not None
    ]
    phase1_diagnostics["lane_summary"] = {
        "dataset_count": len(per_dataset),
        "chain_leader_dataset_count": len(chain_available),
        "posterior_top_duplicate_mass_mean": mean_or_nan(
            float(item["posterior_sample_duplicates"]["top_duplicate_mass"]) for item in per_dataset
        ),
        "posterior_top_10_duplicate_mass_mean": mean_or_nan(
            float(item["posterior_sample_duplicates"]["top_10_duplicate_mass"]) for item in per_dataset
        ),
        "chain_leader_top_duplicate_mass_mean": mean_or_nan(
            float(item["chain_leader_duplicates"]["top_duplicate_mass"]) for item in chain_available
        ),
        "chain_leader_top_10_duplicate_mass_mean": mean_or_nan(
            float(item["chain_leader_duplicates"]["top_10_duplicate_mass"]) for item in chain_available
        ),
        "top_duplicate_mass_delta_mean": mean_or_nan(
            float(item["comparison_vs_posterior_samples"]["top_duplicate_mass_delta"])
            for item in chain_available
        ),
        "top_10_duplicate_mass_delta_mean": mean_or_nan(
            float(item["comparison_vs_posterior_samples"]["top_10_duplicate_mass_delta"])
            for item in chain_available
        ),
    }

    diagnostics_json_path = diagnostics_dir / "phase1_duplicate_particle_metrics.json"
    diagnostics_json_path.write_text(
        json.dumps(_strict_jsonable(phase1_diagnostics), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    def _fmt_ratio(value: float | None) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float) and math.isnan(value):
            return "n/a"
        return f"{100.0 * float(value):.2f}%"

    report_lines = [
        "# Phase 1 Duplicate-Particle Diagnostics",
        "",
        f"Workflow: `{workflow_name}`",
        "",
        "Posterior-figure policy: use raw Phase 1 posterior samples for plotting; do not filter duplicates.",
        "",
        "| Dataset | Unique particles (posterior) | Top duplicate mass (posterior) | Top-10 duplicate mass (posterior) | Top duplicate mass (chain leaders) | Top-10 duplicate mass (chain leaders) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name, payload in phase1_diagnostics["datasets"].items():
        posterior = payload["posterior_sample_duplicates"]
        chain = payload.get("chain_leader_duplicates") or {}
        report_lines.append(
            "| "
            f"{dataset_name} | "
            f"{int(posterior['unique_particle_count'])} | "
            f"{_fmt_ratio(float(posterior['top_duplicate_mass']))} | "
            f"{_fmt_ratio(float(posterior['top_10_duplicate_mass']))} | "
            f"{_fmt_ratio(chain.get('top_duplicate_mass'))} | "
            f"{_fmt_ratio(chain.get('top_10_duplicate_mass'))} |"
        )
        if payload.get("chain_leader_error"):
            report_lines.append("")
            report_lines.append(
                f"> Chain leader diagnostics unavailable for `{dataset_name}`: {payload['chain_leader_error']}"
            )

    lane_summary = phase1_diagnostics["lane_summary"]
    report_lines.extend(
        [
            "",
            "## Lane Summary",
            "",
            f"- Datasets audited: {lane_summary['dataset_count']}",
            f"- Datasets with chain leaders available: {lane_summary['chain_leader_dataset_count']}",
            f"- Mean top duplicate mass delta (chain leaders - posterior): {_fmt_ratio(lane_summary['top_duplicate_mass_delta_mean'])}",
            f"- Mean top-10 duplicate mass delta (chain leaders - posterior): {_fmt_ratio(lane_summary['top_10_duplicate_mass_delta_mean'])}",
        ]
    )
    diagnostics_md_path = diagnostics_dir / "phase1_duplicate_particle_report.md"
    diagnostics_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Postprocessed workflow '{workflow_name}' into {workflow_dir}")
    return {
        "phase1_posterior_dir": str(phase1_posterior_dir),
        "phase3b_posterior_dir": str(posterior_dir),
        "phase1_samples_dir": str(phase1_samples_dir),
        "phase3b_samples_dir": str(samples_dir),
        "phase1_duplicate_metrics_json": str(diagnostics_json_path),
        "phase1_duplicate_report_md": str(diagnostics_md_path),
        "phase1_posterior_policy": PHASE1_POSTERIOR_FIGURE_POLICY["policy_id"],
    }


def _phase2_command(python_bin: str, cpu_ranks: int, script: Path, config_path: Path, results_dir: Path) -> list[str]:
    if cpu_ranks > 1:
        return [
            "mpirun", "--oversubscribe", "-np", str(cpu_ranks),
            python_bin, str(script), "--config", str(config_path), "--output-dir", str(results_dir),
        ]
    return [python_bin, str(script), "--config", str(config_path), "--output-dir", str(results_dir)]


def run_workflow(
    workflow_name: str,
    workflow_spec: dict[str, Any],
    output_root: Path,
    python_bin: str,
    korali_pythonpath: str | None,
    cpu_ranks: int,
    population_size: int | None,
) -> dict[str, Any]:
    selection = VegaWorkflowSelection(
        workflow_spec["experiment"],
        workflow_spec["model_family"],
        workflow_spec["profile"],
    )
    workflow_slug = selection_slug(selection)
    workflow_output_name = workflow_slug if population_size is None else f"{workflow_slug}_{population_size}"
    workflow_dir = output_root / workflow_output_name
    workflow_dir.mkdir(parents=True, exist_ok=True)
    results_dir = workflow_dir / "results"
    config_name = "config.yaml" if population_size is None else f"config_pop{population_size}.yaml"
    config_path = workflow_dir / config_name
    derived_config = _write_derived_config(Path(workflow_spec["config"]), config_path, population_size)

    env = _build_env(korali_pythonpath)
    timings: list[dict[str, object]] = []

    phase1_script = PROJECT_ROOT / "inference" / "scripts" / "run_phase_1.py"
    phase2_script = PROJECT_ROOT / "inference" / "scripts" / "run_phase_2.py"
    phase3b_script = PROJECT_ROOT / "inference" / "scripts" / "run_phase_3b.py"
    prop1_script = PROJECT_ROOT / "propagation" / "scripts" / "run_phase1_propagation.py"
    prop3_script = PROJECT_ROOT / "propagation" / "scripts" / "run_phase3b_propagation.py"

    _run_step(
        "Phase 1",
        [python_bin, str(phase1_script), "--config", str(config_path), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        env,
        timings,
    )
    _run_step(
        "Phase 2",
        _phase2_command(python_bin, cpu_ranks, phase2_script, config_path, results_dir),
        PROJECT_ROOT,
        env,
        timings,
    )
    _run_step(
        "Phase 3b",
        [python_bin, str(phase3b_script), "--config", str(config_path), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        env,
        timings,
    )
    _run_step(
        "Propagation Phase 1",
        [python_bin, str(prop1_script), "--config", str(config_path), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        env,
        timings,
    )
    _run_step(
        "Propagation Phase 3b",
        [python_bin, str(prop3_script), "--config", str(config_path), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        env,
        timings,
    )

    postprocess_artifacts = _postprocess_workflow(
        workflow_name, workflow_spec["experiment"], config_path, workflow_dir
    )

    summary = {
        "workflow": workflow_output_name,
        "workflow_base_name": selection_key(selection),
        "experiment": workflow_spec["experiment"],
        "model_family": workflow_spec["model_family"],
        "profile": workflow_spec["profile"],
        "selection": selection_key(selection),
        "config": str(config_path),
        "population_settings": {
            key: derived_config.get(key)
            for key in ("pop_size", "hbi_pop_size", "phase3b_pop_size")
            if key in derived_config
        },
        "population_override": population_size,
        "step_timings": timings,
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in timings),
        "postprocess_artifacts": postprocess_artifacts,
    }
    with (workflow_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU-batched workflow validation suite.")
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=DEFAULT_WORKFLOWS,
        help="Validation workflow selections in experiment:model-family:profile form.",
    )
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--korali-pythonpath",
        type=str,
        default=str(DEFAULT_KORALI_PYTHONPATH) if DEFAULT_KORALI_PYTHONPATH.exists() else None,
        help="Optional site-packages path for the locally built Korali install.",
    )
    parser.add_argument("--cpu-ranks", type=int, default=1, help="MPI rank count for Phase 2.")
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Optional explicit population size for pop_size/hbi_pop_size/phase3b_pop_size. If omitted, uses half the base config values.",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Optional workflow-specific base config override in the form workflow_name=/abs/path/config.yaml.",
    )
    args = parser.parse_args()

    resolved_site = resolve_hpc_site(cli_site=args.site, env=os.environ, allow_hostname=True, default="vega")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else default_runs_root(PROJECT_ROOT, "validation_suite", site=resolved_site, run_tag=args.run_tag)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    config_overrides: dict[str, Path] = {}
    for item in args.config_override:
        if "=" not in item:
            raise ValueError(f"Invalid --config-override value '{item}'. Expected workflow=path.")
        workflow_name, path_str = item.split("=", 1)
        resolved_name = _resolve_validation_selection(workflow_name)
        config_overrides[resolved_name] = Path(path_str).resolve()

    suite_summary = []
    for requested_name in args.workflows:
        workflow_name = _resolve_validation_selection(requested_name)
        workflow_spec = dict(WORKFLOW_CONFIGS[workflow_name])
        if workflow_name in config_overrides:
            workflow_spec["config"] = config_overrides[workflow_name]
        suite_summary.append(
            run_workflow(
                workflow_name=workflow_name,
                workflow_spec=workflow_spec,
                output_root=output_root,
                python_bin=args.python_bin,
                korali_pythonpath=args.korali_pythonpath,
                cpu_ranks=args.cpu_ranks,
                population_size=args.population_size,
            )
        )

    with (output_root / "workflow_suite_summary.json").open("w") as handle:
        json.dump(suite_summary, handle, indent=2)
    print(f"Validation suite complete under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
