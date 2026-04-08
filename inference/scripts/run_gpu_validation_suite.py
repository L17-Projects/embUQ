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
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / "korali_gpu"
DEFAULT_KORALI_PYTHONPATH = (
    PROJECT_ROOT.parent / "korali_gpu" / "local_install" / "usr" / "local" / "lib" / "python3.8" / "site-packages"
)
DEFAULT_WORKFLOWS = [
    "compression_reduced",
    "compression_full",
    "indentation_reduced",
    "indentation_full",
]
WORKFLOW_CONFIGS: Dict[str, Dict[str, Any]] = {
    "compression_reduced": {
        "experiment": "compression",
        "scope": "reduced",
        "model_family": "reduced-model",
        "profile": "validation",
        "config": PROJECT_ROOT / "reduced" / "configs" / "validation" / "validation_config_compression.yaml",
    },
    "compression_full": {
        "experiment": "compression",
        "scope": "full",
        "model_family": "full-model",
        "profile": "validation",
        "config": PROJECT_ROOT / "inference" / "configs" / "validation" / "validation_config_compression.yaml",
    },
    "indentation_reduced": {
        "experiment": "indentation",
        "scope": "reduced",
        "model_family": "reduced-model",
        "profile": "validation",
        "config": PROJECT_ROOT / "reduced" / "configs" / "validation" / "validation_config_indentation.yaml",
    },
    "indentation_full": {
        "experiment": "indentation",
        "scope": "full",
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
        for key in ("pop_size", "hbi_pop_size", "phase3a_pop_size", "phase3b_pop_size"):
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


def _load_experiment_spec(config_path: Path, experiment_name: str):
    from meso_uq.experiments import load_experiments

    with config_path.open("rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)
    experiments = [exp for exp in load_experiments(config, PROJECT_ROOT) if exp.enabled]
    for experiment in experiments:
        if experiment.name == experiment_name:
            return experiment
    raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")


def _postprocess_workflow(workflow_name: str, experiment_name: str, config_path: Path, workflow_dir: Path) -> None:
    from meso_uq.postprocess.maps import extract_map_from_directory, load_posterior_samples
    from meso_uq.postprocess.plots import (
        plot_d0_correlations,
        plot_posterior_marginals,
        plot_validation_overlay,
    )

    results_dir = workflow_dir / "results"
    map_dir = workflow_dir / "map_phase3b"
    posterior_dir = workflow_dir / "posteriors_phase3b"
    overlay_dir = workflow_dir / "overlay_uq_ref"
    samples_dir = workflow_dir / "samples_phase3b"
    map_dir.mkdir(parents=True, exist_ok=True)
    posterior_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    experiment = _load_experiment_spec(config_path, experiment_name)
    combined_map: Dict[str, Any] = {}

    for diameter_um in experiment.diameters:
        exp_name = experiment.dataset_name(diameter_um)
        phase3b_dir = results_dir / "results_phase_3b" / exp_name
        map_csv = map_dir / f"{diameter_um}um_map.csv"
        map_df = extract_map_from_directory(phase3b_dir, output_csv=str(map_csv))
        combined_map[f"{diameter_um}um"] = map_df.iloc[0].to_dict()

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

    print(f"Postprocessed workflow '{workflow_name}' into {workflow_dir}")


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
    workflow_output_name = workflow_name if population_size is None else f"{workflow_name}_{population_size}"
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

    _postprocess_workflow(workflow_name, workflow_spec["experiment"], config_path, workflow_dir)

    summary = {
        "workflow": workflow_output_name,
        "workflow_base_name": workflow_name,
        "scope": workflow_spec["scope"],
        "experiment": workflow_spec["experiment"],
        "model_family": workflow_spec["model_family"],
        "profile": workflow_spec["profile"],
        "selection": f"{workflow_spec['experiment']}:{workflow_spec['model_family']}:{workflow_spec['profile']}",
        "config": str(config_path),
        "population_settings": {
            key: derived_config.get(key)
            for key in ("pop_size", "hbi_pop_size", "phase3a_pop_size", "phase3b_pop_size")
            if key in derived_config
        },
        "population_override": population_size,
        "step_timings": timings,
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in timings),
    }
    with (workflow_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU-batched workflow validation suite.")
    parser.add_argument("--workflows", nargs="+", default=DEFAULT_WORKFLOWS, choices=sorted(WORKFLOW_CONFIGS))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
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

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config_overrides: dict[str, Path] = {}
    for item in args.config_override:
        if "=" not in item:
            raise ValueError(f"Invalid --config-override value '{item}'. Expected workflow=path.")
        workflow_name, path_str = item.split("=", 1)
        if workflow_name not in WORKFLOW_CONFIGS:
            raise ValueError(f"Unknown workflow '{workflow_name}' in --config-override.")
        config_overrides[workflow_name] = Path(path_str).resolve()

    suite_summary = []
    for workflow_name in args.workflows:
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
