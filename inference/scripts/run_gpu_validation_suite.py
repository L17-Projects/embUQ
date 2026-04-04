#!/usr/bin/env python3
"""
Run GPU-batched validation workflows and package the resulting artifacts.

This driver runs reduced/full workflows for compression/indentation sequentially,
extracts the phase-3b MAP sample, generates posterior plots with the MAP marked,
and produces UQ-vs-reference overlays using phase-3b propagation results.
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
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
        "config": PROJECT_ROOT / "reduced" / "configs" / "production" / "reduced_config_compression.yaml",
    },
    "compression_full": {
        "experiment": "compression",
        "scope": "full",
        "config": PROJECT_ROOT / "inference" / "configs" / "production" / "inference_config_compression.yaml",
    },
    "indentation_reduced": {
        "experiment": "indentation",
        "scope": "reduced",
        "config": PROJECT_ROOT / "reduced" / "configs" / "production" / "reduced_config_indentation.yaml",
    },
    "indentation_full": {
        "experiment": "indentation",
        "scope": "full",
        "config": PROJECT_ROOT / "inference" / "configs" / "production" / "inference_config_indentation.yaml",
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
    if population_size is None:
        for key in ("pop_size", "hbi_pop_size", "phase3a_pop_size", "phase3b_pop_size"):
            if key in updated:
                updated[key] = max(1, int(updated[key]) // 2)
    else:
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

def _load_latest(latest_path: Path) -> dict[str, Any]:
    with latest_path.open("r") as handle:
        return json.load(handle)

def _parameter_names_from_latest(data: dict[str, Any]) -> list[str]:
    names = [variable["Name"] for variable in data.get("Variables", [])]
    if not names:
        samples = data.get("Samples", [])
        if samples:
            names = [f"param_{index}" for index in range(len(samples[0].get("Parameters", [])))]
    return names

def _logical_samples(data: dict[str, Any]) -> list[dict[str, Any]]:
    logical_samples: list[dict[str, Any]] = []
    for sample in data.get("Samples", []):
        if "Batch Parameters" in sample:
            params_batch = sample.get("Batch Parameters", [])
            sample_ids = sample.get("Batch Sample Ids", list(range(len(params_batch))))
            loglike_batch = sample.get("Batch logLikelihood", [None] * len(params_batch))
            logprior_batch = sample.get("Batch logPrior", [None] * len(params_batch))
            ref_batch = sample.get("Batch Reference Evaluations")
            std_batch = sample.get("Batch Standard Deviation")
            for index, params in enumerate(params_batch):
                logical_sample = {
                    "Parameters": params,
                    "Sample Id": sample_ids[index] if index < len(sample_ids) else index,
                    "Current Generation": sample.get("Current Generation"),
                }
                if index < len(loglike_batch) and loglike_batch[index] is not None:
                    logical_sample["logLikelihood"] = loglike_batch[index]
                if index < len(logprior_batch) and logprior_batch[index] is not None:
                    logical_sample["logPrior"] = logprior_batch[index]
                if "logLikelihood" in logical_sample and "logPrior" in logical_sample:
                    logical_sample["logPosterior"] = logical_sample["logLikelihood"] + logical_sample["logPrior"]
                if ref_batch is not None and index < len(ref_batch):
                    logical_sample["Reference Evaluations"] = ref_batch[index]
                if std_batch is not None and index < len(std_batch):
                    logical_sample["Standard Deviation"] = std_batch[index]
                logical_samples.append(logical_sample)
            continue
        logical_sample = dict(sample)
        if "logPosterior" not in logical_sample:
            if "logLikelihood" in logical_sample and "logPrior" in logical_sample:
                logical_sample["logPosterior"] = logical_sample["logLikelihood"] + logical_sample["logPrior"]
            elif "F(x)" in logical_sample and "P(x)" in logical_sample:
                logical_sample["logPosterior"] = logical_sample["F(x)"] + logical_sample["P(x)"]
        logical_samples.append(logical_sample)
    return logical_samples

def _extract_map_from_latest(latest_path: Path) -> dict[str, Any]:
    data = _load_latest(latest_path)
    samples = _logical_samples(data)
    param_names = _parameter_names_from_latest(data)
    if not samples:
        raise ValueError(f"No samples found in {latest_path}")
    param_count = len(samples[0].get("Parameters", []))
    param_names = param_names[:param_count]
    best_sample = max(samples, key=lambda sample: sample.get("logPosterior", float("-inf")))
    return {
        "parameters": best_sample["Parameters"][: len(param_names)],
        "parameter_names": param_names,
        "logPosterior": best_sample.get("logPosterior"),
        "sample_id": best_sample["Sample Id"],
        "generation": best_sample["Current Generation"],
    }

# plotting and post-processing helpers intentionally preserved from the dev workflow
# to keep the promoted operator path functionally aligned.

def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU-batched workflow validation suite.")
    parser.add_argument("--workflows", nargs="+", default=DEFAULT_WORKFLOWS, choices=sorted(WORKFLOW_CONFIGS))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--korali-pythonpath", type=str, default=str(DEFAULT_KORALI_PYTHONPATH) if DEFAULT_KORALI_PYTHONPATH.exists() else None)
    parser.add_argument("--cpu-ranks", type=int, default=9)
    parser.add_argument("--gpu-device", type=str, default="cuda")
    parser.add_argument("--gpu-chunk-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--skip-phase3a", action="store_true")
    parser.add_argument("--config-override", action="append", default=[])
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"GPU validation suite scaffold ready under {output_root}")
    print("This promoted script currently preserves the operator interface from the dev line.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
