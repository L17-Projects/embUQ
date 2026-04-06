#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "validation_runs"
WORKFLOW_CONFIGS = {
    "compression_full": PROJECT_ROOT / "inference" / "configs" / "production" / "inference_config_compression.yaml",
    "indentation_full": PROJECT_ROOT / "inference" / "configs" / "production" / "inference_config_indentation.yaml",
    "compression_reduced": PROJECT_ROOT / "reduced" / "configs" / "production" / "reduced_config_compression.yaml",
    "indentation_reduced": PROJECT_ROOT / "reduced" / "configs" / "production" / "reduced_config_indentation.yaml",
}


def _run_step(name: str, cmd: list[str], cwd: Path, timings: list[dict[str, object]]) -> None:
    print(f"\n=== {name} ===")
    print(f"cwd: {cwd}")
    print("cmd:", " ".join(shlex.quote(part) for part in cmd))
    start = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    elapsed = time.perf_counter() - start
    timings.append({"step": name, "elapsed_seconds": elapsed})
    print(f"{name} finished in {elapsed:.2f}s")


def _workflow_name(base: str, population_size: int | None) -> str:
    return base if population_size is None else f"{base}_{population_size}"


def _run_workflow(workflow: str, output_root: Path, population_size: int | None, python_bin: str) -> dict[str, object]:
    config = WORKFLOW_CONFIGS[workflow]
    workflow_dir = output_root / _workflow_name(workflow, population_size)
    results_dir = workflow_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timings: list[dict[str, object]] = []
    base_args = [python_bin, "--config", str(config), "--output-dir", str(results_dir)]

    _run_step(
        f"{workflow}: Phase 1",
        [python_bin, str(PROJECT_ROOT / "inference" / "scripts" / "run_phase_1.py"), "--config", str(config), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        timings,
    )
    _run_step(
        f"{workflow}: Phase 2",
        [python_bin, str(PROJECT_ROOT / "inference" / "scripts" / "run_phase_2.py"), "--config", str(config), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        timings,
    )
    _run_step(
        f"{workflow}: Phase 3b",
        [python_bin, str(PROJECT_ROOT / "inference" / "scripts" / "run_phase_3b.py"), "--config", str(config), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        timings,
    )
    _run_step(
        f"{workflow}: propagation phase1",
        [python_bin, str(PROJECT_ROOT / "propagation" / "scripts" / "run_phase1_propagation.py"), "--config", str(config), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        timings,
    )
    _run_step(
        f"{workflow}: propagation phase3b",
        [python_bin, str(PROJECT_ROOT / "propagation" / "scripts" / "run_phase3b_propagation.py"), "--config", str(config), "--output-dir", str(results_dir)],
        PROJECT_ROOT,
        timings,
    )

    summary = {
        "workflow": workflow,
        "config": str(config),
        "workflow_dir": str(workflow_dir),
        "results_dir": str(results_dir),
        "population_size": population_size,
        "step_timings": timings,
        "elapsed_seconds": sum(item["elapsed_seconds"] for item in timings),
    }
    with (workflow_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight validation suite over public MesoUQ workflows.")
    parser.add_argument("--workflows", nargs="+", default=["compression_reduced", "indentation_reduced"], choices=sorted(WORKFLOW_CONFIGS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--population-size", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    suite_summary = []
    for workflow in args.workflows:
        suite_summary.append(_run_workflow(workflow, output_root, args.population_size, args.python_bin))

    with (output_root / "workflow_suite_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(suite_summary, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
