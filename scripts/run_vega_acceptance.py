#!/usr/bin/env python3
"""
Run the Vega-first acceptance command for MesoUQ.

This is intentionally a thin wrapper around the richer
`scripts/vega/run_validation_suite.py` operator runner.
It captures environment metadata, invokes the validation suite once, and writes
one machine-readable acceptance report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / "vega_acceptance"
VALIDATION_RUNNER = PROJECT_ROOT / "scripts" / "vega" / "run_validation_suite.py"
DEFAULT_WORKFLOWS = [
    "compression:reduced-model:validation",
    "indentation:reduced-model:validation",
]


def _run_command(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(cwd), env=env, text=True, capture_output=True)


def _capture_command(command: list[str], cwd: Path, env: dict[str, str], logs_dir: Path, name: str) -> dict[str, Any]:
    start = time.perf_counter()
    result = _run_command(command, cwd, env)
    elapsed = time.perf_counter() - start
    stdout_path = logs_dir / f"{name}.stdout.log"
    stderr_path = logs_dir / f"{name}.stderr.log"
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _optional_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        text = (result.stdout or result.stderr or "").strip()
        return text[:4000]
    except Exception as exc:
        return f"unavailable: {exc}"


def _environment_snapshot(python_bin: str, korali_pythonpath: str | None) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python_bin": python_bin,
        "python_version": platform.python_version(),
        "korali_pythonpath": korali_pythonpath or "",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "loaded_modules": os.environ.get("LOADEDMODULES", ""),
        "which_python": _optional_command(["which", python_bin]),
        "nvidia_smi": _optional_command(["nvidia-smi", "-L"]),
        "nvcc_version": _optional_command(["nvcc", "--version"]),
        "mpirun_version": _optional_command(["mpirun", "--version"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Vega-first MesoUQ acceptance command.")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--korali-pythonpath", type=str, default=None)
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=DEFAULT_WORKFLOWS,
        help=(
            "Validation selections in experiment:model-family:profile form. "
            "Legacy aliases remain accepted by the wrapped runner."
        ),
    )
    parser.add_argument("--cpu-ranks", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--config-override", action="append", default=[])
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.korali_pythonpath:
        env["PYTHONPATH"] = ":".join([args.korali_pythonpath, str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(":")
    else:
        env["PYTHONPATH"] = ":".join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).rstrip(":")

    report: dict[str, Any] = {
        "target": "vega",
        "status": "running",
        "environment": _environment_snapshot(args.python_bin, args.korali_pythonpath),
        "steps": [],
        "artifacts": {},
    }

    runner_output_root = output_root / "validation_suite"
    runner_cmd = [
        args.python_bin,
        str(VALIDATION_RUNNER),
        "--output-root",
        str(runner_output_root),
        "--python-bin",
        args.python_bin,
        "--cpu-ranks",
        str(args.cpu_ranks),
    ]
    if args.korali_pythonpath:
        runner_cmd.extend(["--korali-pythonpath", args.korali_pythonpath])
    if args.population_size is not None:
        runner_cmd.extend(["--population-size", str(args.population_size)])
    if args.workflows:
        runner_cmd.extend(["--workflows", *args.workflows])
    for override in args.config_override:
        runner_cmd.extend(["--config-override", override])

    step = _capture_command(runner_cmd, PROJECT_ROOT, env, logs_dir, "validation_suite")
    report["steps"].append(step)

    suite_summary_path = runner_output_root / "workflow_suite_summary.json"
    report["artifacts"]["validation_suite_root"] = str(runner_output_root)
    report["artifacts"]["workflow_suite_summary"] = str(suite_summary_path)

    if step["returncode"] == 0 and suite_summary_path.exists():
        report["status"] = "passed"
        try:
            report["workflow_suite"] = json.loads(suite_summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report["status"] = "partial"
            report["workflow_suite_load_error"] = str(exc)
    else:
        report["status"] = "failed"

    report_path = output_root / "vega_acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Acceptance report: {report_path}")
    if report["status"] == "failed":
        print("Vega acceptance failed. See logs and report for details.")
        return 1
    print(f"Vega acceptance status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
