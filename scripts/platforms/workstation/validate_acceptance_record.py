#!/usr/bin/env python3
"""
Validate a machine-readable workstation acceptance record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_ENVIRONMENT_FIELDS = {
    "timestamp_utc",
    "hostname",
    "platform",
    "python_bin",
    "python_version",
    "gpu_summary",
    "git_commit",
}
REQUIRED_STEP_NAMES = {
    "install",
    "surrogate_retraining",
    "phase1_gpu_batched",
    "phase3b_gpu_batched",
    "map_extraction_and_plotting",
}
ALLOWED_REPORT_STATUSES = {"passed", "failed", "partial"}
ALLOWED_STEP_STATUSES = {"passed", "failed", "partial", "skipped"}


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_artifact_paths(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, str):
        paths.append(Path(value))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                paths.append(Path(item))
    elif isinstance(value, dict):
        for nested in value.values():
            paths.extend(_iter_artifact_paths(nested))
    return paths


def validate_report(report: dict[str, Any], must_exist: bool = False) -> list[str]:
    errors: list[str] = []

    if report.get("target") != "workstation":
        errors.append("report.target must be 'workstation'")
    if report.get("status") not in ALLOWED_REPORT_STATUSES:
        errors.append(f"report.status must be one of {sorted(ALLOWED_REPORT_STATUSES)}")

    environment = report.get("environment")
    if not isinstance(environment, dict):
        errors.append("report.environment must be a mapping")
    else:
        missing_environment = sorted(REQUIRED_ENVIRONMENT_FIELDS - set(environment))
        if missing_environment:
            errors.append(f"report.environment is missing fields: {missing_environment}")

    steps = report.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("report.steps must be a non-empty list")
    else:
        seen_names = set()
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"report.steps[{index}] must be a mapping")
                continue
            name = step.get("name")
            status = step.get("status")
            commands = step.get("commands")
            artifacts = step.get("artifacts")
            if not isinstance(name, str) or not name:
                errors.append(f"report.steps[{index}].name must be a non-empty string")
            else:
                seen_names.add(name)
            if status not in ALLOWED_STEP_STATUSES:
                errors.append(f"report.steps[{index}].status must be one of {sorted(ALLOWED_STEP_STATUSES)}")
            if not isinstance(commands, list) or not commands or not all(isinstance(command, str) and command for command in commands):
                errors.append(f"report.steps[{index}].commands must be a non-empty list of strings")
            if not isinstance(artifacts, dict) or not artifacts:
                errors.append(f"report.steps[{index}].artifacts must be a non-empty mapping")
            elif must_exist:
                for artifact_path in _iter_artifact_paths(artifacts):
                    if not artifact_path.exists():
                        errors.append(f"missing step artifact path: {artifact_path}")
        missing_steps = sorted(REQUIRED_STEP_NAMES - seen_names)
        if missing_steps:
            errors.append(f"report.steps is missing required step names: {missing_steps}")

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("report.artifacts must be a non-empty mapping")
    elif must_exist:
        for artifact_path in _iter_artifact_paths(artifacts):
            if not artifact_path.exists():
                errors.append(f"missing top-level artifact path: {artifact_path}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a workstation acceptance report.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--must-exist", action="store_true", help="Require referenced artifact paths to exist locally.")
    args = parser.parse_args(argv)

    report = _load_report(args.report)
    errors = validate_report(report, must_exist=args.must_exist)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated workstation acceptance report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
