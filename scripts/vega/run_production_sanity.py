#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.production_sanity import (  # noqa: E402
    PRODUCTION_SMOKE_OVERRIDES,
    load_korali_build_state,
    resolve_production_sanity_selections,
    write_production_smoke_config,
)
from meso_uq.vega_workflows import format_command, selection_key  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_vega" / "production_sanity"


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical Vega production sanity smoke with reduced-cost overrides."
    )
    parser.add_argument(
        "--selection",
        action="append",
        default=[],
        help="Selection in experiment:model-family:production form. Defaults to compression:full-model:production.",
    )
    parser.add_argument(
        "--all-lanes",
        action="store_true",
        default=False,
        help="Run all production lanes across experiment and model-family axes.",
    )
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--phase2-cpu-ranks", type=int, default=2)
    args = parser.parse_args(argv)

    if args.phase2_cpu_ranks < 1:
        raise ValueError("--phase2-cpu-ranks must be a positive integer.")

    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    selections = resolve_production_sanity_selections(args.selection, all_lanes=args.all_lanes)
    base_configs: dict[str, str] = {}
    sanity_configs: dict[str, str] = {}
    config_overrides: list[str] = []
    for selection in selections:
        base_config, sanity_config = write_production_smoke_config(REPO_ROOT, selection, output_root)
        base_configs[selection_key(selection)] = str(base_config)
        sanity_configs[selection_key(selection)] = str(sanity_config)
        config_overrides.append(f"{selection_key(selection)}={sanity_config}")

    matrix_root = output_root / "matrix"
    command = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "vega" / "run_workflow_matrix.py"),
        "--output-root",
        str(matrix_root),
        "--python-bin",
        args.python_bin,
        "--phase2-cpu-ranks",
        str(args.phase2_cpu_ranks),
        "--skip-phase1-map",
    ]
    for selection in selections:
        command.extend(["--selection", selection_key(selection)])
    for override in config_overrides:
        command.extend(["--config-override", override])

    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    stdout_log = logs_root / "workflow_matrix.stdout.log"
    stderr_log = logs_root / "workflow_matrix.stderr.log"
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")

    matrix_report_path = matrix_root / "workflow_matrix_report.json"
    matrix_report = None
    if matrix_report_path.exists():
        matrix_report = json.loads(matrix_report_path.read_text(encoding="utf-8"))

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "status": "passed" if result.returncode == 0 else "failed",
        "python_bin": args.python_bin,
        "phase2_cpu_ranks": args.phase2_cpu_ranks,
        "default_selection": selection_key(selections[0]) if selections else None,
        "selections": [
            {
                "selection": selection_key(selection),
                "base_config": base_configs[selection_key(selection)],
                "sanity_config": sanity_configs[selection_key(selection)],
            }
            for selection in selections
        ],
        "smoke_overrides": dict(PRODUCTION_SMOKE_OVERRIDES),
        "korali": load_korali_build_state(REPO_ROOT),
        "matrix": {
            "command": command,
            "command_text": format_command(command),
            "returncode": result.returncode,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "report_path": str(matrix_report_path),
            "report_status": matrix_report.get("status") if matrix_report else None,
        },
    }

    report_path = output_root / "production_sanity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Production sanity report: {report_path}")
    print(f"Production sanity status: {report['status']}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
