#!/usr/bin/env python3
"""
Run reduced Phase 3b for a single dataset by writing a temporary config override.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "reduced" / "configs" / "production" / "reduced_config_compression.yaml"
MAIN_DRIVER = PROJECT_ROOT / "reduced" / "scripts" / "run_phase_3b.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--diameter", type=float, required=True)
    parser.add_argument("--output-dir", type=str, default="_setup")
    parser.add_argument("--profiling", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config, "rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)
    config["experiments"] = [{"name": args.experiment, "enabled": True, "diameters": [args.diameter]}]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        tmp_config = handle.name

    cmd = [sys.executable, str(MAIN_DRIVER), "--config", tmp_config, "--output-dir", args.output_dir]
    if args.profiling:
        cmd.append("--profiling")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
