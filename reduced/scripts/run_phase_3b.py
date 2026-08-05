#!/usr/bin/env python3
"""
Reduced-model wrapper for hierarchical Phase 3b.

This delegates to the main hierarchical phase-3b driver while selecting the
canonical reduced configuration by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "reduced" / "configs" / "production" / "reduced_config_compression.yaml"
)
MAIN_DRIVER = PROJECT_ROOT / "inference" / "scripts" / "run_phase_3b.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=str, default="_setup")
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--korali-random-seed", type=int, default=None)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(MAIN_DRIVER),
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
    ]
    if args.profiling:
        cmd.append("--profiling")
    cmd.extend(["--device", args.device])
    if args.dataset_name is not None:
        cmd.extend(["--dataset-name", args.dataset_name])
    if args.diameter is not None:
        cmd.extend(["--diameter", str(args.diameter)])
    if args.korali_random_seed is not None:
        cmd.extend(["--korali-random-seed", str(args.korali_random_seed)])
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
