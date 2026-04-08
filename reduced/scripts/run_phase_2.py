#!/usr/bin/env python3
"""
Reduced-model wrapper for hierarchical Phase 2.

This delegates to the main hierarchical phase-2 driver while selecting the
canonical reduced configuration by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "reduced" / "configs" / "production" / "reduced_config_compression.yaml"
MAIN_DRIVER = PROJECT_ROOT / "inference" / "scripts" / "run_phase_2.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=str, default="_setup")
    args = parser.parse_args()

    cmd = [sys.executable, str(MAIN_DRIVER), "--config", args.config, "--output-dir", args.output_dir]
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
