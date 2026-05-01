#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
WORKFLOW_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_runtime.py"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    return subprocess.call([sys.executable, str(WORKFLOW_PATH), "--platform", "karolina", *args])


if __name__ == "__main__":
    raise SystemExit(main())
