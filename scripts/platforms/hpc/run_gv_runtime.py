#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_target_script() -> Path:
    site = os.environ.get("HPC_SITE", "vega").strip().lower()
    if site not in {"vega", "karolina"}:
        raise SystemExit(f"Unsupported HPC_SITE={site!r}. Expected 'vega' or 'karolina'.")
    return SCRIPT_DIR.parent / site / "run_gv_runtime.py"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    target = _resolve_target_script()
    return subprocess.call([sys.executable, str(target), *args])


if __name__ == "__main__":
    raise SystemExit(main())
