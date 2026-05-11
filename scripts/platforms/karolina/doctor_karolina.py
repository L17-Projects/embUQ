#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    script_dir = Path(__file__).resolve().parent
    target = script_dir.parent / "vega" / "doctor_vega.py"
    env = os.environ.copy()
    env["HPC_SITE"] = "karolina"
    env["MESOUQ_SITE"] = "karolina"
    return subprocess.call([sys.executable, str(target), *args], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
