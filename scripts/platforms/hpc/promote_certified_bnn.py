#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    script_dir = Path(__file__).resolve().parent
    site = os.environ.get("HPC_SITE", "vega").strip().lower()

    if site not in {"vega", "karolina"}:
        raise SystemExit(f"Unsupported HPC_SITE={site!r}. Expected 'vega' or 'karolina'.")

    target = script_dir.parent / site / "promote_certified_bnn.py"
    return subprocess.call([sys.executable, str(target), *args])


if __name__ == "__main__":
    raise SystemExit(main())
