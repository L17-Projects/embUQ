#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SITE = os.environ.get("HPC_SITE", "vega").strip().lower()

if SITE not in {"vega", "karolina"}:
    raise SystemExit(f"Unsupported HPC_SITE={SITE!r}. Expected 'vega' or 'karolina'.")

TARGET = SCRIPT_DIR.parent / SITE / "train_bnn_surrogates.py"
raise SystemExit(subprocess.call([sys.executable, str(TARGET), *sys.argv[1:]]))
