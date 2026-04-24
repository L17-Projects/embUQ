#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SITE = os.environ.get("HPC_SITE", "vega").strip().lower()

if SITE == "vega":
    target = SCRIPT_DIR.parent / "vega" / "doctor_vega.py"
elif SITE == "karolina":
    target = SCRIPT_DIR.parent / "karolina" / "doctor_karolina.py"
else:
    raise SystemExit(f"Unsupported HPC_SITE={SITE!r}. Expected 'vega' or 'karolina'.")

raise SystemExit(subprocess.call([sys.executable, str(target), *sys.argv[1:]]))
