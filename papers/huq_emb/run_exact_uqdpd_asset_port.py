#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = REPO_ROOT / "scripts" / "workflows" / "emb" / "huq_emb" / "run_exact_uqdpd_asset_port.py"


def main() -> int:
    namespace = runpy.run_path(str(TARGET), run_name="mesouq_papers_huq_emb_run_exact_uqdpd_asset_port")
    entry = namespace.get("main")
    if not callable(entry):
        raise RuntimeError(f"Missing callable main() in {TARGET}")
    return int(entry())


if __name__ == "__main__":
    raise SystemExit(main())
