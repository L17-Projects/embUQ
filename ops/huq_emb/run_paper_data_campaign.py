#!/usr/bin/env python3
"""Compatibility shim for the non-live HUQ-EMB paper-data runner."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = REPO_ROOT / "scripts" / "workflows" / "emb" / "huq_emb" / "run_paper_data_campaign.py"


def _target_main():
    namespace = runpy.run_path(str(TARGET), run_name="mesouq_ops_huq_emb_run_paper_data_campaign")
    main = namespace.get("main")
    if not callable(main):
        raise RuntimeError(f"Compatibility target does not expose a callable main(): {TARGET}")
    return main


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    return int(_target_main()(args))


if __name__ == "__main__":
    raise SystemExit(main())
