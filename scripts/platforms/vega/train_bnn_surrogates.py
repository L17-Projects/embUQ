#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SITE = 'vega'
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TARGET = REPO_ROOT / "scripts" / "platforms" / "hpc" / 'train_bnn_surrogates.py'


def _requested_site(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg == "--site":
            if index + 1 >= len(args):
                print(f"{Path(__file__).name}: --site requires a value", file=sys.stderr)
                raise SystemExit(2)
            return args[index + 1]
        if arg.startswith("--site="):
            return arg.split("=", 1)[1]
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    requested_site = _requested_site(args)
    if requested_site is None:
        args = ["--site", SITE, *args]
    elif requested_site != SITE:
        print(
            f"{Path(__file__).name} is the {SITE} compatibility entrypoint; "
            f"use scripts/platforms/hpc/train_bnn_surrogates.py for --site {requested_site}.",
            file=sys.stderr,
        )
        return 2
    return subprocess.call([sys.executable, str(TARGET), *args])


if __name__ == "__main__":
    raise SystemExit(main())
