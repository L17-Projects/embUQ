#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.platforms.site_selector import SiteSelectionError, resolve_hpc_site  # noqa: E402

TARGET = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_runtime.py"


def _pop_arg_value(args: list[str], flag: str) -> tuple[list[str], str | None]:
    remaining: list[str] = []
    value: str | None = None
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == flag:
            if index + 1 >= len(args):
                print(f"{flag} requires a value", file=sys.stderr)
                raise SystemExit(2)
            value = args[index + 1]
            index += 2
            continue
        prefix = f"{flag}="
        if arg.startswith(prefix):
            value = arg[len(prefix):]
            index += 1
            continue
        remaining.append(arg)
        index += 1
    return remaining, value


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    try:
        args, cli_site = _pop_arg_value(args, "--site")
        args, downstream_platform = _pop_arg_value(args, "--platform")
    except SystemExit as exc:
        return int(exc.code)

    try:
        site = resolve_hpc_site(cli_site=cli_site, env=os.environ, allow_hostname=True, default="vega")
    except SiteSelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if downstream_platform is not None and downstream_platform != site:
        print(
            f"Conflicting site selectors: --site={site} and --platform={downstream_platform}.",
            file=sys.stderr,
        )
        return 2

    return subprocess.call([sys.executable, str(TARGET), "--platform", site, *args])


if __name__ == "__main__":
    raise SystemExit(main())
