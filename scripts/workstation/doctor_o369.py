#!/usr/bin/env python3
"""
Pre-flight health check for o369 workstation runs.

Checks:
  - repo-local Korali install under _vega/korali/install
  - _vega/korali/env.sh exists
  - `python -c "import korali"` resolves inside _vega/korali/install/
  - mpirun is available on PATH
  - free RAM > 2 GB

Prints PASS / WARN / FAIL per check and exits non-zero if any FAIL.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KORALI_INSTALL = REPO_ROOT / "_vega" / "korali" / "install"
KORALI_ENV_SH = REPO_ROOT / "_vega" / "korali" / "env.sh"
MIN_FREE_RAM_GB = 2.0

_PASS = "PASS"
_WARN = "WARN"
_FAIL = "FAIL"


def _tag(level: str, msg: str) -> str:
    return f"[{level}] {msg}"


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def check_korali_install() -> tuple[str, str]:
    if KORALI_INSTALL.is_dir():
        return _PASS, f"Korali install dir found: {KORALI_INSTALL}"
    return _FAIL, f"Korali install dir missing: {KORALI_INSTALL}"


def check_korali_env_sh() -> tuple[str, str]:
    if KORALI_ENV_SH.is_file():
        return _PASS, f"Korali env.sh found: {KORALI_ENV_SH}"
    return _FAIL, f"Korali env.sh missing: {KORALI_ENV_SH}"


def check_korali_import(python_bin: str = sys.executable) -> tuple[str, str]:
    try:
        result = subprocess.run(
            [
                python_bin,
                "-c",
                "import korali, inspect; print(inspect.getfile(korali))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return _FAIL, f"Python binary not found: {python_bin}"
    except subprocess.TimeoutExpired:
        return _FAIL, "Timeout while checking `import korali`"

    if result.returncode != 0:
        stderr = result.stderr.strip().splitlines()
        detail = stderr[-1] if stderr else "(no stderr)"
        return _FAIL, f"`import korali` failed: {detail}"

    korali_file = result.stdout.strip()
    resolved_korali_path = Path(korali_file).resolve()
    expected_root = KORALI_INSTALL.resolve()
    if _path_is_within(resolved_korali_path, expected_root):
        return _PASS, f"korali resolves inside _vega/korali/install: {resolved_korali_path}"
    return _WARN, (
        f"korali found but NOT under _vega/korali/install.\n"
        f"  Found:    {resolved_korali_path}\n"
        f"  Expected: {expected_root}/..."
    )


def check_mpirun() -> tuple[str, str]:
    path = shutil.which("mpirun")
    if path:
        return _PASS, f"mpirun found: {path}"
    return _FAIL, "mpirun not found on PATH"


_MEMINFO_PATH = "/proc/meminfo"


def check_free_ram(meminfo_path: str = _MEMINFO_PATH) -> tuple[str, str]:
    try:
        with open(meminfo_path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    free_gb = kb / (1024 * 1024)
                    if free_gb >= MIN_FREE_RAM_GB:
                        return _PASS, f"Free RAM: {free_gb:.2f} GB (>= {MIN_FREE_RAM_GB} GB)"
                    return _FAIL, (
                        f"Free RAM: {free_gb:.2f} GB — below required {MIN_FREE_RAM_GB} GB. "
                        "Close other processes before running."
                    )
        return _WARN, f"MemAvailable not found in {meminfo_path}; cannot check RAM"
    except OSError:
        return _WARN, f"{meminfo_path} unavailable; skipping RAM check"


def run_checks(python_bin: str = sys.executable) -> int:
    checks = [
        ("korali_install", check_korali_install),
        ("korali_env_sh", check_korali_env_sh),
        ("korali_import", lambda: check_korali_import(python_bin)),
        ("mpirun", check_mpirun),
        ("free_ram", check_free_ram),
    ]

    failures = 0
    for name, fn in checks:
        level, msg = fn()
        print(_tag(level, f"{name}: {msg}"))
        if level == _FAIL:
            failures += 1

    print()
    if failures:
        print(f"Doctor: {failures} check(s) FAILED — fix before running local matrix.")
    else:
        print("Doctor: all checks passed.")
    return failures


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-flight health check for o369 workstation runs."
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python binary to use for `import korali` check (default: current interpreter).",
    )
    args = parser.parse_args(argv)
    return run_checks(python_bin=args.python_bin)


if __name__ == "__main__":
    raise SystemExit(main())
