#!/usr/bin/env python3
"""Inspect the local Vega environment for fresh-clone workflow readiness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.vega import DEFAULT_VEGA_MODULES, find_external_korali_entries, get_vega_paths


def _command_path(name: str) -> str:
    return shutil.which(name) or ""


def _python_module_spec(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return ""
    return str(spec.origin or "")


def _pkg_config_version(package: str) -> str:
    result = subprocess.run(
        ["pkg-config", "--modversion", package],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _check(name: str, status: str, details: str) -> dict[str, str]:
    return {"name": name, "status": status, "details": details}


def collect_diagnostics(python_bin: str) -> dict[str, object]:
    paths = get_vega_paths(REPO_ROOT)
    checks: list[dict[str, str]] = []

    loaded_modules = os.environ.get("LOADEDMODULES", "")
    checks.append(
        _check(
            "loaded_modules",
            "ok" if loaded_modules else "warn",
            loaded_modules or f"empty; recommended stack: {' '.join(DEFAULT_VEGA_MODULES)}",
        )
    )

    for command in ("python", "mpicxx", "nvcc", "pkg-config", "meson", "ninja"):
        resolved = _command_path(command)
        checks.append(
            _check(
                f"command:{command}",
                "ok" if resolved else "warn",
                resolved or f"missing from PATH; expected for Vega bootstrap",
            )
        )

    for package in ("gsl", "eigen3"):
        version = _pkg_config_version(package)
        checks.append(
            _check(
                f"pkg-config:{package}",
                "ok" if version else "warn",
                version or f"missing; load the matching module before bootstrap",
            )
        )

    for module_name in ("meso_uq", "mpi4py", "pybind11", "mesonbuild"):
        origin = _python_module_spec(module_name)
        checks.append(
            _check(
                f"python:{module_name}",
                "ok" if origin else "warn",
                origin or "not importable in the current Python environment",
            )
        )

    local_korali_init = paths.korali_site_packages / "korali" / "__init__.py"
    checks.append(
        _check(
            "repo_local_korali",
            "ok" if local_korali_init.is_file() else "warn",
            str(local_korali_init if local_korali_init.is_file() else paths.korali_prefix),
        )
    )

    korali_origin = _python_module_spec("korali")
    if korali_origin:
        status = "ok" if str(paths.korali_site_packages) in korali_origin else "warn"
        details = korali_origin
    else:
        status = "warn"
        details = "korali is not importable in the current Python environment"
    checks.append(_check("python:korali", status, details))

    external_korali = find_external_korali_entries(
        os.environ.get("PYTHONPATH", ""),
        paths.repo_root,
        paths.korali_site_packages,
    )
    checks.append(
        _check(
            "pythonpath:external_korali",
            "warn" if external_korali else "ok",
            ", ".join(external_korali) if external_korali else "none",
        )
    )

    env_script_exists = paths.korali_env_script.is_file()
    checks.append(
        _check(
            "repo_local_env_script",
            "ok" if env_script_exists else "warn",
            str(paths.korali_env_script),
        )
    )

    return {
        "repo_root": str(paths.repo_root),
        "python_bin": python_bin,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "recommended_modules": list(DEFAULT_VEGA_MODULES),
        "paths": {
            "vega_root": str(paths.vega_root),
            "korali_source": str(paths.korali_source),
            "korali_prefix": str(paths.korali_prefix),
            "korali_site_packages": str(paths.korali_site_packages),
            "korali_env_script": str(paths.korali_env_script),
        },
        "checks": checks,
    }


def _print_human(report: dict[str, object]) -> None:
    print(f"Repo root: {report['repo_root']}")
    print(f"Python: {report['python_bin']} ({report['python_version']})")
    print(f"Host: {report['hostname']}")
    print("Recommended modules:")
    print("  module load " + " ".join(report["recommended_modules"]))
    print("")
    for check in report["checks"]:
        label = check["status"].upper().ljust(4)
        print(f"[{label}] {check['name']}: {check['details']}")


def main(argv: list[str] | None = None) -> int:
    print(
        "WARNING: scripts/vega/doctor_vega.py is deprecated. "
        "Use scripts/hpc/doctor_hpc.py with HPC_SITE=vega|karolina.",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="Inspect the local Vega bootstrap/runtime surface.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    report = collect_diagnostics(args.python_bin)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)

    non_ok = [check for check in report["checks"] if check["status"] != "ok"]
    return 1 if args.strict and non_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
