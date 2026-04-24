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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.vega import (
    DEFAULT_VEGA_MIRHEO_MODULES,
    DEFAULT_VEGA_MODULES,
    find_external_korali_entries,
    get_vega_paths,
    load_mirheo_source_lock,
    resolve_mirheo_source,
)


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


def _kpsewhich(name: str) -> str:
    resolved = _command_path("kpsewhich")
    if not resolved:
        return ""
    result = subprocess.run(
        [resolved, name],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def collect_diagnostics(
    python_bin: str,
    *,
    with_mirheo: bool = False,
    with_tex: bool = False,
) -> dict[str, object]:
    paths = get_vega_paths(REPO_ROOT)
    checks: list[dict[str, str]] = []
    recommended_modules = list(DEFAULT_VEGA_MIRHEO_MODULES if with_mirheo else DEFAULT_VEGA_MODULES)

    loaded_modules = os.environ.get("LOADEDMODULES", "")
    checks.append(
        _check(
            "loaded_modules",
            "ok" if loaded_modules else "warn",
            loaded_modules or f"empty; recommended stack: {' '.join(recommended_modules)}",
        )
    )

    required_commands = ["python", "mpicxx", "nvcc", "pkg-config", "meson", "ninja"]
    if with_mirheo:
        required_commands.extend(["cmake", "make", "h5dump"])
    if with_tex:
        required_commands.extend(["latex", "pdflatex", "kpsewhich", "dvipng"])
    for command in required_commands:
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

    python_modules = ["meso_uq", "mpi4py", "pybind11", "mesonbuild"]
    if with_mirheo:
        python_modules.extend(["h5py", "mirheo"])
    for module_name in python_modules:
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

    if with_mirheo:
        mirheo_lock = load_mirheo_source_lock(REPO_ROOT)
        checks.append(
            _check(
                "mirheo_lock",
                "ok" if paths.mirheo_source_lock.is_file() else "warn",
                str(paths.mirheo_source_lock if paths.mirheo_source_lock.is_file() else mirheo_lock["source_path"]),
            )
        )
        try:
            mirheo_source = resolve_mirheo_source(REPO_ROOT)
            source_status = "ok" if mirheo_source.is_dir() else "warn"
            source_details = str(mirheo_source)
        except Exception as exc:  # pragma: no cover - defensive path
            source_status = "warn"
            source_details = str(exc)
            mirheo_source = None
        checks.append(_check("mirheo_source", source_status, source_details))
        checks.append(
            _check(
                "repo_local_mirheo_env_script",
                "ok" if paths.mirheo_env_script.is_file() else "warn",
                str(paths.mirheo_env_script),
            )
        )
        checks.append(
            _check(
                "repo_local_mirheo_snapshot",
                "ok" if paths.mirheo_snapshot_path.is_file() else "warn",
                str(paths.mirheo_snapshot_path),
            )
        )
    if with_tex:
        checks.append(
            _check(
                "repo_local_tinytex_env_script",
                "ok" if paths.tinytex_env_script.is_file() else "warn",
                str(paths.tinytex_env_script),
            )
        )
        checks.append(
            _check(
                "repo_local_tinytex_bin_dir",
                "ok" if paths.tinytex_bin_dir.is_dir() else "warn",
                str(paths.tinytex_bin_dir),
            )
        )
        for tex_file in ("helvet.sty", "sansmath.sty", "revtex4-2.cls", "preview.sty"):
            resolved = _kpsewhich(tex_file)
            checks.append(
                _check(
                    f"tex:{tex_file}",
                    "ok" if resolved else "warn",
                    resolved or "not found in current TeX tree",
                )
            )

    return {
        "repo_root": str(paths.repo_root),
        "python_bin": python_bin,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "recommended_modules": recommended_modules,
        "paths": {
            "vega_root": str(paths.vega_root),
            "korali_source": str(paths.korali_source),
            "korali_prefix": str(paths.korali_prefix),
            "korali_site_packages": str(paths.korali_site_packages),
            "korali_env_script": str(paths.korali_env_script),
            "tinytex_root": str(paths.tinytex_root),
            "tinytex_bin_dir": str(paths.tinytex_bin_dir),
            "tinytex_env_script": str(paths.tinytex_env_script),
            "mirheo_source_lock": str(paths.mirheo_source_lock),
            "mirheo_build_dir": str(paths.mirheo_build_dir),
            "mirheo_prefix": str(paths.mirheo_prefix),
            "mirheo_env_script": str(paths.mirheo_env_script),
            "mirheo_snapshot_path": str(paths.mirheo_snapshot_path),
        },
        "checks": checks,
        "with_mirheo": with_mirheo,
        "with_tex": with_tex,
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
        "WARNING: scripts/platforms/vega/doctor_vega.py is deprecated. "
        "Use scripts/platforms/hpc/doctor_hpc.py with HPC_SITE=vega|karolina.",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="Inspect the local Vega bootstrap/runtime surface.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument(
        "--with-mirheo",
        action="store_true",
        default=False,
        help="Include repo-local Mirheo bootstrap/runtime checks in addition to the core Korali checks.",
    )
    parser.add_argument(
        "--with-tex",
        action="store_true",
        default=False,
        help="Include repo-local TinyTeX/Matplotlib usetex checks needed by paper-facing figure generation.",
    )
    args = parser.parse_args(argv)

    report = collect_diagnostics(
        args.python_bin,
        with_mirheo=args.with_mirheo,
        with_tex=args.with_tex,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)

    non_ok = [check for check in report["checks"] if check["status"] != "ok"]
    return 1 if args.strict and non_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
