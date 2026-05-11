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
    DEFAULT_KAROLINA_MIRHEO_MODULES,
    DEFAULT_KAROLINA_MODULES,
    DEFAULT_KAROLINA_RUNTIME_MODULES,
    DEFAULT_VEGA_MIRHEO_MODULES,
    DEFAULT_VEGA_MODULES,
    DEFAULT_VEGA_RUNTIME_MODULES,
    find_external_korali_entries,
    get_runtime_paths,
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


def _resolve_scale_space_binary() -> tuple[str, str]:
    env_binary = os.environ.get("GV_SCALE_SPACE_BINARY", "").strip()
    if env_binary and Path(env_binary).is_file():
        return env_binary, "GV_SCALE_SPACE_BINARY"

    cgal_root = os.environ.get("GV_CGAL_TOOLS_ROOT", "").strip()
    if cgal_root:
        candidates = (
            Path(cgal_root) / "scale_space",
            Path(cgal_root) / "cgal_scripts" / "scale_space",
            Path(cgal_root) / "build" / "cgal_scripts" / "scale_space",
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate), "GV_CGAL_TOOLS_ROOT"

    which_binary = _command_path("scale_space")
    if which_binary:
        return which_binary, "PATH"
    return "", ""


def _resolve_scale_space_dynamic_libs(binary: str) -> tuple[bool, str]:
    if not binary:
        return False, "scale_space binary was not resolved"
    result = subprocess.run(["ldd", binary], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return False, f"ldd failed: {(result.stderr or result.stdout).strip()}"
    output = ((result.stdout or "") + (result.stderr or "")).strip()
    missing: list[str] = [
        line.strip()
        for line in output.splitlines()
        if "not found" in line
    ]
    if missing:
        return False, "missing dynamic libs: " + ", ".join(missing)
    return True, "all dynamic libs resolved"


def _find_mirheo_lib_paths(paths) -> list[str]:
    candidates = []
    for root in (paths.mirheo_prefix, paths.mirheo_package_dir, paths.gv_venv_site_packages):
        prefix = Path(root)
        if not prefix.is_dir():
            continue
        candidates.extend(str(path) for path in prefix.glob("**/libmirheo*.so*"))
    return candidates


def _check(name: str, status: str, details: str) -> dict[str, str]:
    return {"name": name, "status": status, "details": details}


def _default_runtime_site() -> str:
    for name in ("MESOUQ_SITE", "HPC_SITE"):
        value = os.environ.get(name, "").strip().lower()
        if value in {"vega", "karolina"}:
            return value
    parent = Path(__file__).resolve().parent.name
    return parent if parent in {"vega", "karolina"} else "vega"


def _recommended_modules(site: str, *, with_mirheo: bool, with_gv_runtime: bool) -> list[str]:
    if site == "karolina":
        if with_gv_runtime:
            return list(DEFAULT_KAROLINA_MIRHEO_MODULES + DEFAULT_KAROLINA_RUNTIME_MODULES[len(DEFAULT_KAROLINA_MODULES) :])
        if with_mirheo:
            return list(DEFAULT_KAROLINA_MIRHEO_MODULES)
        return list(DEFAULT_KAROLINA_MODULES)
    if with_gv_runtime:
        return list(DEFAULT_VEGA_MIRHEO_MODULES + DEFAULT_VEGA_RUNTIME_MODULES[len(DEFAULT_VEGA_MODULES) :])
    if with_mirheo:
        return list(DEFAULT_VEGA_MIRHEO_MODULES)
    return list(DEFAULT_VEGA_MODULES)


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
    site: str | None = None,
    with_mirheo: bool = False,
    with_gv_runtime: bool = False,
    with_tex: bool = False,
) -> dict[str, object]:
    paths = get_runtime_paths(REPO_ROOT, site=site or _default_runtime_site())
    checks: list[dict[str, str]] = []
    include_mirheo_checks = with_mirheo or with_gv_runtime
    recommended_modules = _recommended_modules(
        paths.site,
        with_mirheo=with_mirheo,
        with_gv_runtime=with_gv_runtime,
    )

    loaded_modules = os.environ.get("LOADEDMODULES", "")
    checks.append(
        _check(
            "loaded_modules",
            "ok" if loaded_modules else "warn",
            loaded_modules or f"empty; recommended stack: {' '.join(recommended_modules)}",
        )
    )

    required_commands = ["python", "mpicxx", "nvcc", "pkg-config"]
    if not with_gv_runtime:
        required_commands.extend(["meson", "ninja"])
    if include_mirheo_checks:
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

    python_modules = ["meso_uq"]
    if not with_gv_runtime:
        python_modules.extend(["mpi4py", "pybind11", "mesonbuild"])
    if include_mirheo_checks:
        python_modules.extend(["h5py", "mirheo"])
    if with_gv_runtime:
        python_modules.append("MDAnalysis")
    for module_name in python_modules:
        origin = _python_module_spec(module_name)
        checks.append(
            _check(
                f"python:{module_name}",
                "ok" if origin else "warn",
                origin or "not importable in the current Python environment",
            )
        )

    if not with_gv_runtime:
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

    if include_mirheo_checks:
        mirheo_lock = load_mirheo_source_lock(REPO_ROOT, site=paths.site)
        checks.append(
            _check(
                "mirheo_lock",
                "ok" if paths.mirheo_source_lock.is_file() else "warn",
                str(paths.mirheo_source_lock if paths.mirheo_source_lock.is_file() else mirheo_lock["source_path"]),
            )
        )
        try:
            mirheo_source = resolve_mirheo_source(REPO_ROOT, site=paths.site)
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
        libmirheo_paths = _find_mirheo_lib_paths(paths)
        checks.append(
            _check(
                "mirheo_libmirheo",
                "ok" if libmirheo_paths else "warn",
                ", ".join(libmirheo_paths) if libmirheo_paths else "libmirheo .so not found in repo-local Mirheo install area",
            )
        )
        checks.append(
            _check(
                "repo_local_gv_venv_env_script",
                "ok" if paths.gv_venv_env_script.is_file() else "warn",
                str(paths.gv_venv_env_script),
            )
        )
        scale_binary, scale_origin = _resolve_scale_space_binary()
        if scale_binary:
            scale_status, scale_details = _resolve_scale_space_dynamic_libs(scale_binary)
            checks.append(
                _check(
                    f"scale_space_binary:{scale_origin}",
                    "ok" if scale_status else "warn",
                    f"{scale_binary}: {scale_details}",
                )
            )
        else:
            checks.append(
                _check(
                    "scale_space_binary",
                    "warn",
                    "GV_SCALE_SPACE_BINARY was not set and scale_space is not resolvable from PATH/GV_CGAL_TOOLS_ROOT",
                )
            )
        if any(check["name"] == "command:mpicxx" and check["status"] == "ok" for check in checks):
            checks.append(
                _check(
                    "openmpi_lib_path",
                    "ok" if bool(os.environ.get("MESOUQ_OPENMPI_LIB_DIR")) else "warn",
                    os.environ.get("MESOUQ_OPENMPI_LIB_DIR", "MESOUQ_OPENMPI_LIB_DIR is not set; runtime env script should infer it"),
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
        "site": paths.site,
        "python_bin": python_bin,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "recommended_modules": recommended_modules,
        "paths": {
            "site_root": str(paths.site_root),
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
            "gv_venv_root": str(paths.gv_venv_root),
            "gv_venv_site_packages": str(paths.gv_venv_site_packages),
            "gv_venv_env_script": str(paths.gv_venv_env_script),
            "gv_cgal_tools_root": str(paths.gv_cgal_tools_root),
            "gv_cgal_tools_env_script": str(paths.gv_cgal_tools_env_script),
            "scale_space_binary": str(paths.scale_space_binary),
        },
        "checks": checks,
        "with_mirheo": with_mirheo,
        "with_gv_runtime": with_gv_runtime,
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
        "--with-gv-runtime",
        action="store_true",
        default=False,
        help="Include GV runtime hardening checks (Mirheo artifacts, scale_space, OpenMPI lib path, MDAnalysis).",
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
        with_gv_runtime=args.with_gv_runtime,
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
