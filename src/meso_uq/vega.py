"""Backward-compatible Vega helpers backed by site-neutral runtime paths."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path

from meso_uq.platforms.site_selector import resolve_hpc_site
from meso_uq.site_runtime import RuntimePaths as VegaPaths
from meso_uq.site_runtime import get_site_runtime_paths, resolve_repo_root

DEFAULT_VEGA_MODULES = (
    "Python/3.10.8-GCCcore-12.2.0",
    "OpenMPI/4.1.4-GCC-12.2.0",
    "CUDA/12.2.2",
    "GSL/2.7-GCC-12.2.0",
    "Eigen/3.4.0-GCCcore-12.2.0",
)
DEFAULT_VEGA_RUNTIME_MODULES = DEFAULT_VEGA_MODULES + (
    "MPFR/4.2.0-GCCcore-12.2.0",
    "GMP/6.2.1-GCCcore-12.2.0",
)
DEFAULT_VEGA_MIRHEO_MODULES = DEFAULT_VEGA_MODULES + (
    "CMake/3.24.3-GCCcore-12.2.0",
    "HDF5/1.14.0-gompi-2022b",
)
DEFAULT_VEGA_GV_CGAL_MODULES = DEFAULT_VEGA_MIRHEO_MODULES + (
    "MPFR/4.2.0-GCCcore-12.2.0",
    "GMP/6.2.1-GCCcore-12.2.0",
    "CGAL/5.6-GCCcore-12.3.0",
    "Boost/1.82.0-GCC-12.3.0",
)
DEFAULT_KAROLINA_MODULES = (
    "CUDA/12.4.0",
    "HDF5/1.14.0-gompi-2022b",
    "GSL/2.7-GCC-12.3.0",
    "Eigen/3.4.0-GCCcore-12.2.0",
    "Python/3.10.8-GCCcore-12.2.0",
)
DEFAULT_KAROLINA_RUNTIME_MODULES = DEFAULT_KAROLINA_MODULES + (
    "MPFR/4.2.0-GCCcore-12.2.0",
    "GMP/6.2.1-GCCcore-12.2.0",
)
DEFAULT_KAROLINA_MIRHEO_MODULES = DEFAULT_KAROLINA_MODULES + (
    "CMake/3.24.3-GCCcore-12.2.0",
)
DEFAULT_KAROLINA_GV_CGAL_MODULES = DEFAULT_KAROLINA_MIRHEO_MODULES + (
    "MPFR/4.2.0-GCCcore-12.2.0",
    "GMP/6.2.1-GCCcore-12.2.0",
    "CGAL/5.6-GCCcore-12.3.0",
    "Boost/1.82.0-GCC-12.3.0",
)
DEFAULT_GV_CGAL_LIBRARY_PATHS = {
    "karolina": (
        "/apps/all/MPFR/4.2.0-GCCcore-12.2.0/lib",
        "/apps/all/GMP/6.2.1-GCCcore-12.2.0/lib",
        "/apps/all/GCCcore/12.2.0/lib64",
    )
}
DEFAULT_MIRHEO_SOURCE_PATH = "/ceph/hpc/home/eubrieucb/software/Mirheo"
DEFAULT_MIRHEO_SOURCE_PATHS = {
    "vega": DEFAULT_MIRHEO_SOURCE_PATH,
    "karolina": "/home/it4i-bbenvegnen/software/Mirheo",
}
HDF5_RUNTIME_ROOT_ENV_VARS = ("MESOUQ_HDF5_ROOT", "EBROOTHDF5", "HDF5_DIR")
CAPTURED_RUNTIME_PATH_VARS = (
    "PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "PKG_CONFIG_PATH",
    "CMAKE_PREFIX_PATH",
)
CAPTURED_RUNTIME_SCALAR_VARS = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "EBROOTCUDA",
    "EBROOTOPENMPI",
    "EBROOTHDF5",
    "EBROOTGSL",
    "EBROOTEIGEN",
    "HDF5_DIR",
)
UNIFIED_ENV_EXTRAS = ("dev", "bnn", "plot", "mpi", "gv", "hpc-build", "hpc")
MIRHEO_LOCK_FILENAME = "mirheo.lock.json"
MIRHEO_TREE_HASH_IGNORE = {
    ".git",
    "build",
    "__pycache__",
    "Mirheo.egg-info",
}


def get_vega_paths(
    repo_root: str | Path | None = None,
    *,
    runtime_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> VegaPaths:
    return get_site_runtime_paths(repo_root, site="vega", runtime_root=runtime_root, env=env)


def get_runtime_paths(
    repo_root: str | Path | None = None,
    *,
    site: str | None = None,
    runtime_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> VegaPaths:
    return get_site_runtime_paths(repo_root, site=site, runtime_root=runtime_root, env=env)


def split_pythonpath(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(entry).expanduser() for entry in value.split(os.pathsep) if entry]


def _normalize_runtime_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if path.name == "lib":
        path = path.parent
    return path.resolve(strict=False)


def _dedupe_runtime_roots(roots: Sequence[str | Path]) -> tuple[Path, ...]:
    normalized: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not str(root):
            continue
        path = _normalize_runtime_root(root)
        text = str(path)
        if text in seen:
            continue
        normalized.append(path)
        seen.add(text)
    return tuple(normalized)


def discover_hdf5_runtime_roots(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    source = os.environ if env is None else env
    return _dedupe_runtime_roots(
        tuple(source[name] for name in HDF5_RUNTIME_ROOT_ENV_VARS if source.get(name))
    )


def _dedupe_library_dirs(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    normalized: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not str(path):
            continue
        resolved = Path(path).expanduser().resolve(strict=False)
        text = str(resolved)
        if text in seen:
            continue
        normalized.append(resolved)
        seen.add(text)
    return tuple(normalized)


def _python_library_dirs_from_root(root: str | Path) -> tuple[Path, ...]:
    path = Path(root).expanduser().resolve(strict=False)
    candidates = [path]
    if path.name not in {"lib", "lib64"}:
        candidates.extend([path / "lib", path / "lib64"])

    lib_dirs: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        has_shared_libpython = any(candidate.glob("libpython*.so*")) or any(
            candidate.glob("libpython*.dylib")
        )
        if has_shared_libpython:
            lib_dirs.append(candidate)
    return tuple(lib_dirs)


def discover_python_runtime_library_dirs(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Return library directories required by the current Python executable.

    HPC module Python builds commonly link the venv Python binary to
    ``libpython`` outside the venv. The canonical env script records that
    directory so a clean login shell can start ``${MESOUQ_ENV_ROOT}/bin/python``.
    """

    source = os.environ if env is None else env
    lib_dirs: list[Path] = []
    for entry in source.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry).expanduser().resolve(strict=False)
        if candidate.is_dir():
            lib_dirs.append(candidate)

    candidates: list[str | Path] = []
    candidates.extend(
        source[name]
        for name in ("MESOUQ_PYTHON_LIB_DIR", "EBROOTPYTHON")
        if source.get(name)
    )
    candidates.extend(
        value
        for value in (sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LIBPL"))
        if value
    )
    candidates.extend(path for path in (sys.base_prefix, sys.exec_prefix, sys.prefix) if path)

    for candidate in candidates:
        lib_dirs.extend(_python_library_dirs_from_root(candidate))
    return _dedupe_library_dirs(lib_dirs)


def _split_env_path_entries(value: str | None) -> tuple[Path, ...]:
    if not value:
        return ()
    return _dedupe_library_dirs(entry for entry in value.split(os.pathsep) if entry)


def _render_captured_path_var_lines(var_name: str, entries: Sequence[Path]) -> list[str]:
    if not entries:
        return []
    array_name = f"_mesouq_{var_name.lower()}_entries"
    lines = [f"{array_name}=("]
    lines.extend(f"  {shlex.quote(str(entry))}" for entry in reversed(tuple(entries)))
    lines.extend(
        [
            ")",
            f'for _mesouq_path_entry in "${{{array_name}[@]}}"; do',
            '  if [[ ! -d "${_mesouq_path_entry}" ]]; then',
            "    continue",
            "  fi",
            f'  case ":${{{var_name}:-}}:" in',
            '    *":${_mesouq_path_entry}:"*) ;;',
            f'    *) export {var_name}="${{_mesouq_path_entry}}${{{var_name}:+:${{{var_name}}}}}" ;;',
            "  esac",
            "done",
            f"unset {array_name}",
        ]
    )
    return lines


def _render_captured_runtime_env_lines(env: Mapping[str, str]) -> list[str]:
    lines: list[str] = []
    for var_name in CAPTURED_RUNTIME_PATH_VARS:
        lines.extend(_render_captured_path_var_lines(var_name, _split_env_path_entries(env.get(var_name))))
    for var_name in CAPTURED_RUNTIME_SCALAR_VARS:
        value = env.get(var_name, "").strip()
        if not value:
            continue
        lines.extend(
            [
                f'if [[ -z "${{{var_name}:-}}" ]]; then',
                f"  export {var_name}={shlex.quote(value)}",
                "fi",
            ]
        )
    if lines:
        lines.append("unset _mesouq_path_entry")
    return lines


def _has_korali_package(path: Path) -> bool:
    package_dir = path / "korali"
    return package_dir.is_dir() or (package_dir / "__init__.py").is_file()


def find_external_korali_entries(
    current_pythonpath: str | None,
    repo_root: str | Path,
    korali_site_packages: str | Path,
) -> list[str]:
    root = Path(repo_root).expanduser().resolve()
    src_root = (root / "src").resolve()
    local_korali = Path(korali_site_packages).expanduser().resolve()
    allowed = {root, src_root, local_korali}
    external: list[str] = []
    for entry in split_pythonpath(current_pythonpath):
        resolved = entry.resolve(strict=False)
        if resolved in allowed:
            continue
        if _has_korali_package(resolved):
            external.append(str(resolved))
    return external


def build_runtime_pythonpath(
    repo_root: str | Path,
    korali_site_packages: str | Path,
    current_pythonpath: str | None = None,
    *,
    include_existing: bool = False,
) -> str:
    root = Path(repo_root).expanduser().resolve()
    src_root = (root / "src").resolve()
    local_korali = Path(korali_site_packages).expanduser().resolve()
    ordered = [local_korali, src_root, root]
    seen = {str(path) for path in ordered}

    if include_existing:
        for entry in split_pythonpath(current_pythonpath):
            resolved = entry.resolve(strict=False)
            text = str(resolved)
            if text in seen:
                continue
            if _has_korali_package(resolved):
                continue
            ordered.append(resolved)
            seen.add(text)

    return os.pathsep.join(str(path) for path in ordered)

def build_unified_env_pythonpath(
    repo_root: str | Path,
    env_site_packages: str | Path,
    korali_site_packages: str | Path,
    current_pythonpath: str | None = None,
    *,
    include_existing: bool = False,
) -> str:
    root = Path(repo_root).expanduser().resolve()
    src_root = (root / "src").resolve()
    env_site = Path(env_site_packages).expanduser().resolve()
    local_korali = Path(korali_site_packages).expanduser().resolve()
    ordered = [env_site, local_korali, src_root, root]
    deduped: list[Path] = []
    seen: set[str] = set()
    for entry in ordered:
        text = str(entry)
        if text in seen:
            continue
        deduped.append(entry)
        seen.add(text)

    if include_existing:
        for entry in split_pythonpath(current_pythonpath):
            resolved = entry.resolve(strict=False)
            text = str(resolved)
            if text in seen:
                continue
            if _has_korali_package(resolved):
                continue
            deduped.append(resolved)
            seen.add(text)

    return os.pathsep.join(str(path) for path in deduped)


def render_korali_env_script(paths: VegaPaths) -> str:
    pythonpath = build_runtime_pythonpath(paths.repo_root, paths.korali_site_packages)
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/platforms/{paths.site}/bootstrap_korali.sh.",
        "# This intentionally replaces inherited PYTHONPATH entries so repo-local",
        "# Korali wins over any preexisting user-global installation.",
        f"export MESOUQ_SITE={shlex.quote(paths.site)}",
        f"export MESOUQ_REPO_ROOT={shlex.quote(str(paths.repo_root))}",
        f"export MESOUQ_SITE_RUNTIME_ROOT={shlex.quote(str(paths.site_root))}",
        f"export MESOUQ_PROVENANCE_ROOT={shlex.quote(str(paths.provenance_root))}",
        f"export KORALI_PREFIX={shlex.quote(str(paths.korali_prefix))}",
        f"export KORALI_PYTHONPATH={shlex.quote(str(paths.korali_site_packages))}",
        f"export PYTHONPATH={shlex.quote(pythonpath)}",
    ]
    return "\n".join(lines) + "\n"

def render_unified_env_script(
    paths: VegaPaths,
    *,
    source_root: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    hdf5_runtime_roots: Sequence[str | Path] | None = None,
    python_runtime_lib_dirs: Sequence[str | Path] | None = None,
    captured_env: Mapping[str, str] | None = None,
) -> str:
    pythonpath = build_unified_env_pythonpath(
        paths.repo_root,
        env_site_packages=paths.env_site_packages,
        korali_site_packages=paths.korali_site_packages,
    )
    source_text = ""
    if source_root is not None:
        source_text = str(Path(source_root).expanduser().resolve())
    snapshot_text = ""
    if snapshot_path is not None:
        snapshot_text = str(Path(snapshot_path).expanduser().resolve())
    if hdf5_runtime_roots is None:
        hdf5_runtime_roots = discover_hdf5_runtime_roots()
    normalized_hdf5_roots = _dedupe_runtime_roots(hdf5_runtime_roots)
    if python_runtime_lib_dirs is None:
        python_runtime_lib_dirs = discover_python_runtime_library_dirs()
    normalized_python_lib_dirs = _dedupe_library_dirs(python_runtime_lib_dirs)
    runtime_env_lines = _render_captured_runtime_env_lines(os.environ if captured_env is None else captured_env)

    runtime_library_lines = [
        "_mesouq_runtime_library_roots=(",
        '  "${MESOUQ_HDF5_ROOT:-}"',
        '  "${EBROOTHDF5:-}"',
        '  "${HDF5_DIR:-}"',
    ]
    runtime_library_lines.extend(f"  {shlex.quote(str(root))}" for root in normalized_hdf5_roots)
    runtime_library_lines.extend(
        [
            ")",
            'for _mesouq_runtime_root in "${_mesouq_runtime_library_roots[@]}"; do',
            '  if [[ -z "${_mesouq_runtime_root}" ]]; then',
            "    continue",
            '  elif [[ -d "${_mesouq_runtime_root}/lib" ]]; then',
            '    _mesouq_runtime_lib="${_mesouq_runtime_root}/lib"',
            '  elif [[ "${_mesouq_runtime_root##*/}" == "lib" && -d "${_mesouq_runtime_root}" ]]; then',
            '    _mesouq_runtime_lib="${_mesouq_runtime_root}"',
            "  else",
            "    continue",
            "  fi",
            '  case ":${LD_LIBRARY_PATH:-}:" in',
            '    *":${_mesouq_runtime_lib}:"*) ;;',
            '    *) export LD_LIBRARY_PATH="${_mesouq_runtime_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;',
            "  esac",
            "done",
            "_mesouq_runtime_library_dirs=(",
        ]
    )
    runtime_library_lines.extend(f"  {shlex.quote(str(path))}" for path in normalized_python_lib_dirs)
    runtime_library_lines.extend(
        [
            ")",
            'for _mesouq_runtime_lib in "${_mesouq_runtime_library_dirs[@]}"; do',
            '  if [[ -z "${_mesouq_runtime_lib}" ]]; then',
            "    continue",
            '  elif [[ ! -d "${_mesouq_runtime_lib}" ]]; then',
            "    continue",
            "  fi",
            '  case ":${LD_LIBRARY_PATH:-}:" in',
            '    *":${_mesouq_runtime_lib}:"*) ;;',
            '    *) export LD_LIBRARY_PATH="${_mesouq_runtime_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;',
            "  esac",
            "done",
            "unset _mesouq_runtime_root _mesouq_runtime_lib _mesouq_runtime_library_roots _mesouq_runtime_library_dirs",
        ]
    )

    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/platforms/{paths.site}/bootstrap_env.sh.",
        f"export MESOUQ_SITE={shlex.quote(paths.site)}",
        f"export MESOUQ_REPO_ROOT={shlex.quote(str(paths.repo_root))}",
        f"export MESOUQ_SITE_RUNTIME_ROOT={shlex.quote(str(paths.site_root))}",
        f"export MESOUQ_PROVENANCE_ROOT={shlex.quote(str(paths.provenance_root))}",
        f"export MESOUQ_ENV_ROOT={shlex.quote(str(paths.env_root))}",
        f"export MESOUQ_ENV_SCRIPT={shlex.quote(str(paths.env_script))}",
        f"export VIRTUAL_ENV={shlex.quote(str(paths.env_root))}",
        *runtime_env_lines,
        'export PATH="${MESOUQ_ENV_ROOT}/bin${PATH:+:${PATH}}"',
        f"export KORALI_PREFIX={shlex.quote(str(paths.korali_prefix))}",
        f"export KORALI_PYTHONPATH={shlex.quote(str(paths.korali_site_packages))}",
        f"export MESOUQ_GV_VENV_ROOT={shlex.quote(str(paths.env_root))}",
        f"export MESOUQ_GV_VENV_SITE_PACKAGES={shlex.quote(str(paths.env_site_packages))}",
        f"export MESOUQ_GV_ENV_SCRIPT={shlex.quote(str(paths.env_script))}",
        f"export PYTHONPATH={shlex.quote(pythonpath)}",
        *runtime_library_lines,
        "_mesouq_native_env_scripts=(",
        f"  {shlex.quote(str(paths.mirheo_env_script))}",
        '  "${MESOUQ_SITE_RUNTIME_ROOT}/mirheoOBMD/env.sh"',
        f"  {shlex.quote(str(paths.gv_cgal_tools_env_script))}",
        ")",
        'for _mesouq_native_env_script in "${_mesouq_native_env_scripts[@]}"; do',
        '  if [[ -f "${_mesouq_native_env_script}" ]]; then',
        '    source "${_mesouq_native_env_script}"',
        "  fi",
        "done",
        "unset _mesouq_native_env_script _mesouq_native_env_scripts",
        f"export VIRTUAL_ENV={shlex.quote(str(paths.env_root))}",
        *runtime_env_lines,
        'export PATH="${MESOUQ_ENV_ROOT}/bin${PATH:+:${PATH}}"',
        f"export PYTHONPATH={shlex.quote(pythonpath)}",
    ]
    if source_text:
        lines.extend(
            [
                f"export MESOUQ_MIRHEO_SRC={shlex.quote(source_text)}",
                f"export MIRHEO_SOURCE_ROOT={shlex.quote(source_text)}",
                f"export MIRHEO_BUILD_DIR={shlex.quote(str(paths.mirheo_build_dir))}",
                f"export MIRHEO_INSTALL_PREFIX={shlex.quote(str(paths.mirheo_prefix))}",
            ]
        )
    if snapshot_text:
        lines.append(f"export MIRHEO_SOURCE_SNAPSHOT={shlex.quote(snapshot_text)}")
    lines.extend(
        [
            'if command -v mpicxx >/dev/null 2>&1; then',
            '  export MESOUQ_OPENMPI_LIB_DIR="$(dirname "$(dirname "$(command -v mpicxx)")")/lib"',
            '  if [[ -d "$MESOUQ_OPENMPI_LIB_DIR" ]]; then',
            '    case ":${LD_LIBRARY_PATH:-}:" in',
            '      *":${MESOUQ_OPENMPI_LIB_DIR}:"*) ;;',
            '      *) export LD_LIBRARY_PATH="${MESOUQ_OPENMPI_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;',
            "    esac",
            "  fi",
            "fi",
        ]
    )
    return "\n".join(lines) + "\n"

def render_gv_cgal_tools_env_script(
    paths: VegaPaths,
    *,
    library_paths: tuple[str, ...] | None = None,
) -> str:
    libs = library_paths if library_paths is not None else DEFAULT_GV_CGAL_LIBRARY_PATHS.get(paths.site, ())
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/platforms/{paths.site}/bootstrap_gv_cgal_tools.sh.",
        f"export MESOUQ_SITE={shlex.quote(paths.site)}",
        f"export MESOUQ_REPO_ROOT={shlex.quote(str(paths.repo_root))}",
        f"export MESOUQ_SITE_RUNTIME_ROOT={shlex.quote(str(paths.site_root))}",
        f"export MESOUQ_PROVENANCE_ROOT={shlex.quote(str(paths.provenance_root))}",
        f"export GV_CGAL_TOOLS_ROOT={shlex.quote(str(paths.gv_cgal_tools_bin_dir))}",
        f"export GV_SCALE_SPACE_BINARY={shlex.quote(str(paths.scale_space_binary))}",
        'case ":${PATH}:" in',
        '  *":${GV_CGAL_TOOLS_ROOT}:"*) ;;',
        '  *) export PATH="${GV_CGAL_TOOLS_ROOT}${PATH:+:${PATH}}" ;;',
        "esac",
    ]
    if libs:
        lines.append("for _mesouq_cgal_lib in \\")
        for index, lib_path in enumerate(libs):
            suffix = " \\" if index < len(libs) - 1 else "; do"
            lines.append(f"  {shlex.quote(lib_path)}{suffix}")
        lines.extend(
            [
                '  if [[ -d "${_mesouq_cgal_lib}" ]]; then',
                '    case ":${LD_LIBRARY_PATH:-}:" in',
                '      *":${_mesouq_cgal_lib}:"*) ;;',
                '      *) export LD_LIBRARY_PATH="${_mesouq_cgal_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;',
                "    esac",
                "  fi",
                "done",
                "unset _mesouq_cgal_lib",
            ]
        )
    return "\n".join(lines) + "\n"


def load_mirheo_source_lock(
    repo_root: str | Path | None = None,
    *,
    site: str | None = None,
) -> dict[str, object]:
    resolved_site = resolve_hpc_site(cli_site=site, default="vega")
    root = resolve_repo_root(repo_root)
    lock_path = root / "extern" / MIRHEO_LOCK_FILENAME
    default_source = DEFAULT_MIRHEO_SOURCE_PATHS.get(resolved_site, DEFAULT_MIRHEO_SOURCE_PATH)
    if not lock_path.is_file():
        return {"source_path": default_source}
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Mirheo source lock payload: {lock_path}")
    locked_source = payload.get("source_path")
    source_path = str(locked_source or default_source)
    if locked_source:
        locked_candidate = Path(str(locked_source)).expanduser()
        default_candidate = Path(default_source).expanduser()
        if not locked_candidate.exists() and default_candidate.exists():
            payload["source_path_fallback_reason"] = (
                f"locked source unavailable for site {resolved_site}: {locked_candidate}"
            )
            source_path = default_source
    payload["source_path"] = source_path
    return payload


def resolve_mirheo_source(
    repo_root: str | Path | None = None,
    *,
    override: str | Path | None = None,
    site: str | None = None,
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    env_override = os.environ.get("MESOUQ_MIRHEO_SRC", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    lock = load_mirheo_source_lock(repo_root, site=site)
    return Path(str(lock["source_path"])).expanduser().resolve()


def _git_output(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _iter_mirheo_source_files(source_root: Path):
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in MIRHEO_TREE_HASH_IGNORE for part in path.parts):
            continue
        yield path


def gather_mirheo_source_snapshot(source_root: str | Path) -> dict[str, object]:
    root = Path(source_root).expanduser().resolve()
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in _iter_mirheo_source_files(root):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        data = path.read_bytes()
        digest.update(data)
        digest.update(b"\0")
        file_count += 1
        total_bytes += len(data)
    git_commit = _git_output(root, "rev-parse", "HEAD")
    git_remote = _git_output(root, "remote", "get-url", "origin")
    return {
        "source_root": str(root),
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "git_commit": git_commit or None,
        "git_remote": git_remote or None,
    }


def render_mirheo_env_script(
    paths: VegaPaths,
    *,
    source_root: str | Path,
    snapshot_path: str | Path | None = None,
) -> str:
    snapshot_text = ""
    if snapshot_path is not None:
        snapshot_text = str(Path(snapshot_path).expanduser().resolve())
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/platforms/{paths.site}/bootstrap_mirheo.sh.",
        f"export MESOUQ_SITE={shlex.quote(paths.site)}",
        f"export MESOUQ_REPO_ROOT={shlex.quote(str(paths.repo_root))}",
        f"export MESOUQ_SITE_RUNTIME_ROOT={shlex.quote(str(paths.site_root))}",
        f"export MESOUQ_PROVENANCE_ROOT={shlex.quote(str(paths.provenance_root))}",
        f"export MESOUQ_MIRHEO_SRC={shlex.quote(str(Path(source_root).expanduser().resolve()))}",
        f"export MIRHEO_SOURCE_ROOT={shlex.quote(str(Path(source_root).expanduser().resolve()))}",
        f"export MIRHEO_BUILD_DIR={shlex.quote(str(paths.mirheo_build_dir))}",
        f"export MIRHEO_INSTALL_PREFIX={shlex.quote(str(paths.mirheo_prefix))}",
    ]
    if snapshot_text:
        lines.append(f"export MIRHEO_SOURCE_SNAPSHOT={shlex.quote(snapshot_text)}")
    return "\n".join(lines) + "\n"


def render_tinytex_env_script(paths: VegaPaths) -> str:
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/platforms/{paths.site}/bootstrap_tex.sh.",
        f"export MESOUQ_SITE={shlex.quote(paths.site)}",
        f"export MESOUQ_REPO_ROOT={shlex.quote(str(paths.repo_root))}",
        f"export MESOUQ_SITE_RUNTIME_ROOT={shlex.quote(str(paths.site_root))}",
        f"export MESOUQ_PROVENANCE_ROOT={shlex.quote(str(paths.provenance_root))}",
        f"export MESOUQ_TINYTEX_ROOT={shlex.quote(str(paths.tinytex_root))}",
        f"export PATH={shlex.quote(str(paths.tinytex_bin_dir))}:$PATH",
    ]
    return "\n".join(lines) + "\n"
