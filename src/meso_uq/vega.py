"""Helpers for repo-local Vega bootstrap and runtime paths."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VEGA_MODULES = (
    "Python/3.10.8-GCCcore-12.2.0",
    "openmpi/4.1.2.1",
    "CUDA/12.2.2",
    "GSL/2.7-GCC-12.2.0",
    "Eigen/3.4.0-GCCcore-12.2.0",
)
DEFAULT_VEGA_MIRHEO_MODULES = DEFAULT_VEGA_MODULES + (
    "CMake/3.24.3-GCCcore-12.2.0",
    "HDF5/1.14.0-gompi-2022b",
)
DEFAULT_MIRHEO_SOURCE_PATH = "/ceph/hpc/home/eubrieucb/software/Mirheo"
MIRHEO_LOCK_FILENAME = "mirheo.lock.json"
MIRHEO_TREE_HASH_IGNORE = {
    ".git",
    "build",
    "__pycache__",
    "Mirheo.egg-info",
}


@dataclass(frozen=True)
class VegaPaths:
    repo_root: Path
    src_root: Path
    vega_root: Path
    logs_dir: Path
    korali_source: Path
    korali_root: Path
    korali_build_dir: Path
    korali_prefix: Path
    korali_site_packages: Path
    korali_env_script: Path
    venv_root: Path
    venv_site_packages: Path
    tinytex_root: Path
    tinytex_bin_dir: Path
    tinytex_env_script: Path
    mirheo_source_lock: Path
    mirheo_root: Path
    mirheo_build_dir: Path
    mirheo_prefix: Path
    mirheo_package_dir: Path
    mirheo_env_script: Path
    mirheo_snapshot_path: Path


def resolve_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start).expanduser().resolve() if start is not None else Path(__file__).resolve()
    search_root = current if current.is_dir() else current.parent
    for candidate in (search_root, *search_root.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "extern" / "korali").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the MesoUQ repo root from the provided path.")


def get_vega_paths(repo_root: str | Path | None = None) -> VegaPaths:
    root = resolve_repo_root(repo_root)
    vega_root = root / "_vega"
    korali_root = vega_root / "korali"
    mirheo_root = vega_root / "mirheo"
    venv_root = vega_root / "venv"
    tinytex_root = vega_root / "tinytex"
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return VegaPaths(
        repo_root=root,
        src_root=root / "src",
        vega_root=vega_root,
        logs_dir=vega_root / "logs",
        korali_source=root / "extern" / "korali",
        korali_root=korali_root,
        korali_build_dir=korali_root / "build",
        korali_prefix=korali_root / "install",
        korali_site_packages=korali_root / "install" / "lib" / py_tag / "site-packages",
        korali_env_script=korali_root / "env.sh",
        venv_root=venv_root,
        venv_site_packages=venv_root / "lib" / py_tag / "site-packages",
        tinytex_root=tinytex_root,
        tinytex_bin_dir=tinytex_root / "bin" / "x86_64-linux",
        tinytex_env_script=tinytex_root / "env.sh",
        mirheo_source_lock=root / "extern" / MIRHEO_LOCK_FILENAME,
        mirheo_root=mirheo_root,
        mirheo_build_dir=mirheo_root / "build",
        mirheo_prefix=mirheo_root / "install",
        mirheo_package_dir=mirheo_root / "package",
        mirheo_env_script=mirheo_root / "env.sh",
        mirheo_snapshot_path=mirheo_root / "source_snapshot.json",
    )


def split_pythonpath(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(entry).expanduser() for entry in value.split(os.pathsep) if entry]


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


def render_korali_env_script(paths: VegaPaths) -> str:
    pythonpath = build_runtime_pythonpath(paths.repo_root, paths.korali_site_packages)
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by scripts/platforms/vega/bootstrap_korali.sh.",
        "# This intentionally replaces inherited PYTHONPATH entries so repo-local",
        "# Korali wins over any preexisting user-global installation.",
        f"export MESOUQ_REPO_ROOT={shlex.quote(str(paths.repo_root))}",
        f"export MESOUQ_VEGA_ROOT={shlex.quote(str(paths.vega_root))}",
        f"export KORALI_PREFIX={shlex.quote(str(paths.korali_prefix))}",
        f"export KORALI_PYTHONPATH={shlex.quote(str(paths.korali_site_packages))}",
        f"export PYTHONPATH={shlex.quote(pythonpath)}",
    ]
    return "\n".join(lines) + "\n"


def load_mirheo_source_lock(repo_root: str | Path | None = None) -> dict[str, object]:
    paths = get_vega_paths(repo_root)
    if not paths.mirheo_source_lock.is_file():
        return {"source_path": DEFAULT_MIRHEO_SOURCE_PATH}
    payload = json.loads(paths.mirheo_source_lock.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Mirheo source lock payload: {paths.mirheo_source_lock}")
    source_path = payload.get("source_path") or DEFAULT_MIRHEO_SOURCE_PATH
    payload["source_path"] = source_path
    return payload


def resolve_mirheo_source(
    repo_root: str | Path | None = None,
    *,
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    env_override = os.environ.get("MESOUQ_MIRHEO_SRC", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    lock = load_mirheo_source_lock(repo_root)
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
        "# Generated by scripts/platforms/vega/bootstrap_mirheo.sh.",
        f"export MESOUQ_REPO_ROOT={shlex.quote(str(paths.repo_root))}",
        f"export MESOUQ_VEGA_ROOT={shlex.quote(str(paths.vega_root))}",
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
        "# Generated by scripts/platforms/vega/bootstrap_tex.sh.",
        f"export MESOUQ_REPO_ROOT={shlex.quote(str(paths.repo_root))}",
        f"export MESOUQ_VEGA_ROOT={shlex.quote(str(paths.vega_root))}",
        f"export MESOUQ_TINYTEX_ROOT={shlex.quote(str(paths.tinytex_root))}",
        f"export PATH={shlex.quote(str(paths.tinytex_bin_dir))}:$PATH",
    ]
    return "\n".join(lines) + "\n"
