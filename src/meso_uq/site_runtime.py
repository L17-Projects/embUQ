"""Site-neutral runtime path helpers for HPC platform bootstrap state."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from meso_uq.platforms.site_selector import VALID_MESOUQ_SITES, resolve_hpc_site

VALID_RUNTIME_SITES = set(VALID_MESOUQ_SITES)


@dataclass(frozen=True)
class RuntimePaths:
    site: str
    repo_root: Path
    src_root: Path
    site_root: Path
    provenance_root: Path
    logs_dir: Path
    env_root: Path
    env_site_packages: Path
    env_script: Path
    korali_source: Path
    korali_root: Path
    korali_build_dir: Path
    korali_prefix: Path
    korali_site_packages: Path
    korali_env_script: Path
    gv_cgal_tools_root: Path
    gv_cgal_tools_bin_dir: Path
    gv_cgal_tools_env_script: Path
    scale_space_binary: Path
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

    @property
    def vega_root(self) -> Path:
        """Backward-compatible alias for existing Vega helper callers."""
        return self.site_root

    @property
    def karolina_root(self) -> Path:
        return self.site_root


def normalize_runtime_site(site: str | None) -> str:
    resolved = (site or "vega").strip().lower()
    if resolved not in VALID_RUNTIME_SITES:
        raise ValueError(f"Unsupported runtime site '{resolved}'. Expected one of: {sorted(VALID_RUNTIME_SITES)}")
    return resolved


def resolve_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start).expanduser().resolve() if start is not None else Path(__file__).resolve()
    search_root = current if current.is_dir() else current.parent
    for candidate in (search_root, *search_root.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "extern" / "korali").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the MesoUQ repo root from the provided path.")


def _resolve_site_root(
    repo_root: Path,
    *,
    site: str,
    runtime_root: str | Path | None,
    env: dict[str, str],
) -> Path:
    if runtime_root is not None:
        return Path(runtime_root).expanduser().resolve()
    env_root = env.get("MESOUQ_SITE_RUNTIME_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    raise RuntimeError(
        "MESOUQ_SITE_RUNTIME_ROOT is required for MesoUQ site runtime paths. "
        "Set it to a per-site runtime root, or pass runtime_root explicitly."
    )


def _resolve_provenance_root(
    repo_root: Path,
    *,
    site: str,
    site_root: Path,
    env: dict[str, str],
) -> Path:
    env_root = env.get("MESOUQ_PROVENANCE_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if site == "karolina":
        scratch_root = env.get("MESOUQ_SCRATCH_ROOT", "").strip()
        if scratch_root:
            return (Path(scratch_root).expanduser().resolve() / "provenance").resolve()
        return (site_root.parent / "provenance").resolve()
    return (repo_root / "gv").resolve()


def get_site_runtime_paths(
    repo_root: str | Path | None = None,
    *,
    site: str | None = None,
    runtime_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> RuntimePaths:
    source_env = env if env is not None else os.environ
    resolved_site = resolve_hpc_site(cli_site=site, env=source_env, default="vega")
    root = resolve_repo_root(repo_root)
    site_root = _resolve_site_root(root, site=resolved_site, runtime_root=runtime_root, env=source_env)
    provenance_root = _resolve_provenance_root(
        root,
        site=resolved_site,
        site_root=site_root,
        env=source_env,
    )
    korali_root = site_root / "korali"
    mirheo_root = site_root / "mirheo"
    gv_cgal_tools_root = site_root / "gv_cgal_tools"
    env_root = site_root / "env"
    venv_root = site_root / "venv"
    tinytex_root = site_root / "tinytex"
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    env_site_packages = env_root / "lib" / py_tag / "site-packages"
    return RuntimePaths(
        site=resolved_site,
        repo_root=root,
        src_root=root / "src",
        site_root=site_root,
        provenance_root=provenance_root,
        logs_dir=site_root / "logs",
        env_root=env_root,
        env_site_packages=env_site_packages,
        env_script=env_root / "env.sh",
        korali_source=root / "extern" / "korali",
        korali_root=korali_root,
        korali_build_dir=korali_root / "build",
        korali_prefix=korali_root / "install",
        korali_site_packages=korali_root / "install" / "lib" / py_tag / "site-packages",
        korali_env_script=korali_root / "env.sh",
        gv_cgal_tools_root=gv_cgal_tools_root,
        gv_cgal_tools_bin_dir=gv_cgal_tools_root / "bin",
        gv_cgal_tools_env_script=gv_cgal_tools_root / "env.sh",
        scale_space_binary=gv_cgal_tools_root / "bin" / "scale_space",
        venv_root=venv_root,
        venv_site_packages=venv_root / "lib" / py_tag / "site-packages",
        tinytex_root=tinytex_root,
        tinytex_bin_dir=tinytex_root / "bin" / "x86_64-linux",
        tinytex_env_script=tinytex_root / "env.sh",
        mirheo_source_lock=root / "extern" / "mirheo.lock.json",
        mirheo_root=mirheo_root,
        mirheo_build_dir=mirheo_root / "build",
        mirheo_prefix=mirheo_root / "install",
        mirheo_package_dir=mirheo_root / "package",
        mirheo_env_script=mirheo_root / "env.sh",
        mirheo_snapshot_path=mirheo_root / "source_snapshot.json",
    )
