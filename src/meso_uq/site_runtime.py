"""Site-neutral runtime path helpers for HPC platform bootstrap state."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

VALID_RUNTIME_SITES = {"vega", "karolina"}
DEFAULT_SITE_ROOT_NAMES = {
    "vega": "_vega",
    "karolina": "_karolina",
}


@dataclass(frozen=True)
class RuntimePaths:
    site: str
    repo_root: Path
    src_root: Path
    site_root: Path
    logs_dir: Path
    korali_source: Path
    korali_root: Path
    korali_build_dir: Path
    korali_prefix: Path
    korali_site_packages: Path
    korali_env_script: Path
    gv_venv_root: Path
    gv_venv_site_packages: Path
    gv_venv_env_script: Path
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
    return repo_root / DEFAULT_SITE_ROOT_NAMES[site]


def get_site_runtime_paths(
    repo_root: str | Path | None = None,
    *,
    site: str | None = None,
    runtime_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> RuntimePaths:
    source_env = env if env is not None else os.environ
    resolved_site = normalize_runtime_site(site or source_env.get("MESOUQ_SITE") or source_env.get("HPC_SITE") or "vega")
    root = resolve_repo_root(repo_root)
    site_root = _resolve_site_root(root, site=resolved_site, runtime_root=runtime_root, env=source_env)
    korali_root = site_root / "korali"
    mirheo_root = site_root / "mirheo"
    venv_root = site_root / "venv"
    tinytex_root = site_root / "tinytex"
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return RuntimePaths(
        site=resolved_site,
        repo_root=root,
        src_root=root / "src",
        site_root=site_root,
        logs_dir=site_root / "logs",
        korali_source=root / "extern" / "korali",
        korali_root=korali_root,
        korali_build_dir=korali_root / "build",
        korali_prefix=korali_root / "install",
        korali_site_packages=korali_root / "install" / "lib" / py_tag / "site-packages",
        korali_env_script=korali_root / "env.sh",
        gv_venv_root=site_root / "gv_venv",
        gv_venv_site_packages=site_root / "gv_venv" / "lib" / py_tag / "site-packages",
        gv_venv_env_script=site_root / "gv_venv" / "env.sh",
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

