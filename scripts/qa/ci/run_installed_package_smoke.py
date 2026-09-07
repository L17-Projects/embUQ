#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import site
import subprocess
import sys
import tempfile
import textwrap
import venv
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
HEAVY_OPTIONAL_MODULES = ("torch", "pyro", "matplotlib", "mpi4py", "mirheo", "korali", "slurm")
SMOKE_MODULES = (
    "meso_uq",
    "meso_uq.core",
    "meso_uq.public_api",
    "meso_uq.agents",
    "meso_uq.agents.registry",
    "meso_uq.modalities",
    "meso_uq.modalities.registry",
    "meso_uq.structures",
    "meso_uq.structures.registry",
    "meso_uq.surrogates",
    "meso_uq.surrogates.contracts",
    "meso_uq.surrogates.registry",
    "meso_uq.surrogate.compat",
    "meso_uq.surrogate.catalogs",
    "meso_uq.surrogate.emb_catalog",
    "meso_uq.noise",
    "meso_uq.noise.contracts",
    "meso_uq.noise.registry",
    "meso_uq.config",
    "meso_uq.config.loader",
    "meso_uq.config.aliases",
    "meso_uq.config.examples",
    "meso_uq.config.models",
    "meso_uq.configs.policy",
    "meso_uq.artifacts",
    "meso_uq.artifacts.policy",
    "meso_uq.artifacts.relocation",
    "meso_uq.platforms",
    "meso_uq.platforms.policy",
    "meso_uq.platforms.korali_runtime",
    "meso_uq.experiments",
    "meso_uq.inference.contracts",
    "meso_uq.orchestration",
    "meso_uq.orchestration.lineage",
    "meso_uq.plotting",
    "meso_uq.plotting.contracts",
    "meso_uq.reporting",
    "meso_uq.reporting.contracts",
    "meso_uq.site_runtime",
    "meso_uq.workflows.legacy",
)
PACKAGE_DATA_FILES: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class InstalledPackageSmokeResult:
    wheel_path: Path
    venv_root: Path
    workdir: Path
    command: tuple[str, ...]
    imported_modules: tuple[str, ...]


def _resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or base is None:
        return path.resolve()
    return (base / path).resolve()


def _pythonpath_entries(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(entry for entry in value.split(os.pathsep) if entry)


def _looks_like_repo_path(entry: str, repo_root: Path, src_root: Path) -> bool:
    try:
        resolved = Path(entry).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return resolved == repo_root or resolved == src_root or repo_root in resolved.parents or src_root in resolved.parents


def _looks_like_site_packages_path(entry: str | Path) -> bool:
    try:
        resolved = Path(entry).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return any(part in {"site-packages", "dist-packages"} for part in resolved.parts)


def _filter_repo_paths(entries: Iterable[str], repo_root: Path, src_root: Path) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for entry in entries:
        if _looks_like_repo_path(entry, repo_root, src_root):
            removed.append(entry)
        else:
            kept.append(entry)
    return kept, removed


def _wheel_candidates(wheel_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted((path for path in wheel_dir.glob("*.whl") if path.is_file()), key=lambda item: item.name))


def _resolve_wheel_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        wheels = _wheel_candidates(candidate)
        if not wheels:
            raise FileNotFoundError(f"No wheel files found in {candidate}")
        return wheels[-1].resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Wheel file not found: {candidate}")
    if candidate.suffix != ".whl":
        raise ValueError(f"Expected a wheel file, got {candidate}")
    return candidate.resolve()


def _run_logged_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _build_wheel(repo_root: Path, wheelhouse: Path, python_bin: str) -> Path:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    _run_logged_command(
        [
            python_bin,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
            str(repo_root),
        ],
        cwd=repo_root,
    )
    wheels = _wheel_candidates(wheelhouse)
    if not wheels:
        raise RuntimeError(f"Wheel build completed without producing a wheel in {wheelhouse}")
    return wheels[-1]


def _venv_python(venv_root: Path) -> Path:
    if sys.platform == "win32":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _create_venv(venv_root: Path, _python_bin: str) -> Path:
    # Install MesoUQ non-editably while reusing the dependency environment that
    # CI/local validation already prepared. The probe removes repo-local
    # sys.path entries before import, so an editable parent install cannot
    # satisfy the package-under-test imports.
    builder = venv.EnvBuilder(with_pip=True, clear=True, system_site_packages=True)
    builder.create(venv_root)
    python_path = _venv_python(venv_root)
    if not python_path.exists():
        raise FileNotFoundError(f"Virtualenv python not found: {python_path}")
    return python_path


def _dependency_path_entries(repo_root: Path) -> tuple[str, ...]:
    src_root = repo_root / "src"
    site_paths = []
    with contextlib.suppress(Exception):
        site_paths.extend(site.getsitepackages())
    with contextlib.suppress(Exception):
        site_paths.append(site.getusersitepackages())
    kept = [
        entry
        for entry in site_paths
        if entry and (not _looks_like_repo_path(entry, repo_root, src_root) or _looks_like_site_packages_path(entry))
    ]
    return tuple(
        str(Path(entry).expanduser().resolve())
        for entry in kept
        if entry and Path(entry).expanduser().exists()
    )


def _install_wheel(python_bin: Path, wheel_path: Path) -> None:
    _run_logged_command(
        [str(python_bin), "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel_path)],
        cwd=wheel_path.parent,
    )


def _render_smoke_probe(
    *,
    repo_root: Path,
    modules: Sequence[str] = SMOKE_MODULES,
    heavy_modules: Sequence[str] = HEAVY_OPTIONAL_MODULES,
    package_data_files: Sequence[tuple[str, str]] = PACKAGE_DATA_FILES,
    dependency_paths: Sequence[str] = (),
) -> str:
    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import importlib
        import importlib.resources
        import json
        import pathlib
        import sys

        REPO_ROOT = pathlib.Path(r"{repo_root}").resolve()
        SRC_ROOT = (REPO_ROOT / "src").resolve()
        MODULES = {tuple(modules)!r}
        HEAVY_MODULES = {tuple(heavy_modules)!r}
        PACKAGE_DATA_FILES = {tuple(package_data_files)!r}
        DEPENDENCY_PATHS = {tuple(dependency_paths)!r}

        def _is_repo_path(entry: str) -> bool:
            try:
                resolved = pathlib.Path(entry).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                return False
            return resolved == REPO_ROOT or resolved == SRC_ROOT or REPO_ROOT in resolved.parents or SRC_ROOT in resolved.parents

        def _is_site_packages_path(entry: str) -> bool:
            try:
                resolved = pathlib.Path(entry).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                return False
            return any(part in {{"site-packages", "dist-packages"}} for part in resolved.parts)

        def _origin_path(module_name: str) -> pathlib.Path | None:
            module = sys.modules[module_name]
            origin = getattr(module, "__file__", None)
            if origin is None:
                spec = getattr(module, "__spec__", None)
                origin = getattr(spec, "origin", None) if spec is not None else None
            if origin is None:
                return None
            return pathlib.Path(origin).resolve()

        def _is_under(entry: str, root: pathlib.Path) -> bool:
            try:
                resolved = pathlib.Path(entry).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                return False
            return resolved == root or root in resolved.parents

        imported = []
        leaked_origins = []
        initial_sys_path = list(sys.path)
        removed_sys_path = [entry for entry in initial_sys_path if entry and _is_repo_path(entry)]
        clean_sys_path = [entry for entry in initial_sys_path if not entry or not _is_repo_path(entry)]
        venv_prefix = pathlib.Path(sys.prefix).resolve()
        venv_sys_path = [entry for entry in clean_sys_path if entry and _is_under(entry, venv_prefix)]
        other_sys_path = [entry for entry in clean_sys_path if entry not in venv_sys_path]
        sys.path[:] = venv_sys_path
        appended_dependency_paths = []
        for entry in DEPENDENCY_PATHS:
            if not entry or entry in sys.path:
                continue
            if _is_repo_path(entry) and not _is_site_packages_path(entry):
                continue
            sys.path.append(entry)
            appended_dependency_paths.append(entry)
        for entry in other_sys_path:
            if entry not in sys.path:
                sys.path.append(entry)

        try:
            for module_name in MODULES:
                importlib.import_module(module_name)
                imported.append(module_name)
                origin = _origin_path(module_name)
                if origin is not None and (origin == REPO_ROOT or origin == SRC_ROOT or REPO_ROOT in origin.parents or SRC_ROOT in origin.parents):
                    leaked_origins.append(f"{{module_name}} -> {{origin}}")

            loaded_heavy = [name for name in HEAVY_MODULES if name in sys.modules]
            if loaded_heavy:
                raise RuntimeError("lightweight smoke loaded optional heavy modules: " + ", ".join(loaded_heavy))
            if leaked_origins:
                raise RuntimeError("repository-root import leakage detected: " + "; ".join(leaked_origins))
            missing_data = []
            for package_name, relative_path in PACKAGE_DATA_FILES:
                resource = importlib.resources.files(package_name).joinpath(relative_path)
                if not resource.is_file():
                    missing_data.append(f"{{package_name}}/{{relative_path}}")
            if missing_data:
                raise RuntimeError("installed package missing data files: " + ", ".join(missing_data))
        except Exception as exc:
            raise RuntimeError(
                "installed-package smoke failed; pruned repo-root sys.path entries: "
                + ", ".join(removed_sys_path or ["<none>"])
            ) from exc

        print(json.dumps({{"status": "passed", "imported": imported, "pruned_sys_path": removed_sys_path, "dependency_paths": appended_dependency_paths}}, sort_keys=True))
        """
    ).strip()


def _run_smoke_probe(
    python_bin: Path,
    *,
    repo_root: Path,
    workdir: Path,
    modules: Sequence[str] = SMOKE_MODULES,
    heavy_modules: Sequence[str] = HEAVY_OPTIONAL_MODULES,
) -> subprocess.CompletedProcess[str]:
    probe = _render_smoke_probe(
        repo_root=repo_root,
        modules=modules,
        heavy_modules=heavy_modules,
        package_data_files=PACKAGE_DATA_FILES,
        dependency_paths=_dependency_path_entries(repo_root),
    )
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}}
    env["PYTHONNOUSERSITE"] = "1"
    return _run_logged_command([str(python_bin), "-I", "-c", probe], cwd=workdir, env=env)


def run_installed_package_smoke(
    *,
    repo_root: Path = REPO_ROOT,
    python_bin: str = sys.executable,
    wheel: str | Path | None = None,
    modules: Sequence[str] = SMOKE_MODULES,
    heavy_modules: Sequence[str] = HEAVY_OPTIONAL_MODULES,
) -> InstalledPackageSmokeResult:
    repo_root = _resolve_path(repo_root)

    with tempfile.TemporaryDirectory(prefix="mesouq-installed-package-smoke-") as temp_root:
        temp_root_path = Path(temp_root)
        wheel_path = _resolve_wheel_path(wheel) if wheel is not None else _build_wheel(
            repo_root,
            temp_root_path / "wheelhouse",
            python_bin,
        )

        venv_root = temp_root_path / "venv"
        workdir = temp_root_path / "workdir"
        venv_python = _create_venv(venv_root, python_bin)
        workdir.mkdir(parents=True, exist_ok=True)

        _install_wheel(venv_python, wheel_path)
        result = _run_smoke_probe(
            venv_python,
            repo_root=repo_root,
            workdir=workdir,
            modules=modules,
            heavy_modules=heavy_modules,
        )
        imported = tuple(json.loads(result.stdout)["imported"])
        return InstalledPackageSmokeResult(
            wheel_path=wheel_path,
            venv_root=venv_root,
            workdir=workdir,
            command=tuple([str(venv_python), "-I", "-c", _render_smoke_probe(repo_root=repo_root, modules=modules, heavy_modules=heavy_modules)]),
            imported_modules=imported,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or install the package into an isolated environment and smoke installed imports.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--wheel", default=None, help="Use an existing wheel file or a directory containing a wheel.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = run_installed_package_smoke(
        repo_root=_resolve_path(args.repo_root),
        python_bin=args.python_bin,
        wheel=args.wheel,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "wheel": str(result.wheel_path),
                "venv_root": str(result.venv_root),
                "workdir": str(result.workdir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
