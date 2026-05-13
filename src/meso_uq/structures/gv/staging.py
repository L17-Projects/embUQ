from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import secrets
import shutil
from typing import Mapping, Sequence

from .runtime.catalog import RUNTIME_EXPERIMENTS
from .staging_manifest import RuntimeStagingManifest, StagedTemplateFile, TemplateSpec


class RuntimeStagingError(ValueError):
    """Base class for runtime staging contract errors."""


class MissingTemplateError(RuntimeStagingError):
    """Raised when a requested template file is missing."""


class UnsafeOutputPathError(RuntimeStagingError):
    """Raised when output/run paths violate staging safety rules."""


class UnsupportedStagingTargetError(RuntimeStagingError):
    """Raised when modality or experiment are not supported by staging."""


@dataclass(frozen=True)
class MirheoSourceResolution:
    root: Path
    origin: str


def resolve_mirheo_source_root(
    *,
    explicit_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    env_var: str = "MESOUQ_GV_MIRHEO_SOURCE_ROOT",
) -> MirheoSourceResolution:
    """Resolve Mirheo source root from explicit input first, then environment."""

    if explicit_root is not None:
        root = Path(explicit_root).expanduser().resolve()
        return MirheoSourceResolution(root=root, origin="explicit")
    source_env = dict(os.environ) if env is None else dict(env)
    value = source_env.get(env_var)
    if value:
        return MirheoSourceResolution(root=Path(value).expanduser().resolve(), origin=f"env:{env_var}")
    raise RuntimeStagingError(
        "Mirheo source root is required. Pass explicit_root or set MESOUQ_GV_MIRHEO_SOURCE_ROOT."
    )


def build_runtime_staging_plan(
    *,
    modality: str,
    experiment: str,
    output_root: str | Path,
    run_id: str,
    mirheo_source_root: str | Path,
    templates: Sequence[TemplateSpec],
) -> RuntimeStagingManifest:
    _validate_modality_and_experiment(modality=modality, experiment=experiment)
    output = Path(output_root).expanduser().resolve()
    source_root = Path(mirheo_source_root).expanduser().resolve()
    run_root = output / modality / experiment / run_id
    planned = tuple(
        _build_staged_entry(
            source_root=source_root,
            run_root=run_root,
            spec=spec,
            allow_missing=True,
        )
        for spec in sorted(templates, key=lambda item: item.destination_relative_path)
    )
    return RuntimeStagingManifest(
        structure="gv",
        modality=modality,
        experiment=experiment,
        dry_run=True,
        mirheo_source_root=source_root,
        mirheo_source_root_origin="explicit",
        output_root=output,
        run_root=run_root,
        run_id=run_id,
        staged_files=planned,
    )


def stage_gv_runtime(
    *,
    modality: str,
    experiment: str,
    output_root: str | Path,
    templates: Sequence[TemplateSpec],
    mirheo_source_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    run_id: str | None = None,
    dry_run: bool = True,
    allow_existing_nonempty_output: bool = False,
) -> RuntimeStagingManifest:
    _validate_modality_and_experiment(modality=modality, experiment=experiment)
    resolution = resolve_mirheo_source_root(explicit_root=mirheo_source_root, env=env)
    _validate_source_root(resolution.root)
    output = Path(output_root).expanduser().resolve()
    _validate_output_root(output, allow_existing_nonempty_output=allow_existing_nonempty_output)

    actual_run_id = run_id or _new_run_id()
    run_root = output / modality / experiment / actual_run_id
    if dry_run:
        staged_files = tuple(
            _build_staged_entry(source_root=resolution.root, run_root=run_root, spec=spec, allow_missing=False)
            for spec in sorted(templates, key=lambda item: item.destination_relative_path)
        )
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        staged_files = tuple(
            _materialize_staged_entry(source_root=resolution.root, run_root=run_root, spec=spec)
            for spec in sorted(templates, key=lambda item: item.destination_relative_path)
        )
    return RuntimeStagingManifest(
        structure="gv",
        modality=modality,
        experiment=experiment,
        dry_run=dry_run,
        mirheo_source_root=resolution.root,
        mirheo_source_root_origin=resolution.origin,
        output_root=output,
        run_root=run_root,
        run_id=actual_run_id,
        staged_files=staged_files,
    )


def reserve_run_directory(
    *,
    output_root: str | Path,
    modality: str,
    experiment: str,
    max_attempts: int = 12,
) -> tuple[str, Path]:
    output = Path(output_root).expanduser().resolve()
    _validate_output_root(output, allow_existing_nonempty_output=True)
    _validate_modality_and_experiment(modality=modality, experiment=experiment)
    base = output / modality / experiment
    base.mkdir(parents=True, exist_ok=True)

    for _ in range(max_attempts):
        run_id = _new_run_id()
        run_root = base / run_id
        try:
            run_root.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_root
    raise RuntimeStagingError("Unable to reserve a unique GV runtime run directory.")


def _validate_modality_and_experiment(*, modality: str, experiment: str) -> None:
    if modality != "gv":
        raise UnsupportedStagingTargetError(f"Unsupported GV staging modality '{modality}'.")
    if experiment not in RUNTIME_EXPERIMENTS:
        supported = ", ".join(RUNTIME_EXPERIMENTS)
        raise UnsupportedStagingTargetError(
            f"Unsupported GV runtime experiment '{experiment}'. Supported: {supported}"
        )


def _validate_source_root(source_root: Path) -> None:
    if not source_root.exists() or not source_root.is_dir():
        raise RuntimeStagingError(f"Mirheo source root does not exist or is not a directory: {source_root}")


def _validate_output_root(output_root: Path, *, allow_existing_nonempty_output: bool) -> None:
    repo_root = _find_repo_root()
    unsafe_roots = (
        repo_root / "src",
        repo_root / "tests",
        repo_root / "gv",
        repo_root / "gv_simulation_files",
        repo_root / "scripts",
    )
    for unsafe_root in unsafe_roots:
        if output_root == unsafe_root or unsafe_root in output_root.parents:
            raise UnsafeOutputPathError(
                f"GV runtime staging output_root must not be inside repository source roots: {unsafe_root}"
            )
    if output_root.exists() and output_root.is_dir() and not allow_existing_nonempty_output:
        try:
            next(output_root.iterdir())
        except StopIteration:
            pass
        else:
            raise UnsafeOutputPathError(
                f"GV runtime staging output_root already exists and is non-empty: {output_root}"
            )


def _build_staged_entry(
    *,
    source_root: Path,
    run_root: Path,
    spec: TemplateSpec,
    allow_missing: bool,
) -> StagedTemplateFile:
    source_path = _resolve_template_source(source_root=source_root, spec=spec)
    if not allow_missing and not source_path.is_file():
        raise MissingTemplateError(
            f"Missing runtime template '{spec.source_relative_path}' under Mirheo source root '{source_root}'."
        )
    destination_path = (run_root / spec.destination_relative_path).resolve()
    _assert_child_path(parent=run_root, child=destination_path)
    size_bytes = source_path.stat().st_size if source_path.is_file() else 0
    sha256 = _sha256_path(source_path) if source_path.is_file() else ""
    return StagedTemplateFile(
        source_path=source_path,
        destination_path=destination_path,
        relative_destination_path=spec.destination_relative_path,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _materialize_staged_entry(*, source_root: Path, run_root: Path, spec: TemplateSpec) -> StagedTemplateFile:
    entry = _build_staged_entry(source_root=source_root, run_root=run_root, spec=spec, allow_missing=False)
    entry.destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(entry.source_path, entry.destination_path)
    return entry


def _resolve_template_source(*, source_root: Path, spec: TemplateSpec) -> Path:
    source_path = (source_root / spec.source_relative_path).resolve()
    _assert_source_path(parent=source_root, child=source_path)
    return source_path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run-{stamp}-{secrets.token_hex(4)}"


def _assert_child_path(*, parent: Path, child: Path) -> None:
    try:
        child.relative_to(parent.resolve())
    except ValueError as exc:
        raise UnsafeOutputPathError(f"Staged destination escapes run root: {child}") from exc


def _assert_source_path(*, parent: Path, child: Path) -> None:
    try:
        child.relative_to(parent.resolve())
    except ValueError as exc:
        raise UnsafeOutputPathError(f"Staged source escapes Mirheo source root: {child}") from exc


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repository root for GV runtime staging.")


__all__ = [
    "MissingTemplateError",
    "MirheoSourceResolution",
    "RuntimeStagingError",
    "UnsafeOutputPathError",
    "UnsupportedStagingTargetError",
    "build_runtime_staging_plan",
    "reserve_run_directory",
    "resolve_mirheo_source_root",
    "stage_gv_runtime",
]
