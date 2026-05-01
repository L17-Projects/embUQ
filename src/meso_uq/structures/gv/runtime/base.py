from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.experiments import canonical_dataset_id

from ..geometries import DEFAULT_GV_GEOMETRY


GV_RUNTIME_DEFAULT_ROOT = Path("_runs") / "gv" / "runtime"


@dataclass(frozen=True)
class ControlSweep:
    name: str
    start: float
    stop: float
    steps: int

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"Control sweep '{self.name}' must have at least one step.")

    @property
    def default_value(self) -> float:
        return self.start

    def values(self) -> tuple[float, ...]:
        if self.steps == 1:
            return (self.start,)
        step = (self.stop - self.start) / (self.steps - 1)
        return tuple(self.start + index * step for index in range(self.steps))

    def to_manifest(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "start": self.start,
            "stop": self.stop,
            "steps": self.steps,
        }

    def identifier_part(self) -> str:
        if self.steps == 1 or self.start == self.stop:
            return f"{self.name}_{_format_float(self.start)}"
        return f"{self.name}_{_format_float(self.start)}_{_format_float(self.stop)}"


@dataclass(frozen=True)
class DryRunCommand:
    argv: tuple[str, ...]
    cwd: str
    description: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "description": self.description,
        }


@dataclass(frozen=True)
class KnownIssue:
    id: str
    severity: str
    summary: str
    evidence: str

    def to_manifest(self) -> dict[str, str]:
        return {
            "id": self.id,
            "severity": self.severity,
            "summary": self.summary,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class RuntimeDryRun:
    structure: str
    experiment: str
    geometry: str
    controls: Mapping[str, float]
    control_sweeps: tuple[ControlSweep, ...]
    control_id: str
    dataset_id: str
    provenance_root: str
    source_root: str
    legacy_import_root: str
    output_root: str
    work_dir: str
    commands: tuple[DryRunCommand, ...]
    source_files: tuple[str, ...]
    source_manifest: str
    generated_subdirs: tuple[str, ...]
    runtime_package: str
    sweep_mode: str
    first_restart: bool = False
    analysis_commands: tuple[DryRunCommand, ...] = ()
    experimental: bool = False
    known_issues: tuple[KnownIssue, ...] = ()
    notes: tuple[str, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "experiment": self.experiment,
            "geometry": self.geometry,
            "controls": dict(self.controls),
            "control_sweeps": [sweep.to_manifest() for sweep in self.control_sweeps],
            "control_id": self.control_id,
            "dataset_id": self.dataset_id,
            "provenance_root": self.provenance_root,
            "source_root": self.source_root,
            "legacy_import_root": self.legacy_import_root,
            "output_root": self.output_root,
            "work_dir": self.work_dir,
            "source_manifest": self.source_manifest,
            "commands": [command.to_manifest() for command in self.commands],
            "analysis_commands": [command.to_manifest() for command in self.analysis_commands],
            "source_files": list(self.source_files),
            "generated_subdirs": list(self.generated_subdirs),
            "runtime_package": self.runtime_package,
            "sweep_mode": self.sweep_mode,
            "first_restart": self.first_restart,
            "experimental": self.experimental,
            "known_issues": [issue.to_manifest() for issue in self.known_issues],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RuntimeDescriptor:
    experiment: str
    provenance_root: str
    source_files: tuple[str, ...]
    control_sweeps: tuple[ControlSweep, ...]
    sweep_mode: str
    first_restart: bool
    legacy_import_root: str = ""
    runtime_package: str = "mirheo"
    run_script: str = "run.sh"
    generate_script: str = "generate.py"
    execute_argv: tuple[str, ...] = ("bash", "commands.txt")
    execute_description: str = "Execute generated Mirheo commands."
    generated_subdirs: tuple[str, ...] = ("logs", "mesh", "parameter", "restart")
    analysis_commands: tuple[DryRunCommand, ...] = ()
    experimental: bool = False
    opt_in_flag: str = "include_experimental=True"
    known_issues: tuple[KnownIssue, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def control_names(self) -> tuple[str, ...]:
        return tuple(sweep.name for sweep in self.control_sweeps)

    def default_controls(self) -> dict[str, float]:
        return {sweep.name: sweep.default_value for sweep in self.control_sweeps}

    def plan(
        self,
        *,
        output_root: str | Path = GV_RUNTIME_DEFAULT_ROOT,
        geometry: str = DEFAULT_GV_GEOMETRY.id,
        controls: Mapping[str, float] | None = None,
        include_experimental: bool = False,
    ) -> RuntimeDryRun:
        if self.experimental and not include_experimental:
            raise ValueError(f"Experiment '{self.experiment}' requires {self.opt_in_flag}.")
        selected_controls = _normalize_controls(
            default_controls=self.default_controls(),
            overrides=controls or {},
            allowed_controls=self.control_names,
        )
        resolved_output_root = _safe_output_root(output_root)
        control_id = (
            control_identifier(selected_controls)
            if controls
            else sweep_identifier(self.control_sweeps)
        )
        dataset_id = canonical_dataset_id("gv", self.experiment, geometry, control_id)
        work_dir = (
            resolved_output_root
            / self.experiment
            / geometry
            / control_id
            / "work"
        )
        source_root = Path(self.provenance_root).resolve()
        source_files = self._normalize_source_file_entries(
            source_root=source_root,
            source_file_entries=self.source_files,
        )
        source_manifest = self._write_source_manifest(
            source_root=source_root,
            destination=work_dir,
            source_files=source_files,
            geometry=geometry,
            controls=selected_controls,
            control_id=control_id,
            dataset_id=dataset_id,
        )
        return RuntimeDryRun(
            structure="gv",
            experiment=self.experiment,
            geometry=geometry,
            controls=selected_controls,
            control_sweeps=self.control_sweeps,
            control_id=control_id,
            dataset_id=dataset_id,
            provenance_root=self.provenance_root,
            source_root=str(source_root),
            legacy_import_root=self.legacy_import_root,
            output_root=str(resolved_output_root),
            work_dir=str(work_dir),
            commands=self._dry_run_commands(work_dir),
            source_files=tuple(str(path) for path in source_files),
            source_manifest=str(source_manifest),
            generated_subdirs=self.generated_subdirs,
            runtime_package=self.runtime_package,
            sweep_mode=self.sweep_mode,
            first_restart=self.first_restart,
            analysis_commands=self.analysis_commands,
            experimental=self.experimental,
            known_issues=self.known_issues,
            notes=self.notes,
        )

    def _dry_run_commands(self, work_dir: Path) -> tuple[DryRunCommand, ...]:
        parameter_args: list[str] = []
        for sweep in self.control_sweeps:
            parameter_args.extend(
                [
                    "-p",
                    sweep.name,
                    _format_float(sweep.start),
                    _format_float(sweep.stop),
                    str(sweep.steps),
                ]
            )
        mode_arg = f"--{self.sweep_mode}"
        first_args = ("--first",) if self.first_restart else ()
        return (
            DryRunCommand(
                argv=(
                    "python3",
                    self.generate_script,
                    *parameter_args,
                    "--object",
                    "gv",
                    mode_arg,
                    *first_args,
                ),
                cwd=str(work_dir),
                description="Generate GV Mirheo parameter files and command list.",
            ),
            DryRunCommand(
                argv=self.execute_argv,
                cwd=str(work_dir),
                description=self.execute_description,
            ),
        )

    def _normalize_source_file_entries(
        self,
        source_root: Path,
        source_file_entries: Sequence[str],
    ) -> tuple[Path, ...]:
        if not source_root.exists():
            raise FileNotFoundError(f"GV runtime source root does not exist: {source_root}")
        normalized: list[Path] = []
        for raw_path in source_file_entries:
            path = Path(raw_path)
            resolved = (source_root / path).resolve() if not path.is_absolute() else path.resolve()
            if not resolved.exists():
                raise ValueError(f"GV runtime source file not found: {raw_path} in {source_root}")
            try:
                normalized.append(resolved.relative_to(source_root))
            except ValueError as exc:
                raise ValueError(
                    f"GV runtime source file must be under source root '{source_root}': {raw_path}"
                ) from exc
        return tuple(normalized)

    def _write_source_manifest(
        self,
        source_root: Path,
        destination: Path,
        source_files: Sequence[Path],
        geometry: str,
        controls: Mapping[str, float],
        control_id: str,
        dataset_id: str,
    ) -> Path:
        manifest_path = destination / "source_manifest.json"
        destination.mkdir(parents=True, exist_ok=True)
        source_entries: list[dict[str, str | int]] = []
        digest = hashlib.sha256()
        for relative_source_file in source_files:
            source_path = source_root / relative_source_file
            destination_path = destination / relative_source_file
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            payload = source_path.read_bytes()
            rel = relative_source_file.as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload)
            digest.update(b"\0")
            source_entries.append(
                {
                    "path": rel,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
        payload = {
            "structure": "gv",
            "experiment": self.experiment,
            "geometry": geometry,
            "controls": dict(controls),
            "control_id": control_id,
            "dataset_id": dataset_id,
            "source_root": str(source_root),
            "provenance_root": str(source_root),
            "legacy_import_root": self.legacy_import_root,
            "staged_work_dir": str(destination),
            "source_files": [entry["path"] for entry in source_entries],
            "source_file_entries": source_entries,
            "source_file_count": len(source_entries),
            "source_sha256": digest.hexdigest(),
            "generated_subdirs": list(self.generated_subdirs),
            "generated_files": [],
            "runtime_package": self.runtime_package,
            "experimental": self.experimental,
            "known_issues": [issue.to_manifest() for issue in self.known_issues],
            "source_manifest_schema": 1,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path


def control_identifier(controls: Mapping[str, float]) -> str:
    parts = [f"{name}_{_format_float(value)}" for name, value in sorted(controls.items())]
    return "__".join(parts) or "default"


def sweep_identifier(control_sweeps: Sequence[ControlSweep]) -> str:
    return "__".join(sweep.identifier_part() for sweep in control_sweeps) or "default"


def _normalize_controls(
    *,
    default_controls: Mapping[str, float],
    overrides: Mapping[str, float],
    allowed_controls: Sequence[str],
) -> dict[str, float]:
    allowed = set(allowed_controls)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown controls for GV runtime dry-run: {', '.join(unknown)}")
    normalized = dict(default_controls)
    normalized.update({name: float(value) for name, value in overrides.items()})
    return {name: normalized[name] for name in allowed_controls}


def _safe_output_root(output_root: str | Path) -> Path:
    output_path = Path(output_root).expanduser().resolve()
    repo_root = _find_repo_root()
    unsafe_roots = (
        repo_root / "gv_simulation_files",
        repo_root / "gv",
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "tests",
    )
    for unsafe_root in unsafe_roots:
        if output_path == unsafe_root or unsafe_root in output_path.parents:
            raise ValueError(f"GV runtime dry-run output_root must not be inside '{unsafe_root}'.")
    if output_path.name in {"src", "scripts", "tests"}:
        raise ValueError(
            "GV runtime dry-run output_root must not be a source directory such as src, scripts, or tests."
        )
    if output_path.name == "gv":
        raise ValueError("GV runtime dry-run output_root must not be the repository root 'gv'.")
    return output_path


def _format_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "_")


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repository root for GV runtime descriptors.")
