from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TemplateSpec:
    source_relative_path: str
    destination_relative_path: str


@dataclass(frozen=True)
class StagedTemplateFile:
    source_path: Path
    destination_path: Path
    relative_destination_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "destination_path": str(self.destination_path),
            "relative_destination_path": self.relative_destination_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class RuntimeStagingManifest:
    structure: str
    modality: str
    experiment: str
    dry_run: bool
    mirheo_source_root: Path
    mirheo_source_root_origin: str
    output_root: Path
    run_root: Path
    run_id: str
    staged_files: tuple[StagedTemplateFile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "structure": self.structure,
            "modality": self.modality,
            "experiment": self.experiment,
            "dry_run": self.dry_run,
            "mirheo_source_root": str(self.mirheo_source_root),
            "mirheo_source_root_origin": self.mirheo_source_root_origin,
            "output_root": str(self.output_root),
            "run_root": str(self.run_root),
            "run_id": self.run_id,
            "staged_files": [entry.to_dict() for entry in self.staged_files],
        }
