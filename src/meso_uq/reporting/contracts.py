from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from meso_uq.core import ArtifactClass


def _metadata_dict(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): value for key, value in (metadata or {}).items()}


@dataclass(frozen=True)
class ReportSection:
    section_id: str
    title: str
    artifact_ids: tuple[str, ...] = ()
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_ids", tuple(str(item) for item in self.artifact_ids))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if not self.title:
            raise ValueError(f"Report section '{self.section_id}' must have a title.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "artifact_ids": list(self.artifact_ids),
            "text": self.text,
            "metadata": _metadata_dict(self.metadata),
        }


@dataclass(frozen=True)
class ReportManifest:
    report_id: str
    path: str
    sections: tuple[ReportSection, ...]
    run_id: str | None = None
    config_digest: str | None = None
    artifact_class: ArtifactClass = ArtifactClass.REPORT
    schema_version: str = "meso_uq.report_manifest.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if Path(self.path).is_absolute():
            raise ValueError("Report paths must be relative or placeholder-based.")
        if not self.sections:
            raise ValueError(f"Report '{self.report_id}' must contain at least one section.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "artifact_class": self.artifact_class.value,
            "path": self.path,
            "run_id": self.run_id,
            "config_digest": self.config_digest,
            "sections": [section.as_dict() for section in self.sections],
            "metadata": _metadata_dict(self.metadata),
        }


def build_report_manifest(
    *,
    report_id: str,
    path: str,
    sections: Sequence[ReportSection],
    run_id: str | None = None,
    config_digest: str | None = None,
) -> ReportManifest:
    return ReportManifest(
        report_id=report_id,
        path=path,
        sections=tuple(sections),
        run_id=run_id,
        config_digest=config_digest,
    )
