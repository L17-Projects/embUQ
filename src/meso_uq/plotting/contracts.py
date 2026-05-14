from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from meso_uq.core import ArtifactClass


def _metadata_dict(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): value for key, value in (metadata or {}).items()}


@dataclass(frozen=True)
class PlotSeries:
    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    units: str = "dimensionless"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", tuple(float(value) for value in self.x))
        object.__setattr__(self, "y", tuple(float(value) for value in self.y))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if len(self.x) != len(self.y):
            raise ValueError(f"Plot series '{self.name}' has mismatched x/y lengths.")
        if not self.x:
            raise ValueError(f"Plot series '{self.name}' must contain at least one point.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": list(self.x),
            "y": list(self.y),
            "units": self.units,
            "metadata": _metadata_dict(self.metadata),
        }


@dataclass(frozen=True)
class XYPlotRequest:
    figure_id: str
    title: str
    x_label: str
    y_label: str
    series: tuple[PlotSeries, ...]
    output_path: str
    renderer: str = "matplotlib"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "series", tuple(self.series))
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))
        if not self.series:
            raise ValueError(f"Plot request '{self.figure_id}' must contain at least one series.")
        if Path(self.output_path).is_absolute():
            raise ValueError("Plot output paths must be relative or placeholder-based.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "series": [series.as_dict() for series in self.series],
            "output_path": self.output_path,
            "renderer": self.renderer,
            "metadata": _metadata_dict(self.metadata),
        }


@dataclass(frozen=True)
class FigureManifest:
    figure_id: str
    path: str
    source_request: XYPlotRequest
    artifact_class: ArtifactClass = ArtifactClass.FIGURE
    schema_version: str = "meso_uq.figure_manifest.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _metadata_dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "figure_id": self.figure_id,
            "artifact_class": self.artifact_class.value,
            "path": self.path,
            "source_request": self.source_request.as_dict(),
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_request(cls, request: XYPlotRequest) -> "FigureManifest":
        return cls(figure_id=request.figure_id, path=request.output_path, source_request=request)


def configure_headless_environment(env: MutableMapping[str, str]) -> str:
    env.setdefault("MPLBACKEND", "Agg")
    return env["MPLBACKEND"]
