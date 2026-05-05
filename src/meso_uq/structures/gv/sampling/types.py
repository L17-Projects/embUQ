from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .planner import default_sampling_timeout_seconds


_DEFAULT_GV_SAMPLING_ROOT = "_runs/gv/sampling"


@dataclass(frozen=True)
class GVMaterialGeometry:
    """Minimal geometry description shared by GV sampling entry points."""

    radGV: float
    height: float

    def __post_init__(self) -> None:
        radius = float(self.radGV)
        thickness = float(self.height)
        if not isfinite(radius):
            raise ValueError(f"GV geometry radGV must be finite, got {self.radGV!r}.")
        if not isfinite(thickness):
            raise ValueError(f"GV geometry height must be finite, got {self.height!r}.")
        if radius <= 0:
            raise ValueError(f"GV geometry radGV must be positive, got {radius}.")
        if thickness <= 0:
            raise ValueError(f"GV geometry height must be positive, got {thickness}.")
        object.__setattr__(self, "radGV", radius)
        object.__setattr__(self, "height", thickness)

    @property
    def output_root_default(self) -> Path:
        return Path(_DEFAULT_GV_SAMPLING_ROOT)


@dataclass(frozen=True)
class GVSweep:
    """Single explicit control sweep axis for GV sampling."""

    axis: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        axis = str(self.axis).strip()
        if not axis:
            raise ValueError("GV sweep axis must be a non-empty string.")
        try:
            normalized_values = tuple(float(value) for value in self.values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GV sweep values for {axis!r} must be numeric.") from exc
        if not normalized_values:
            raise ValueError(f"GV sweep values for axis {axis!r} cannot be empty.")
        for value in normalized_values:
            if not isfinite(value):
                raise ValueError(f"GV sweep values for axis {axis!r} must be finite.")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "values", normalized_values)


@dataclass(frozen=True)
class GVRuntimeOptions:
    """Runtime-surface knobs for a GV sampling request."""

    controls: Mapping[str, object] = field(default_factory=dict)
    output_root: str | Path = _DEFAULT_GV_SAMPLING_ROOT
    timeout_seconds: int = default_sampling_timeout_seconds

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_seconds, int):
            raise ValueError("GV runtime timeout_seconds must be an integer.")
        if self.timeout_seconds <= 0:
            raise ValueError("GV runtime timeout_seconds must be positive.")


@dataclass(frozen=True)
class GVSampleResult:
    experiment: str
    geometry: GVMaterialGeometry
    material_parameters: Mapping[str, float]
    runtime_options: GVRuntimeOptions
    sweep: GVSweep
    controls: Mapping[str, float] = field(default_factory=dict)
    channels: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] | None = None
    manifest_path: Path | None = None
    hdf5_path: Path | None = None
    runtime_manifests: tuple[Mapping[str, Any], ...] = ()
    plan_manifests: tuple[Mapping[str, Any], ...] = ()
    execution_manifests: tuple[Mapping[str, Any], ...] = ()
    work_dirs: tuple[Path, ...] = ()
    runtime_seconds: float | None = None
    status: str = "validated"

    def as_manifest(self) -> dict[str, object]:
        channel_shapes = {
            name: list(getattr(values, "shape", ()))
            for name, values in self.channels.items()
        }
        return {
            "experiment": self.experiment,
            "geometry": {"radGV": self.geometry.radGV, "height": self.geometry.height},
            "material_parameters": dict(self.material_parameters),
            "runtime_options": {
                "output_root": str(self.runtime_options.output_root),
                "controls": dict(self.runtime_options.controls),
                "timeout_seconds": self.runtime_options.timeout_seconds,
            },
            "controls": dict(self.controls),
            "sweep": {"axis": self.sweep.axis, "values": list(self.sweep.values)},
            "channels": channel_shapes,
            "manifest": dict(self.manifest) if self.manifest is not None else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "hdf5_path": str(self.hdf5_path) if self.hdf5_path is not None else None,
            "work_dirs": [str(path) for path in self.work_dirs],
            "runtime_seconds": self.runtime_seconds,
            "status": self.status,
        }


__all__ = [
    "GVMaterialGeometry",
    "GVRuntimeOptions",
    "GVSampleResult",
    "GVSweep",
]
