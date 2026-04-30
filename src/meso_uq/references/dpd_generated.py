from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .gv_common import build_manifest_prefix, ensure_existing_reference_path, resolve_gv_reference_context
from .synthetic import (
    DEFAULT_SURROGATE_BACKEND,
    SyntheticReferenceFixture,
    build_gv_synthetic_fixture,
)


@dataclass(frozen=True)
class DPDGeneratedGVReference:
    data_path: Path
    manifest: dict[str, object]

    def to_manifest(self) -> dict[str, object]:
        return dict(self.manifest)


def build_gv_dpd_generated_fixture(
    *,
    experiment: str,
    geometry: str,
    controls: Mapping[str, object],
    sigma: float,
    d0: float | None = None,
    surrogate_backend: str = DEFAULT_SURROGATE_BACKEND,
    points: Sequence[float] | None = None,
    values: Sequence[float] | None = None,
    observable: str | None = None,
    metadata: Mapping[str, object] | None = None,
    dataset_id: str | None = None,
) -> SyntheticReferenceFixture:
    return build_gv_synthetic_fixture(
        experiment=experiment,
        geometry=geometry,
        controls=controls,
        sigma=sigma,
        d0=d0,
        surrogate_backend=surrogate_backend,
        reference_kind="dpd_generated",
        points=points,
        values=values,
        observable=observable,
        metadata={"source": "dpd_generated_fixture", **(metadata or {})},
        dataset_id=dataset_id,
    )


def dpd_generated_fixture_from_runtime_manifest(
    runtime_manifest: Mapping[str, object],
    *,
    sigma: float,
    d0: float | None = None,
    surrogate_backend: str = DEFAULT_SURROGATE_BACKEND,
    points: Sequence[float] | None = None,
    values: Sequence[float] | None = None,
    observable: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SyntheticReferenceFixture:
    source_files = tuple(str(path) for path in runtime_manifest.get("source_files", ()))
    analysis_commands = tuple(runtime_manifest.get("analysis_commands", ()))
    return build_gv_dpd_generated_fixture(
        experiment=str(runtime_manifest["experiment"]),
        geometry=str(runtime_manifest["geometry"]),
        controls=runtime_manifest["controls"],
        sigma=sigma,
        d0=d0,
        surrogate_backend=surrogate_backend,
        points=points,
        values=values,
        observable=observable,
        metadata={
            "source": "runtime_manifest",
            "runtime_dataset_id": runtime_manifest.get("dataset_id"),
            "runtime_package": runtime_manifest.get("runtime_package"),
            "source_files": source_files,
            "analysis_commands": analysis_commands,
            **(metadata or {}),
        },
        dataset_id=str(runtime_manifest["dataset_id"]),
    )


def load_gv_dpd_generated_reference(
    data_path: str | Path,
    *,
    runtime_manifest: Mapping[str, object] | None = None,
    experiment: str | None = None,
    geometry: str | None = None,
    controls: Mapping[str, float] | None = None,
) -> DPDGeneratedGVReference:
    context = resolve_gv_reference_context(
        runtime_manifest=runtime_manifest,
        experiment=experiment,
        geometry=geometry,
        controls=controls,
    )
    resolved_data_path = ensure_existing_reference_path(
        data_path,
        dataset_id=context.dataset_id,
        reference_kind="dpd_generated",
    ).resolve()
    manifest = build_manifest_prefix(
        context,
        reference_kind="dpd_generated",
        reference_source="dpd_simulation_output",
    )
    manifest["data_reference"] = {
        "path": str(resolved_data_path),
        "format": resolved_data_path.suffix.lstrip(".") or "unknown",
        "exists": True,
    }
    return DPDGeneratedGVReference(data_path=resolved_data_path, manifest=manifest)


def build_gv_dpd_generated_reference_manifest(
    data_path: str | Path,
    **kwargs: object,
) -> dict[str, object]:
    return load_gv_dpd_generated_reference(data_path, **kwargs).to_manifest()


__all__ = [
    "DPDGeneratedGVReference",
    "build_gv_dpd_generated_reference_manifest",
    "build_gv_dpd_generated_fixture",
    "dpd_generated_fixture_from_runtime_manifest",
    "load_gv_dpd_generated_reference",
]
