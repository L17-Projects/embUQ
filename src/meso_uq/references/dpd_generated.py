from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .gv_common import (
    DEFAULT_SURROGATE_BACKEND,
    GVReferenceMaterialization,
    build_manifest_prefix,
    ensure_existing_reference_path,
    materialize_reference_records,
    resolve_gv_reference_context,
)
from .synthetic import (
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
    runtime_manifest: Mapping[str, object] | None = None,
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
        runtime_manifest=runtime_manifest,
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
        runtime_manifest=runtime_manifest,
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


def materialize_gv_dpd_generated_reference_dataset(
    fixtures: Sequence[SyntheticReferenceFixture],
    *,
    data_path: str | Path,
    metadata_path: str | Path | None = None,
    collection_id: str | None = None,
    experiments: Sequence[str] | None = None,
    geometries: Sequence[str] | None = None,
    dataset_ids: Sequence[str] | None = None,
) -> GVReferenceMaterialization:
    return materialize_reference_records(
        [fixture.to_record() for fixture in fixtures],
        data_path=data_path,
        metadata_path=metadata_path,
        collection_id=collection_id,
        experiments=experiments,
        geometries=geometries,
        dataset_ids=dataset_ids,
    )


def materialize_gv_dpd_generated_reference(
    data_path: str | Path,
    *,
    output_root: str | Path,
    runtime_manifest: Mapping[str, object] | None = None,
    experiment: str | None = None,
    geometry: str | None = None,
    controls: Mapping[str, float] | None = None,
    dataset_filename: str = "reference_dataset.npz",
    manifest_filename: str = "reference_manifest.json",
) -> GVReferenceMaterialization:
    reference = load_gv_dpd_generated_reference(
        data_path,
        runtime_manifest=runtime_manifest,
        experiment=experiment,
        geometry=geometry,
        controls=controls,
    )
    resolved_root = Path(output_root).resolve()
    dataset_path = resolved_root / dataset_filename
    manifest_path = resolved_root / manifest_filename
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = reference.to_manifest()
    manifest["artifacts"] = {
        "source_data": str(reference.data_path),
        "reference_dataset": str(dataset_path),
        "reference_manifest": str(manifest_path),
    }
    dataset_payload: dict[str, np.ndarray] = {
        "source_data_path": np.asarray(str(reference.data_path), dtype="<U8192"),
        "dataset_id": np.asarray(manifest["dataset_id"], dtype="<U256"),
        "structure": np.asarray(manifest["structure"], dtype="<U32"),
        "experiment": np.asarray(manifest["experiment"], dtype="<U64"),
        "geometry": np.asarray(manifest["geometry"], dtype="<U128"),
        "controls_json": np.asarray(json.dumps(manifest["controls"], sort_keys=True), dtype="<U4096"),
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True), dtype="<U65535"),
    }
    try:
        dataset_payload["data"] = np.loadtxt(reference.data_path, ndmin=2)
    except ValueError:
        dataset_payload["data_parse_status"] = np.asarray("not_numeric_text", dtype="<U64")
    np.savez_compressed(dataset_path, **dataset_payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return GVReferenceMaterialization(
        data_path=dataset_path,
        metadata_path=manifest_path,
        manifest=manifest,
    )


__all__ = [
    "DPDGeneratedGVReference",
    "build_gv_dpd_generated_reference_manifest",
    "build_gv_dpd_generated_fixture",
    "dpd_generated_fixture_from_runtime_manifest",
    "load_gv_dpd_generated_reference",
    "materialize_gv_dpd_generated_reference",
    "materialize_gv_dpd_generated_reference_dataset",
]
