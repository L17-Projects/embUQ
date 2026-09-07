from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from meso_uq.config.models import emb_geometry_id
from meso_uq.experiments import canonical_dataset_id

from .emb_catalog import EMB_DATASET_SPECS

SUPPORTED_REFERENCE_KINDS = ("real", "synthetic", "dpd_generated")
SUPPORTED_SURROGATE_BACKENDS = ("dnn", "bnn")


def _validate_choice(value: str, *, field_name: str, supported: tuple[str, ...]) -> None:
    if value not in supported:
        raise ValueError(f"Unsupported {field_name} '{value}'. Expected one of {supported}.")


def _resolve_relpath(root: Path, relpath: str | None) -> str | None:
    if relpath is None:
        return None
    return str((root / relpath).resolve())


@dataclass(frozen=True)
class SurrogateDatasetIdentity:
    structure: str
    experiment: str
    geometry: str
    controls: str
    reference_kind: str = "real"
    surrogate_backend: str = "dnn"

    def __post_init__(self) -> None:
        _validate_choice(
            self.reference_kind,
            field_name="reference kind",
            supported=SUPPORTED_REFERENCE_KINDS,
        )
        _validate_choice(
            self.surrogate_backend,
            field_name="surrogate backend",
            supported=SUPPORTED_SURROGATE_BACKENDS,
        )

    @property
    def dataset_id(self) -> str:
        return canonical_dataset_id(
            self.structure,
            self.experiment,
            self.geometry,
            self.controls,
        )

    @property
    def catalog_key(self) -> str:
        return f"{self.dataset_id}:{self.reference_kind}:{self.surrogate_backend}"

    def as_dict(self) -> dict[str, str]:
        return {
            "structure": self.structure,
            "experiment": self.experiment,
            "geometry": self.geometry,
            "controls": self.controls,
            "reference_kind": self.reference_kind,
            "surrogate_backend": self.surrogate_backend,
            "dataset_id": self.dataset_id,
            "catalog_key": self.catalog_key,
        }


@dataclass(frozen=True)
class SurrogateCatalogEntry:
    identity: SurrogateDatasetIdentity
    data_relpath: str
    surrogate_artifact_relpath: str
    reference_manifest_relpath: str | None = None
    training_manifest_relpath: str | None = None
    training_script_relpath: str | None = None
    multi_arch_script_relpath: str | None = None
    group_holdout_script_relpath: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self, repo_root: str | Path) -> dict[str, Any]:
        root = Path(repo_root).resolve()
        payload: dict[str, Any] = self.identity.as_dict()
        payload.update(
            {
                "data": str((root / self.data_relpath).resolve()),
                "surrogate_artifact": str((root / self.surrogate_artifact_relpath).resolve()),
                "reference_manifest": _resolve_relpath(root, self.reference_manifest_relpath),
                "training_manifest": _resolve_relpath(root, self.training_manifest_relpath),
                "training_script": _resolve_relpath(root, self.training_script_relpath),
                "multi_arch_script": _resolve_relpath(root, self.multi_arch_script_relpath),
                "group_holdout_script": _resolve_relpath(root, self.group_holdout_script_relpath),
                "metadata": dict(self.metadata),
            }
        )
        return payload


def _iter_emb_catalog_entries() -> Iterable[SurrogateCatalogEntry]:
    for spec in EMB_DATASET_SPECS:
        geometry = emb_geometry_id(float(spec.diameter_um))
        common_metadata = {
            "legacy_name": spec.name,
            "modality": spec.modality,
            "diameter_um": spec.diameter_um,
            "supports_bnn": True,
        }
        yield SurrogateCatalogEntry(
            identity=SurrogateDatasetIdentity(
                structure="emb",
                experiment=spec.modality,
                geometry=geometry,
                controls="default",
                reference_kind="real",
                surrogate_backend="dnn",
            ),
            data_relpath=spec.data_relpath,
            surrogate_artifact_relpath=spec.dnn_artifact_relpath,
            training_script_relpath=spec.dnn_train_script_relpath,
            multi_arch_script_relpath=spec.dnn_multi_arch_script_relpath,
            group_holdout_script_relpath=spec.group_holdout_script_relpath,
            metadata=common_metadata,
        )
        yield SurrogateCatalogEntry(
            identity=SurrogateDatasetIdentity(
                structure="emb",
                experiment=spec.modality,
                geometry=geometry,
                controls="default",
                reference_kind="real",
                surrogate_backend="bnn",
            ),
            data_relpath=spec.data_relpath,
            surrogate_artifact_relpath=spec.bnn_artifact_relpath,
            training_script_relpath=spec.bnn_train_script_relpath,
            multi_arch_script_relpath=spec.dnn_multi_arch_script_relpath,
            group_holdout_script_relpath=spec.group_holdout_script_relpath,
            metadata=common_metadata,
        )


def iter_surrogate_catalog_entries(*, include_experimental: bool = False) -> tuple[SurrogateCatalogEntry, ...]:
    del include_experimental
    return tuple(_iter_emb_catalog_entries())


def resolve_surrogate_catalog_entries(
    repo_root: str | Path,
    *,
    structure: str | None = None,
    surrogate_backend: str | None = None,
    include_experimental: bool = False,
) -> list[dict[str, Any]]:
    if surrogate_backend is not None:
        _validate_choice(
            surrogate_backend,
            field_name="surrogate backend",
            supported=SUPPORTED_SURROGATE_BACKENDS,
        )
    resolved: list[dict[str, Any]] = []
    for entry in iter_surrogate_catalog_entries(include_experimental=include_experimental):
        if structure is not None and entry.identity.structure != structure:
            continue
        if surrogate_backend is not None and entry.identity.surrogate_backend != surrogate_backend:
            continue
        resolved.append(entry.resolve(repo_root))
    return resolved


def resolve_surrogate_catalog_entry(
    repo_root: str | Path,
    *,
    structure: str,
    experiment: str,
    geometry: str | None = None,
    controls: str | None = None,
    reference_kind: str | None = None,
    surrogate_backend: str = "dnn",
    include_experimental: bool = False,
) -> dict[str, Any]:
    if structure != "emb":
        raise ValueError(f"Unsupported surrogate structure '{structure}'. Expected 'emb'.")

    if geometry is None:
        raise ValueError(
            f"{structure.upper()} surrogate lookup requires a geometry selection; experiment-only lookup is not supported."
        )
    selected_controls = controls or "default"
    selected_reference_kind = reference_kind or "real"
    target = SurrogateDatasetIdentity(
        structure=structure,
        experiment=experiment,
        geometry=geometry,
        controls=selected_controls,
        reference_kind=selected_reference_kind,
        surrogate_backend=surrogate_backend,
    ).catalog_key
    for entry in iter_surrogate_catalog_entries(include_experimental=include_experimental):
        if entry.identity.catalog_key == target:
            return entry.resolve(repo_root)
    raise KeyError(f"No surrogate catalog entry registered for '{target}'.")
