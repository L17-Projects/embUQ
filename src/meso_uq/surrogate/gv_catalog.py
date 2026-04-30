from __future__ import annotations

from typing import Any

from meso_uq.structures import get_structure
from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY

from .catalogs import (
    SUPPORTED_REFERENCE_KINDS,
    SurrogateCatalogEntry,
    SurrogateDatasetIdentity,
)

_REFERENCE_DATA_FILENAME = "reference_dataset.npz"
_REFERENCE_MANIFEST_FILENAME = "reference_manifest.json"
_SURROGATE_MANIFEST_FILENAME = "training_manifest.json"
_SURROGATE_ARTIFACT_FILENAME = "model.pt"
_RUNTIME_MANIFEST_FILENAME = "gv_runtime_dry_run_manifest.json"

_GV_EXPERIMENT_LANES: tuple[dict[str, Any], ...] = (
    {
        "experiment": "stretching",
        "controls": "tot_force_500_50000__bpress_-91",
        "control_values": {"tot_force": {"start": 500.0, "stop": 50000.0, "steps": 80}, "bpress": -91.0},
        "provenance_root": "gv_simulation_files/stretching/gv",
        "notes": (
            "Default stretching lane tracks the staged tot_force sweep with the fixed negative-pressure reference.",
        ),
    },
    {
        "experiment": "buckling",
        "controls": "buck_0_0.75__bpress_-91",
        "control_values": {"buck": {"start": 0.0, "stop": 0.75, "steps": 50}, "bpress": -91.0},
        "provenance_root": "gv_simulation_files/buckling/gv",
        "notes": (
            "Default buckling lane tracks the staged buck sweep with fixed background pressure.",
        ),
    },
    {
        "experiment": "torsion",
        "controls": "theta_0.01_0.1",
        "control_values": {"theta": {"start": 0.01, "stop": 0.1, "steps": 10}},
        "provenance_root": "gv_simulation_files/torsion/gv",
        "notes": (
            "The imported torsion lane hard-codes negative pressure in the staging runtime and exposes theta as the catalog control axis.",
        ),
    },
    {
        "experiment": "eigenmodes",
        "controls": "bpress_-91",
        "control_values": {"bpress": -91.0},
        "provenance_root": "gv_simulation_files/eigenmodes/gv",
        "notes": (
            "The default eigenmodes lane is pressure-conditioned and reuses the staged analysis provenance.",
        ),
    },
    {
        "experiment": "shear_flow",
        "controls": "ptan_0.4__afsi_0__bpress_-91",
        "control_values": {"ptan": 0.4, "afsi": 0.0, "bpress": -91.0},
        "provenance_root": "gv_simulation_files/shear_flow",
        "notes": (
            "Shear-flow remains experimental because the staged OBMD runtime currently reports a bouncer collision overflow.",
        ),
        "experimental": True,
        "requires_opt_in": True,
        "known_issues": (
            {
                "id": "bouncer_collision_candidates_coarse",
                "severity": "error",
                "evidence": "gv_simulation_files/shear_flow/a0/output.out:841",
            },
        ),
    },
)


def _gv_catalog_path(
    *,
    experiment: str,
    geometry: str,
    controls: str,
    reference_kind: str,
    leaf: str,
) -> str:
    return f"_runs/gv/{experiment}/{geometry}/{controls}/{reference_kind}/{leaf}"


def _build_gv_entry(
    *,
    experiment: str,
    controls: str,
    control_values: dict[str, Any],
    provenance_root: str,
    reference_kind: str,
    notes: tuple[str, ...] = (),
    experimental: bool = False,
    requires_opt_in: bool = False,
    known_issues: tuple[dict[str, str], ...] = (),
) -> SurrogateCatalogEntry:
    geometry = DEFAULT_GV_GEOMETRY.id
    root = f"_runs/gv/{experiment}/{geometry}/{controls}/{reference_kind}"
    return SurrogateCatalogEntry(
        identity=SurrogateDatasetIdentity(
            structure="gv",
            experiment=experiment,
            geometry=geometry,
            controls=controls,
            reference_kind=reference_kind,
            surrogate_backend="dnn",
        ),
        data_relpath=f"{root}/{_REFERENCE_DATA_FILENAME}",
        surrogate_artifact_relpath=f"{root}/dnn/{_SURROGATE_ARTIFACT_FILENAME}",
        reference_manifest_relpath=f"{root}/{_REFERENCE_MANIFEST_FILENAME}",
        training_manifest_relpath=f"{root}/dnn/{_SURROGATE_MANIFEST_FILENAME}",
        metadata={
            "structure": "gv",
            "geometry_spec": {
                "id": DEFAULT_GV_GEOMETRY.id,
                "label": DEFAULT_GV_GEOMETRY.label,
                "shape": DEFAULT_GV_GEOMETRY.shape,
                "parameters": dict(DEFAULT_GV_GEOMETRY.parameters),
                "source": DEFAULT_GV_GEOMETRY.source,
            },
            "controls": control_values,
            "control_names": tuple(control_values),
            "control_policy": "GV controls are design inputs and excluded from calibrated parameters.",
            "reference_kind": reference_kind,
            "supports_bnn": False,
            "experimental": experimental,
            "requires_opt_in": requires_opt_in,
            "parameter_contract": {
                "calibrated": list(get_structure("gv").parameter_contract.calibrated_names),
                "nuisance": list(get_structure("gv").parameter_contract.nuisance_names),
            },
            "paths": {
                "runtime_manifest": _gv_catalog_path(
                    experiment=experiment,
                    geometry=geometry,
                    controls=controls,
                    reference_kind=reference_kind,
                    leaf=_RUNTIME_MANIFEST_FILENAME,
                ),
                "reference_manifest": f"{root}/{_REFERENCE_MANIFEST_FILENAME}",
                "reference_dataset": f"{root}/{_REFERENCE_DATA_FILENAME}",
                "surrogate_training_manifest": f"{root}/dnn/{_SURROGATE_MANIFEST_FILENAME}",
                "surrogate_artifact": f"{root}/dnn/{_SURROGATE_ARTIFACT_FILENAME}",
                "provenance_root": provenance_root,
            },
            "notes": notes,
            "known_issues": known_issues,
        },
    )


GV_SURROGATE_SPECS: tuple[SurrogateCatalogEntry, ...] = tuple(
    _build_gv_entry(reference_kind=reference_kind, **lane)
    for lane in _GV_EXPERIMENT_LANES
    for reference_kind in ("synthetic", "dpd_generated")
)


def iter_gv_surrogate_catalog_entries(*, include_experimental: bool = False) -> tuple[SurrogateCatalogEntry, ...]:
    entries: list[SurrogateCatalogEntry] = []
    for entry in GV_SURROGATE_SPECS:
        if entry.metadata.get("experimental") and not include_experimental:
            continue
        entries.append(entry)
    return tuple(entries)


def resolve_gv_surrogate_catalog_entries(
    repo_root: str,
    *,
    reference_kind: str | None = None,
    include_experimental: bool = False,
) -> list[dict[str, Any]]:
    if reference_kind is not None and reference_kind not in SUPPORTED_REFERENCE_KINDS:
        raise ValueError(
            f"Unsupported reference kind '{reference_kind}'. Expected one of {SUPPORTED_REFERENCE_KINDS}."
        )
    resolved: list[dict[str, Any]] = []
    for entry in iter_gv_surrogate_catalog_entries(include_experimental=include_experimental):
        if reference_kind is not None and entry.identity.reference_kind != reference_kind:
            continue
        resolved.append(entry.resolve(repo_root))
    return resolved


def resolve_gv_surrogate_catalog_entry(
    repo_root: str,
    *,
    experiment: str,
    geometry: str | None = None,
    controls: str | None = None,
    reference_kind: str = "synthetic",
    surrogate_backend: str = "dnn",
    include_experimental: bool = False,
) -> dict[str, Any]:
    if surrogate_backend != "dnn":
        raise ValueError(
            "GV surrogate catalog supports only the 'dnn' backend in this tranche; 'bnn' is unsupported."
        )
    if geometry is None or controls is None:
        raise ValueError(
            "GV surrogate lookup requires structure, experiment, geometry, and controls; experiment-only lookup is not supported."
        )
    if reference_kind not in SUPPORTED_REFERENCE_KINDS:
        raise ValueError(
            f"Unsupported reference kind '{reference_kind}'. Expected one of {SUPPORTED_REFERENCE_KINDS}."
        )
    target = SurrogateDatasetIdentity(
        structure="gv",
        experiment=experiment,
        geometry=geometry,
        controls=controls,
        reference_kind=reference_kind,
        surrogate_backend="dnn",
    ).catalog_key
    for entry in iter_gv_surrogate_catalog_entries(include_experimental=include_experimental):
        if entry.identity.catalog_key == target:
            return entry.resolve(repo_root)
    raise KeyError(f"No GV surrogate catalog entry registered for '{target}'.")
