from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from meso_uq.structures import get_structure
from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY
from meso_uq.structures.registry import GeometrySpec
from meso_uq.structures.gv.runtime.base import control_identifier

from .catalogs import (
    SUPPORTED_REFERENCE_KINDS,
    SurrogateCatalogEntry,
    SurrogateDatasetIdentity,
)

_REFERENCE_DATA_FILENAME = "reference_dataset.npz"
_REFERENCE_MANIFEST_FILENAME = "reference_manifest.json"
_SOURCE_REFERENCE_MANIFEST_FILENAME = "source_reference_manifest.json"
_SURROGATE_MANIFEST_FILENAME = "training_manifest.json"
_SURROGATE_ARTIFACT_FILENAME = "model.pt"
_RUNTIME_MANIFEST_FILENAME = "gv_runtime_dry_run_manifest.json"
_GV_AXIS_COLUMN_BY_EXPERIMENT: Mapping[str, str] = {
    "shear_flow": "shear_coord",
}
_GV_RESPONSE_COLUMN_BY_EXPERIMENT: Mapping[str, str] = {
    "shear_flow": "shear_response",
}
_GV_CONTROL_POLICY = "GV controls are design inputs and excluded from calibrated parameters."

_GV_EXPERIMENT_LANES: tuple[dict[str, Any], ...] = (
    {
        "experiment": "stretching",
        "controls": "tot_force_500_50000__bpress_-91",
        "control_values": {"tot_force": {"start": 500.0, "stop": 50000.0, "steps": 80}, "bpress": -91.0},
        "source_root": "gv/stretching/src",
        "legacy_import_root": "gv_simulation_files/stretching/gv",
        "notes": (
            "Default stretching lane tracks the staged tot_force sweep with the fixed negative-pressure reference.",
        ),
    },
    {
        "experiment": "buckling",
        "controls": "buck_0_0.75__bpress_-91",
        "control_values": {"buck": {"start": 0.0, "stop": 0.75, "steps": 50}, "bpress": -91.0},
        "source_root": "gv/buckling/src",
        "legacy_import_root": "gv_simulation_files/buckling/gv",
        "notes": (
            "Default buckling lane tracks the staged buck sweep with fixed background pressure.",
        ),
    },
    {
        "experiment": "torsion",
        "controls": "theta_0.01_0.1",
        "control_values": {"theta": {"start": 0.01, "stop": 0.1, "steps": 10}},
        "source_root": "gv/torsion/src",
        "legacy_import_root": "gv_simulation_files/torsion/gv",
        "notes": (
            "The imported torsion lane hard-codes negative pressure in the staging runtime and exposes theta as the catalog control axis.",
        ),
    },
    {
        "experiment": "eigenmodes",
        "controls": "bpress_-91",
        "control_values": {"bpress": -91.0},
        "source_root": "gv/eigenmodes/src",
        "legacy_import_root": "gv_simulation_files/eigenmodes/gv",
        "notes": (
            "The default eigenmodes lane is pressure-conditioned and reuses the staged analysis provenance.",
        ),
    },
    {
        "experiment": "shear_flow",
        "controls": "ptan_0.4__afsi_0__bpress_-91",
        "control_values": {"ptan": 0.4, "afsi": 0.0, "bpress": -91.0},
        "source_root": "gv/shear_flow/src",
        "legacy_import_root": "gv_simulation_files/shear_flow",
        "notes": (
            "Shear-flow remains experimental because the staged OBMD runtime currently reports a bouncer collision overflow.",
        ),
        "experimental": True,
        "requires_opt_in": True,
        "known_issues": (
            {
                "id": "bouncer_collision_candidates_coarse",
                "severity": "error",
                "evidence": "gv/shear_flow/src/fixtures/bouncer_collision_candidates_coarse_excerpt.txt",
            },
        ),
    },
)


def _catalog_geometries() -> tuple[GeometrySpec, ...]:
    return get_structure("gv").geometries


def _axis_column(experiment: str) -> str:
    return _GV_AXIS_COLUMN_BY_EXPERIMENT.get(experiment, "observable_axis")


def _target_column(experiment: str) -> str:
    return _GV_RESPONSE_COLUMN_BY_EXPERIMENT.get(experiment, "response")


def _parameter_order() -> dict[str, list[str]]:
    contract = get_structure("gv").parameter_contract
    return {
        "calibrated": list(contract.calibrated_names),
        "nuisance": list(contract.nuisance_names),
    }


def _observable_schema() -> dict[str, Any]:
    structure = get_structure("gv")
    return {
        "identity": "meso_uq.structures.gv",
        "repository_path": "src/meso_uq/structures/gv",
        "controls": {
            experiment.name: list(experiment.control_names) for experiment in structure.experiments
        },
        "observables": {
            experiment.name: [observable.name for observable in experiment.observables]
            for experiment in structure.experiments
        },
    }


def _validate_controls(controls: Mapping[str, Any], experiment: str) -> None:
    structure = get_structure("gv")
    experiment_spec = structure.get_experiment(experiment, include_experimental=True)
    overlap = {name for name in controls if name in structure.parameter_contract.calibrated_names}
    nuisance = {name for name in controls if name in structure.parameter_contract.nuisance_names}
    if overlap or nuisance:
        raise ValueError(
            "GV reference controls must remain fixed design inputs and cannot collide with calibrated or nuisance parameters."
        )
    unknown = sorted(set(controls) - set(experiment_spec.control_names))
    if unknown:
        raise ValueError(f"Unknown GV controls for {experiment}: {', '.join(unknown)}")


def _feature_order(control_values: Mapping[str, Any], experiment: str) -> tuple[str, ...]:
    return tuple(
        (
            *_parameter_order()["calibrated"],
            "radius",
            "height",
            *tuple(str(name) for name in control_values),
            _axis_column(experiment),
        )
    )


def _reference_provenance(source_root: str, legacy_import_root: str) -> dict[str, str]:
    return {
        "runtime_source_root": source_root,
        "runtime_provenance_root": source_root,
        "runtime_legacy_import_root": legacy_import_root,
    }


def _gv_catalog_path(
    *,
    experiment: str,
    geometry: str,
    controls: str,
    reference_kind: str,
    leaf: str,
) -> str:
    return f"_runs/gv/{experiment}/{geometry}/{controls}/{reference_kind}/{leaf}"


def gv_control_id(controls: Mapping[str, float]) -> str:
    return control_identifier({str(name): float(value) for name, value in controls.items()})


def gv_reference_root_relpath(
    *,
    experiment: str,
    geometry: str,
    controls: str,
    reference_kind: str,
) -> str:
    return _gv_catalog_path(
        experiment=experiment,
        geometry=geometry,
        controls=controls,
        reference_kind=reference_kind,
        leaf="",
    ).rstrip("/")


def gv_reference_artifact_paths(
    repo_root: str | Path,
    *,
    experiment: str,
    geometry: str,
    controls: str,
    reference_kind: str,
) -> dict[str, str]:
    root = Path(repo_root).resolve()
    reference_root = root / gv_reference_root_relpath(
        experiment=experiment,
        geometry=geometry,
        controls=controls,
        reference_kind=reference_kind,
    )
    dnn_root = reference_root / "dnn"
    return {
        "reference_root": str(reference_root),
        "reference_dataset": str(reference_root / _REFERENCE_DATA_FILENAME),
        "reference_manifest": str(reference_root / _REFERENCE_MANIFEST_FILENAME),
        "source_reference_manifest": str(reference_root / _SOURCE_REFERENCE_MANIFEST_FILENAME),
        "runtime_manifest": str(reference_root / _RUNTIME_MANIFEST_FILENAME),
        "surrogate_root": str(dnn_root),
        "surrogate_training_manifest": str(dnn_root / _SURROGATE_MANIFEST_FILENAME),
        "surrogate_artifact": str(dnn_root / _SURROGATE_ARTIFACT_FILENAME),
    }


def _build_gv_entry(
    *,
    experiment: str,
    geometry: GeometrySpec,
    controls: str,
    control_values: dict[str, Any],
    source_root: str,
    legacy_import_root: str,
    reference_kind: str,
    notes: tuple[str, ...] = (),
    experimental: bool = False,
    requires_opt_in: bool = False,
    known_issues: tuple[dict[str, str], ...] = (),
) -> SurrogateCatalogEntry:
    root = f"_runs/gv/{experiment}/{geometry.id}/{controls}/{reference_kind}"
    structure = get_structure("gv")
    _validate_controls(control_values, experiment)
    axis = _axis_column(experiment)
    target = _target_column(experiment)
    return SurrogateCatalogEntry(
        identity=SurrogateDatasetIdentity(
            structure="gv",
            experiment=experiment,
            geometry=geometry.id,
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
            "geometry": geometry.id,
            "geometry_spec": {
                "id": geometry.id,
                "label": geometry.label,
                "shape": geometry.shape,
                "parameters": dict(geometry.parameters),
                "source": geometry.source,
            },
            "controls": control_values,
            "control_names": tuple(control_values),
            "control_policy": _GV_CONTROL_POLICY,
            "feature_order": {
                "input": list(_feature_order(control_values, experiment)),
                "axis": axis,
                "target": target,
            },
            "parameter_order": _parameter_order(),
            "observable_schema": _observable_schema(),
            "reference_provenance": _reference_provenance(source_root, legacy_import_root),
            "reference_kind": reference_kind,
            "supports_bnn": False,
            "experimental": experimental,
            "requires_opt_in": requires_opt_in,
            "backend": "dnn",
            "parameter_contract": {
                "calibrated": list(get_structure("gv").parameter_contract.calibrated_names),
                "nuisance": list(get_structure("gv").parameter_contract.nuisance_names),
            },
            "paths": {
                "runtime_manifest": _gv_catalog_path(
                    experiment=experiment,
                    geometry=geometry.id,
                    controls=controls,
                    reference_kind=reference_kind,
                    leaf=_RUNTIME_MANIFEST_FILENAME,
                ),
                "reference_manifest": f"{root}/{_REFERENCE_MANIFEST_FILENAME}",
                "reference_dataset": f"{root}/{_REFERENCE_DATA_FILENAME}",
                "surrogate_training_manifest": f"{root}/dnn/{_SURROGATE_MANIFEST_FILENAME}",
                "surrogate_artifact": f"{root}/dnn/{_SURROGATE_ARTIFACT_FILENAME}",
                "provenance_root": source_root,
                "source_root": source_root,
                "legacy_import_root": legacy_import_root,
            },
            "notes": notes,
            "known_issues": known_issues,
        },
    )


def _build_gv_specs(*, include_experimental: bool = False) -> tuple[SurrogateCatalogEntry, ...]:
    specs: list[SurrogateCatalogEntry] = []
    reference_kinds = ("synthetic", "dpd_generated")
    for geometry in _catalog_geometries() or (DEFAULT_GV_GEOMETRY,):
        for lane in _GV_EXPERIMENT_LANES:
            if not include_experimental and lane.get("experimental", False):
                continue
            for reference_kind in reference_kinds:
                specs.append(
                    _build_gv_entry(
                        reference_kind=reference_kind,
                        geometry=geometry,
                        **lane,
                    )
                )
    return tuple(specs)


GV_SURROGATE_SPECS: tuple[SurrogateCatalogEntry, ...] = _build_gv_specs()


def iter_gv_surrogate_catalog_entries(
    *,
    include_experimental: bool = False,
    experiments: tuple[str, ...] | None = None,
    geometries: tuple[str, ...] | None = None,
) -> tuple[SurrogateCatalogEntry, ...]:
    entries: list[SurrogateCatalogEntry] = []
    experiment_filter = set(experiments or ())
    geometry_filter = set(geometries or ())
    for entry in _build_gv_specs(include_experimental=include_experimental):
        if experiment_filter and entry.identity.experiment not in experiment_filter:
            continue
        if geometry_filter and entry.identity.geometry not in geometry_filter:
            continue
        entries.append(entry)
    return tuple(entries)


def resolve_gv_surrogate_catalog_entries(
    repo_root: str,
    *,
    reference_kind: str | None = None,
    include_experimental: bool = False,
    experiments: tuple[str, ...] | None = None,
    geometries: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if reference_kind is not None and reference_kind not in SUPPORTED_REFERENCE_KINDS:
        raise ValueError(
            f"Unsupported reference kind '{reference_kind}'. Expected one of {SUPPORTED_REFERENCE_KINDS}."
        )
    resolved: list[dict[str, Any]] = []
    for entry in iter_gv_surrogate_catalog_entries(
        include_experimental=include_experimental,
        experiments=experiments,
        geometries=geometries,
    ):
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
