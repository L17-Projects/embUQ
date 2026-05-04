#!/usr/bin/env python3
"""Materialize canonical GV reference artifacts from runtime or reference manifests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.references.dpd_generated import load_gv_dpd_generated_reference  # noqa: E402
from meso_uq.references.gv_common import (
    resolve_gv_reference_context,
    validate_gv_reference_manifest,
)  # noqa: E402
from meso_uq.references.synthetic import generate_gv_synthetic_reference  # noqa: E402
from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY, build_geometry  # noqa: E402
from meso_uq.structures.registry import GeometrySpec, StructureSpec, get_structure  # noqa: E402
from meso_uq.references.gv_common import GV_REFERENCE_KINDS  # noqa: E402
from meso_uq.surrogate.gv_catalog import gv_reference_artifact_paths  # noqa: E402

MANIFEST_SCHEMA_VERSION = 1
WORKFLOW_NAME = "gv_reference_materialization"
DEFAULT_SYNTHETIC_SEED = 7
DEFAULT_SYNTHETIC_POINT_COUNT = 64
EXPERIMENTAL_STATUS = "experimental_reference_materialized"


@dataclass(frozen=True)
class MaterializationRequest:
    runtime_manifest: dict[str, Any] | None
    runtime_manifest_path: Path | None
    source_reference_manifest: dict[str, Any] | None
    source_reference_manifest_path: Path | None
    experiment: str
    geometry: GeometrySpec
    controls: dict[str, float]
    reference_kind: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--runtime-manifest", action="append", default=[], help="Repeatable GV runtime dry-run manifest.")
    parser.add_argument("--reference-manifest", action="append", default=[], help="Repeatable source GV synthetic/DPD reference manifest.")
    parser.add_argument("--structure", default="gv", choices=("gv",))
    parser.add_argument("--selection", action="append", default=[], help="Repeatable structure-qualified selection, for example gv:stretching.")
    parser.add_argument("--experiment", default=None, help="Single-experiment shorthand for direct synthetic generation.")
    parser.add_argument("--geometry-id", action="append", default=[], help="Repeatable registered GV geometry id.")
    parser.add_argument(
        "--geometry",
        action="append",
        default=[],
        metavar="RADIUS,HEIGHT",
        help="Repeatable custom geometry pair for direct synthetic generation.",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable control override for direct synthetic generation.",
    )
    parser.add_argument("--reference-kind", choices=GV_REFERENCE_KINDS, default="synthetic")
    parser.add_argument("--synthetic-seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--synthetic-point-count", type=int, default=DEFAULT_SYNTHETIC_POINT_COUNT)
    parser.add_argument(
        "--dpd-data",
        action="append",
        default=[],
        metavar="DATASET_ID=PATH",
        help="Repeatable dataset-to-path mapping used when materializing dpd_generated references from runtime manifests.",
    )
    parser.add_argument("--include-experimental", action="store_true", default=False)
    return parser


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _normalize_runtime_manifest(payload: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    required = ("structure", "experiment", "geometry", "controls")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Runtime manifest {path} is missing required fields: {', '.join(missing)}")
    if payload.get("structure") != "gv":
        raise ValueError(f"GV reference materialization only supports structure='gv': {path}")
    controls = payload.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError(f"Runtime manifest controls must be a mapping: {path}")
    normalized = dict(payload)
    normalized["controls"] = {str(name): float(value) for name, value in controls.items()}
    normalized["manifest_path"] = str(path.resolve())
    return normalized


def _normalize_reference_manifest(payload: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    required = (
        "manifest_schema_version",
        "structure",
        "experiment",
        "geometry",
        "controls",
        "reference_kind",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Reference manifest {path} is missing required fields: {', '.join(missing)}")
    if payload.get("structure") != "gv":
        raise ValueError(f"GV reference materialization only supports structure='gv': {path}")
    controls = payload.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError(f"Reference manifest controls must be a mapping: {path}")
    reference_kind = str(payload["reference_kind"])
    if reference_kind not in GV_REFERENCE_KINDS:
        raise ValueError(f"Unsupported GV reference kind {reference_kind!r}. Expected one of {GV_REFERENCE_KINDS}.")
    normalized = dict(payload)
    normalized["controls"] = {str(name): float(value) for name, value in controls.items()}
    normalized["manifest_path"] = str(path.resolve())
    validate_gv_reference_manifest(normalized)
    return normalized


def _resolve_selected_experiments(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    if args.experiment is not None:
        requested.append(str(args.experiment))
    for selection in args.selection:
        parts = str(selection).split(":")
        if len(parts) != 2:
            raise ValueError("GV workflow selection must use the form gv:<experiment>.")
        structure_name, experiment_name = parts
        if structure_name != "gv":
            raise ValueError(f"GV workflow selection does not accept structure '{structure_name}'. Use gv:<experiment>.")
        requested.append(experiment_name)
    deduped: list[str] = []
    for experiment_name in requested:
        if experiment_name not in deduped:
            deduped.append(experiment_name)
    return deduped


def _require_gv_experiment_access(experiment_name: str, *, include_experimental: bool) -> None:
    structure = get_structure("gv")
    structure.get_experiment(experiment_name, include_experimental=include_experimental)


def _parse_controls(control_items: list[str], experiment_name: str, structure: StructureSpec) -> dict[str, float]:
    experiment = structure.get_experiment(experiment_name, include_experimental=True)
    allowed = set(experiment.control_names)
    parsed: dict[str, float] = {}
    for item in control_items:
        if "=" not in item:
            raise ValueError(f"Invalid control override '{item}'. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name not in allowed:
            raise ValueError(
                f"Unsupported control '{name}' for experiment '{experiment_name}'. "
                f"Expected one of: {sorted(allowed)}"
            )
        parsed[name] = float(raw_value)
    missing = [name for name in experiment.control_names if name not in parsed]
    if missing:
        raise ValueError(
            f"Missing GV controls for experiment '{experiment_name}': {', '.join(missing)}. "
            "Direct materialization requires the full control set."
        )
    return parsed


def _parse_custom_geometry(item: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(item).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid geometry '{item}'. Expected RADIUS,HEIGHT.")
    return float(parts[0]), float(parts[1])


def _resolve_geometries(args: argparse.Namespace, structure: StructureSpec) -> list[GeometrySpec]:
    geometries: list[GeometrySpec] = []
    for geometry_id in args.geometry_id:
        geometries.append(structure.get_geometry(str(geometry_id)))
    for item in args.geometry:
        radius, height = _parse_custom_geometry(item)
        geometries.append(
            build_geometry(
                radius=radius,
                height=height,
                source="scripts/workflows/gv/materialize_gv_reference.py",
            )
        )
    if not geometries:
        geometries.append(DEFAULT_GV_GEOMETRY)
    return geometries


def _dpd_data_map(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --dpd-data value '{item}'. Expected DATASET_ID=PATH.")
        dataset_id, raw_path = item.split("=", 1)
        mapping[dataset_id.strip()] = Path(raw_path).expanduser().resolve()
    return mapping


def _build_requests(args: argparse.Namespace) -> list[MaterializationRequest]:
    requests: list[MaterializationRequest] = []
    for raw_path in args.runtime_manifest:
        path = Path(raw_path).expanduser().resolve()
        runtime_manifest = _normalize_runtime_manifest(_load_json(path, label="runtime manifest"), path=path)
        _require_gv_experiment_access(str(runtime_manifest["experiment"]), include_experimental=args.include_experimental)
        context = resolve_gv_reference_context(runtime_manifest=runtime_manifest)
        requests.append(
            MaterializationRequest(
                runtime_manifest=runtime_manifest,
                runtime_manifest_path=path,
                source_reference_manifest=None,
                source_reference_manifest_path=None,
                experiment=context.experiment.name,
                geometry=context.geometry,
                controls=context.controls,
                reference_kind=args.reference_kind,
            )
        )
    for raw_path in args.reference_manifest:
        path = Path(raw_path).expanduser().resolve()
        source_reference_manifest = _normalize_reference_manifest(_load_json(path, label="reference manifest"), path=path)
        _require_gv_experiment_access(
            str(source_reference_manifest["experiment"]),
            include_experimental=args.include_experimental,
        )
        context = resolve_gv_reference_context(
            experiment=str(source_reference_manifest["experiment"]),
            geometry=source_reference_manifest.get("geometry_spec") or str(source_reference_manifest["geometry"]),
            controls=source_reference_manifest["controls"],
        )
        requests.append(
            MaterializationRequest(
                runtime_manifest=None,
                runtime_manifest_path=None,
                source_reference_manifest=source_reference_manifest,
                source_reference_manifest_path=path,
                experiment=context.experiment.name,
                geometry=context.geometry,
                controls=context.controls,
                reference_kind=str(source_reference_manifest["reference_kind"]),
            )
        )

    if requests:
        return requests

    experiments = _resolve_selected_experiments(args)
    if not experiments:
        raise ValueError(
            "Provide at least one --runtime-manifest, --reference-manifest, "
            "--experiment, or --selection gv:<experiment>."
        )
    if len(experiments) != 1:
        raise ValueError(
            "Direct materialization without input manifests currently supports exactly one experiment. "
            "Use repeated --runtime-manifest or --reference-manifest inputs for multi-experiment subsets."
        )

    structure = get_structure(args.structure)
    experiment_name = experiments[0]
    structure.get_experiment(experiment_name, include_experimental=args.include_experimental)
    controls = _parse_controls(args.control, experiment_name, structure)
    for geometry in _resolve_geometries(args, structure):
        requests.append(
            MaterializationRequest(
                runtime_manifest=None,
                runtime_manifest_path=None,
                source_reference_manifest=None,
                source_reference_manifest_path=None,
                experiment=experiment_name,
                geometry=geometry,
                controls=controls,
                reference_kind=args.reference_kind,
            )
        )
    return requests


def _geometry_spec_to_manifest(geometry: GeometrySpec) -> dict[str, Any]:
    return {
        "id": geometry.id,
        "label": geometry.label,
        "shape": geometry.shape,
        "parameters": dict(geometry.parameters),
        "source": geometry.source,
    }


def _portable_path(repo_root: Path, value: object) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return str(path)
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return resolved.name


def _sanitize_manifest_paths(value: Any, *, repo_root: Path) -> Any:
    if isinstance(value, Path):
        return _portable_path(repo_root, value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_manifest_paths(item, repo_root=repo_root) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_sanitize_manifest_paths(item, repo_root=repo_root) for item in value]
    if isinstance(value, list):
        return [_sanitize_manifest_paths(item, repo_root=repo_root) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return _portable_path(repo_root, value)
    return value


def _write_runtime_manifest_artifact(
    *,
    request: MaterializationRequest,
    context: Any,
    source_manifest: Mapping[str, Any],
    destination: Path,
    repo_root: Path,
) -> str:
    if request.runtime_manifest is not None:
        payload: Mapping[str, Any] = request.runtime_manifest
    elif isinstance(source_manifest.get("runtime_manifest"), Mapping):
        payload = source_manifest["runtime_manifest"]
    else:
        payload = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "workflow": WORKFLOW_NAME,
            "structure": "gv",
            "experiment": context.experiment.name,
            "selection": f"gv:{context.experiment.name}",
            "geometry": context.geometry.id,
            "geometry_spec": _geometry_spec_to_manifest(context.geometry),
            "controls": dict(context.controls),
            "control_id": context.control_id,
            "dataset_id": context.dataset_id,
            "runtime_package": "reference_materialization_only",
            "experimental": True,
            "notes": [
                "Runtime dry-run manifest synthesized from the GV reference identity.",
                "No Mirheo execution is performed by the reference materialization step.",
            ],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, _sanitize_manifest_paths(dict(payload), repo_root=repo_root))
    return str(destination)


def _write_source_reference_manifest_artifact(
    source_manifest: Mapping[str, Any] | None,
    *,
    destination: Path,
    repo_root: Path,
) -> str | None:
    if source_manifest is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, _sanitize_manifest_paths(dict(source_manifest), repo_root=repo_root))
    return str(destination)


def _resolve_existing_path(raw_path: object, *, repo_root: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _materialized_dataset_payload(
    reference_manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    source_data_path: Path | None,
    source_reference_manifest_path: Path | None,
    runtime_manifest_path: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "manifest_json": np.asarray(json.dumps(reference_manifest, sort_keys=True), dtype="<U65535"),
        "structure": np.asarray(str(reference_manifest["structure"]), dtype="<U64"),
        "experiment": np.asarray(str(reference_manifest["experiment"]), dtype="<U64"),
        "geometry": np.asarray(str(reference_manifest["geometry"]), dtype="<U128"),
        "control_id": np.asarray(str(reference_manifest["control_id"]), dtype="<U256"),
        "dataset_id": np.asarray(str(reference_manifest["dataset_id"]), dtype="<U256"),
        "reference_kind": np.asarray(str(reference_manifest["reference_kind"]), dtype="<U64"),
        "controls_json": np.asarray(json.dumps(reference_manifest["controls"], sort_keys=True), dtype="<U4096"),
        "geometry_spec_json": np.asarray(
            json.dumps(reference_manifest["geometry_spec"], sort_keys=True),
            dtype="<U8192",
        ),
    }
    points = reference_manifest.get("points")
    values = reference_manifest.get("values")
    if isinstance(points, list):
        payload["points"] = np.asarray(points, dtype=np.float64)
    if isinstance(values, list):
        payload["values"] = np.asarray(values, dtype=np.float64)
    observable = reference_manifest.get("observable")
    if observable is not None:
        payload["observable"] = np.asarray(str(observable), dtype="<U128")
    observable_names = reference_manifest.get("observable_names")
    if isinstance(observable_names, list):
        payload["observable_names"] = np.asarray([str(name) for name in observable_names], dtype="<U128")
    data_reference = reference_manifest.get("data_reference")
    if isinstance(data_reference, Mapping):
        path = data_reference.get("path")
        if path is not None:
            payload["data_reference_path"] = np.asarray(_portable_path(repo_root, path), dtype="<U8192")
            payload["source_data_path"] = np.asarray(_portable_path(repo_root, path), dtype="<U8192")
        data_format = data_reference.get("format")
        if data_format is not None:
            payload["data_reference_format"] = np.asarray(str(data_format), dtype="<U64")
    if source_data_path is not None:
        if source_data_path.exists():
            try:
                payload["data"] = np.loadtxt(source_data_path, ndmin=2)
            except ValueError:
                payload["data_parse_status"] = np.asarray("not_numeric_text", dtype="<U64")
        else:
            payload["data_parse_status"] = np.asarray("missing_source_data", dtype="<U64")
    if source_reference_manifest_path is not None:
        payload["source_reference_manifest"] = np.asarray(
            _portable_path(repo_root, source_reference_manifest_path),
            dtype="<U8192",
        )
    if runtime_manifest_path is not None:
        payload["runtime_manifest"] = np.asarray(_portable_path(repo_root, runtime_manifest_path), dtype="<U8192")
    return payload


def _canonical_reference_manifest(
    *,
    request: MaterializationRequest,
    source_manifest: Mapping[str, Any],
    repo_root: Path,
    source_reference_manifest_path: Path | None,
    runtime_manifest_path: Path | None,
    artifact_paths: Mapping[str, str],
) -> dict[str, Any]:
    context = resolve_gv_reference_context(
        runtime_manifest=request.runtime_manifest,
        experiment=request.experiment,
        geometry=request.geometry,
        controls=request.controls,
    )
    notes = list(source_manifest.get("notes", [])) if isinstance(source_manifest.get("notes"), list) else []
    notes.append("This workflow remains experimental until the minimal GV runtime canary path is wired in.")
    artifact_manifest = {
        "reference_root": _portable_path(repo_root, artifact_paths["reference_root"]),
        "reference_dataset": _portable_path(repo_root, artifact_paths["reference_dataset"]),
        "reference_manifest": _portable_path(repo_root, artifact_paths["reference_manifest"]),
        "runtime_manifest": _portable_path(repo_root, artifact_paths["runtime_manifest"]),
    }
    if source_reference_manifest_path is not None:
        artifact_manifest["source_reference_manifest"] = _portable_path(repo_root, source_reference_manifest_path)
    data_reference = source_manifest.get("data_reference")
    if isinstance(data_reference, Mapping) and data_reference.get("path") is not None:
        artifact_manifest["source_data"] = _portable_path(repo_root, data_reference["path"])

    manifest = _sanitize_manifest_paths(dict(source_manifest), repo_root=repo_root)
    if isinstance(manifest.get("data_reference"), Mapping):
        sanitized_reference = dict(manifest["data_reference"])
        if data_reference is not None and isinstance(data_reference, Mapping) and data_reference.get("path") is not None:
            sanitized_reference["path"] = _portable_path(repo_root, data_reference["path"])
        manifest["data_reference"] = sanitized_reference
    manifest.setdefault("generation_seed", source_manifest.get("generation_seed"))
    manifest.update(
        {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "workflow": WORKFLOW_NAME,
            "status": EXPERIMENTAL_STATUS,
            "experimental": True,
            "selection": f"gv:{context.experiment.name}",
            "structure": "gv",
            "experiment": context.experiment.name,
            "geometry": context.geometry.id,
            "geometry_spec": _geometry_spec_to_manifest(context.geometry),
            "controls": dict(context.controls),
            "control_id": context.control_id,
            "dataset_id": context.dataset_id,
            "reference_kind": request.reference_kind,
            "surrogate_backend": str(source_manifest.get("surrogate_backend", "dnn")),
            "artifacts": artifact_manifest,
            "input_manifests": {
                "runtime_dry_run": _portable_path(repo_root, runtime_manifest_path)
                if runtime_manifest_path is not None
                else None,
                "source_reference": _portable_path(repo_root, source_reference_manifest_path)
                if source_reference_manifest_path is not None
                else None,
            },
            "notes": notes,
        }
    )
    return manifest


def _build_source_manifest(
    request: MaterializationRequest,
    *,
    synthetic_seed: int,
    synthetic_point_count: int,
    dpd_data_map: Mapping[str, Path],
) -> dict[str, Any]:
    if request.source_reference_manifest is not None:
        return dict(request.source_reference_manifest)

    if request.reference_kind == "synthetic":
        return generate_gv_synthetic_reference(
            seed=synthetic_seed,
            runtime_manifest=request.runtime_manifest,
            experiment=request.experiment,
            geometry=request.geometry,
            controls=request.controls,
            point_count=synthetic_point_count,
        ).to_manifest()

    context = resolve_gv_reference_context(
        runtime_manifest=request.runtime_manifest,
        experiment=request.experiment,
        geometry=request.geometry.id,
        controls=request.controls,
    )
    try:
        data_path = dpd_data_map[context.dataset_id]
    except KeyError as exc:
        raise ValueError(
            "DPD materialization requires --dpd-data DATASET_ID=PATH for each selected runtime lane. "
            f"Missing mapping for {context.dataset_id!r}."
        ) from exc
    return load_gv_dpd_generated_reference(
        data_path,
        runtime_manifest=request.runtime_manifest,
        experiment=request.experiment,
        geometry=request.geometry,
        controls=request.controls,
    ).to_manifest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _materialize_request(
    request: MaterializationRequest,
    *,
    repo_root: Path,
    synthetic_seed: int,
    synthetic_point_count: int,
    dpd_data_map: Mapping[str, Path],
) -> dict[str, str]:
    context = resolve_gv_reference_context(
        runtime_manifest=request.runtime_manifest,
        experiment=request.experiment,
        geometry=request.geometry,
        controls=request.controls,
    )

    artifact_paths = gv_reference_artifact_paths(
        repo_root,
        experiment=context.experiment.name,
        geometry=context.geometry.id,
        controls=context.control_id,
        reference_kind=request.reference_kind,
    )
    reference_root = Path(artifact_paths["reference_root"])
    source_manifest = _build_source_manifest(
        request,
        synthetic_seed=synthetic_seed,
        synthetic_point_count=synthetic_point_count,
        dpd_data_map=dpd_data_map,
    )
    source_data_path = None
    data_reference = source_manifest.get("data_reference")
    if isinstance(data_reference, Mapping):
        source_data_path = _resolve_existing_path(data_reference.get("path"), repo_root=repo_root)
    runtime_manifest_artifact = _write_runtime_manifest_artifact(
        request=request,
        context=context,
        source_manifest=source_manifest,
        destination=Path(artifact_paths["runtime_manifest"]),
        repo_root=repo_root,
    )
    source_reference_manifest_artifact = _write_source_reference_manifest_artifact(
        request.source_reference_manifest,
        destination=Path(artifact_paths["source_reference_manifest"]),
        repo_root=repo_root,
    )
    canonical_manifest = _canonical_reference_manifest(
        request=request,
        source_manifest=source_manifest,
        repo_root=repo_root,
        source_reference_manifest_path=Path(source_reference_manifest_artifact)
        if source_reference_manifest_artifact is not None
        else None,
        runtime_manifest_path=Path(runtime_manifest_artifact),
        artifact_paths=artifact_paths,
    )
    reference_root.mkdir(parents=True, exist_ok=True)
    dataset_payload = _materialized_dataset_payload(
        canonical_manifest,
        repo_root=repo_root,
        source_data_path=source_data_path,
        source_reference_manifest_path=Path(source_reference_manifest_artifact)
        if source_reference_manifest_artifact is not None
        else None,
        runtime_manifest_path=Path(runtime_manifest_artifact),
    )
    np.savez_compressed(Path(artifact_paths["reference_dataset"]), **dataset_payload)
    _write_json(Path(artifact_paths["reference_manifest"]), canonical_manifest)
    return {
        "dataset_id": canonical_manifest["dataset_id"],
        "reference_kind": canonical_manifest["reference_kind"],
        "reference_manifest": artifact_paths["reference_manifest"],
        "reference_dataset": artifact_paths["reference_dataset"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()

    try:
        requests = _build_requests(args)
        dpd_data_map = _dpd_data_map(args.dpd_data)
        results = [
            _materialize_request(
                request,
                repo_root=repo_root,
                synthetic_seed=args.synthetic_seed,
                synthetic_point_count=args.synthetic_point_count,
                dpd_data_map=dpd_data_map,
            )
            for request in requests
        ]
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    for result in results:
        print(
            f"{result['dataset_id']} [{result['reference_kind']}]: "
            f"{result['reference_manifest']} :: {result['reference_dataset']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
