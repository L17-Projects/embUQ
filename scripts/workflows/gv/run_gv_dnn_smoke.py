#!/usr/bin/env python3
"""Run a tiny GV DNN smoke workflow from GV runtime or reference manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - exercised via integration tests in this workspace.
    torch = None
    DataLoader = None
    TensorDataset = None

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY, build_geometry  # noqa: E402
from meso_uq.structures.gv.runtime.catalog import plan_runtime  # noqa: E402
from meso_uq.structures.registry import GeometrySpec, StructureSpec, get_structure  # noqa: E402

if torch is not None:
    from meso_uq.surrogate import MLP, init_weights, load_model_states, save_model_states, train_model  # noqa: E402

PARAMETER_NAMES = ("ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c")
GEOMETRY_COLUMNS = ("radius", "height")
REFERENCE_KINDS = ("synthetic", "dpd_generated")
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_SEED = 7
DEFAULT_NUM_CURVES = 5
DEFAULT_POINTS_PER_CURVE = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", default=None, help="GV runtime dry-run manifest from the Mirheo stage.")
    parser.add_argument("--reference-manifest", default=None, help="Optional synthetic or DPD-generated GV reference manifest.")
    parser.add_argument("--structure", default="gv", choices=("gv",))
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--geometry-id", default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable control override, for example --control theta=0.02.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--include-experimental", action="store_true", default=False)
    parser.add_argument("--backend", choices=("dnn",), default="dnn")
    parser.add_argument("--reference-kind", choices=REFERENCE_KINDS, default="synthetic")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-curves", type=int, default=DEFAULT_NUM_CURVES)
    parser.add_argument("--points-per-curve", type=int, default=DEFAULT_POINTS_PER_CURVE)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epoch", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    return parsed


def _resolve_geometry(args: argparse.Namespace, structure: StructureSpec) -> GeometrySpec:
    if args.geometry_id is not None:
        return structure.get_geometry(args.geometry_id)
    if args.radius is None and args.height is None:
        return DEFAULT_GV_GEOMETRY
    if args.radius is None or args.height is None:
        raise ValueError("Custom geometry requires both --radius and --height.")
    return build_geometry(radius=args.radius, height=args.height, source="scripts/workflows/gv/run_gv_dnn_smoke.py")


def _upstream_contract() -> dict[str, Any]:
    return {
        "required_runtime_manifest_fields": [
            "structure",
            "experiment",
            "geometry",
            "geometry_spec",
            "controls",
            "dataset_id",
        ],
        "required_reference_manifest_fields": [
            "structure",
            "experiment",
            "geometry",
            "controls",
            "reference_kind",
        ],
        "optional_reference_manifest_fields": [
            "dataset_id",
            "geometry_spec",
            "runtime_manifest",
            "notes",
        ],
        "catalog_seam": (
            "Reference/catalog workers must provide the identity axes above and either "
            "a dataset_id or a materialized dataset recipe/runtime manifest."
        ),
    }


def _geometry_spec_to_manifest(geometry: GeometrySpec) -> dict[str, Any]:
    return {
        "id": geometry.id,
        "label": geometry.label,
        "parameters": dict(geometry.parameters),
        "source": geometry.source,
    }


def _validate_geometry_parameters(manifest: dict[str, Any], *, manifest_path: Path | None = None) -> dict[str, Any]:
    geometry_spec = manifest.get("geometry_spec")
    if not isinstance(geometry_spec, dict):
        label = f" {manifest_path}" if manifest_path is not None else ""
        raise ValueError(f"Runtime manifest{label} is missing geometry_spec.")
    parameters = geometry_spec.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Runtime manifest is missing geometry_spec.parameters.")
    for required in ("radius", "height"):
        if required not in parameters:
            raise ValueError(f"Runtime manifest is missing geometry_spec.parameters.{required}.")
    return geometry_spec


def _load_runtime_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("structure", "experiment", "geometry", "controls")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"Runtime manifest {path} is missing required fields: {', '.join(missing)}"
        )
    if payload["structure"] != "gv":
        raise ValueError(f"GV smoke workflow only supports structure='gv', got {payload['structure']!r}.")
    if not isinstance(payload["controls"], dict):
        raise ValueError("runtime manifest controls must be a mapping.")
    _validate_geometry_parameters(payload, manifest_path=path)
    normalized = dict(payload)
    normalized["manifest_path"] = str(path.resolve())
    return normalized


def _load_reference_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("structure", "experiment", "geometry", "controls", "reference_kind")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"Reference manifest {path} is missing required fields: {', '.join(missing)}"
        )
    if payload["structure"] != "gv":
        raise ValueError(f"GV smoke workflow only supports structure='gv', got {payload['structure']!r}.")
    if not isinstance(payload["controls"], dict):
        raise ValueError("reference manifest controls must be a mapping.")
    if payload["reference_kind"] not in REFERENCE_KINDS:
        raise ValueError(f"Unsupported GV reference kind {payload['reference_kind']!r}. Expected one of {REFERENCE_KINDS}.")
    normalized = dict(payload)
    normalized.setdefault("backend", "dnn")
    normalized.setdefault("manifest_schema_version", MANIFEST_SCHEMA_VERSION)
    normalized.setdefault("upstream_contract", _upstream_contract())
    return normalized


def _build_reference_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.runtime_manifest is not None:
        if args.reference_manifest is not None:
            raise ValueError("Use either --runtime-manifest or --reference-manifest, not both.")
        runtime_manifest_path = Path(args.runtime_manifest).expanduser().resolve()
        runtime = _load_runtime_manifest(runtime_manifest_path)
        return {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "structure": "gv",
            "experiment": runtime["experiment"],
            "geometry": runtime["geometry"],
            "geometry_spec": runtime["geometry_spec"],
            "controls": runtime["controls"],
            "dataset_id": runtime.get("dataset_id"),
            "reference_kind": args.reference_kind,
            "backend": args.backend,
            "runtime_manifest": runtime,
            "input_manifests": {"runtime_dry_run": str(runtime_manifest_path)},
            "notes": [
                "Synthetic smoke reference derived from GV runtime dry-run metadata.",
                "No Mirheo execution is performed by this workflow.",
            ],
            "upstream_contract": _upstream_contract(),
        }

    if args.reference_manifest is not None:
        return _load_reference_manifest(Path(args.reference_manifest).resolve())

    if not args.experiment:
        raise ValueError("--experiment is required when --reference-manifest is not provided.")

    structure = get_structure(args.structure)
    structure.get_experiment(args.experiment, include_experimental=args.include_experimental)
    geometry = _resolve_geometry(args, structure)
    controls = _parse_controls(args.control, args.experiment, structure)
    runtime = plan_runtime(
        args.experiment,
        output_root=Path(args.output_root).resolve() / "runtime_plan",
        geometry=geometry.id,
        controls=controls or None,
        include_experimental=args.include_experimental,
    ).to_manifest()
    runtime["geometry_spec"] = _geometry_spec_to_manifest(geometry)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "structure": args.structure,
        "experiment": args.experiment,
        "geometry": geometry.id,
        "geometry_spec": runtime["geometry_spec"],
        "controls": runtime["controls"],
        "dataset_id": runtime["dataset_id"],
        "reference_kind": args.reference_kind,
        "backend": args.backend,
        "runtime_manifest": runtime,
        "notes": [
            "Synthetic smoke reference derived from GV runtime dry-run metadata.",
            "No Mirheo execution is performed by this workflow.",
        ],
        "upstream_contract": _upstream_contract(),
    }


def _experiment_factor(experiment: str) -> float:
    return {
        "stretching": 1.0,
        "buckling": 1.2,
        "torsion": 1.4,
        "eigenmodes": 1.6,
        "shear_flow": 1.8,
    }.get(experiment, 1.1)


def _default_parameter_vector() -> dict[str, float]:
    return {
        "ka": 1.10,
        "kb": 0.85,
        "mu": 0.55,
        "b1": 0.18,
        "b2": 0.27,
        "a3": 0.33,
        "a4": 0.41,
        "mu_l": 0.62,
        "c": 0.74,
    }


def _axis_target_columns(experiment: str) -> tuple[str, str]:
    if experiment == "shear_flow":
        return "shear_coord", "shear_response"
    return "observable_axis", "response"


def _control_columns(reference_manifest: dict[str, Any]) -> list[str]:
    experiment_name = str(reference_manifest["experiment"])
    controls = {str(name) for name in reference_manifest["controls"]}
    try:
        experiment = get_structure("gv").get_experiment(experiment_name, include_experimental=True)
    except KeyError:
        return sorted(controls)
    ordered = [name for name in experiment.control_names if name in controls]
    ordered.extend(sorted(controls - set(ordered)))
    return ordered


def _axis_points(points_per_curve: int) -> tuple[float, ...]:
    if points_per_curve < 2:
        raise ValueError("--points-per-curve must be at least 2.")
    return tuple(index / float(points_per_curve - 1) for index in range(points_per_curve))


def _curve_offsets(num_curves: int) -> tuple[float, ...]:
    if num_curves < 2:
        raise ValueError("--num-curves must be at least 2.")
    midpoint = (num_curves - 1) / 2.0
    return tuple((index - midpoint) * 0.08 for index in range(num_curves))


def _build_smoke_rows(
    reference_manifest: dict[str, Any],
    *,
    num_curves: int,
    points_per_curve: int,
) -> list[dict[str, float | str]]:
    controls = {str(key): float(value) for key, value in reference_manifest["controls"].items()}
    geometry_spec = reference_manifest.get("geometry_spec", {})
    geometry_parameters = geometry_spec.get("parameters", {}) if isinstance(geometry_spec, dict) else {}
    radius = float(geometry_parameters.get("radius", 2.0))
    height = float(geometry_parameters.get("height", 14.28))
    base = _default_parameter_vector()
    axis_col, target_col = _axis_target_columns(str(reference_manifest["experiment"]))
    axis_points = _axis_points(points_per_curve)
    control_names = _control_columns(reference_manifest)
    rows: list[dict[str, float | str]] = []
    for curve_idx, offset in enumerate(_curve_offsets(num_curves)):
        curve_params = {name: value * (1.0 + offset) for name, value in base.items()}
        for axis_value in axis_points:
            row: dict[str, float | str] = {
                "source_curve_id": f"curve_{curve_idx}",
                "radius": radius,
                "height": height,
                axis_col: float(axis_value),
            }
            row.update(curve_params)
            row.update(controls)
            raw_response = (
                0.55 * curve_params["ka"]
                + 0.35 * curve_params["kb"]
                + 0.30 * curve_params["mu"]
                + 0.20 * curve_params["b1"]
                + 0.25 * curve_params["b2"]
                + 0.22 * curve_params["a3"]
                + 0.18 * curve_params["a4"]
                + 0.28 * curve_params["mu_l"]
                + 0.24 * curve_params["c"]
                + 0.17 * axis_value
                + 0.03 * radius
                + 0.005 * height
                + sum((index + 1) * 0.01 * controls[name] for index, name in enumerate(control_names))
            )
            row[target_col] = max(0.0, _experiment_factor(str(reference_manifest["experiment"])) + raw_response)
            rows.append(row)
    return rows


def _ordered_dataset_columns(reference_manifest: dict[str, Any]) -> list[str]:
    axis_col, target_col = _axis_target_columns(str(reference_manifest["experiment"]))
    return (
        list(PARAMETER_NAMES)
        + list(GEOMETRY_COLUMNS)
        + _control_columns(reference_manifest)
        + [axis_col, target_col, "source_curve_id"]
    )


def _write_dataset_csv(path: Path, rows: list[dict[str, float | str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def _write_split_manifest_csv(
    path: Path,
    *,
    n_rows: int,
    val_fraction: float,
    seed: int | None,
) -> int:
    perm, n_val = _split_row_indices(n_rows, val_fraction=val_fraction, seed=seed)
    labels = ["train"] * n_rows
    for index in perm[:n_val]:
        labels[int(index)] = "validation"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "split", "seed", "val_fraction"])
        writer.writeheader()
        for row_id, split in enumerate(labels):
            writer.writerow(
                {
                    "row_id": row_id,
                    "split": split,
                    "seed": "" if seed is None else int(seed),
                    "val_fraction": float(val_fraction),
                }
            )
    return n_val


def _split_row_indices(
    n_rows: int,
    *,
    val_fraction: float,
    seed: int | None,
) -> tuple[np.ndarray, int]:
    if n_rows < 2:
        raise ValueError("Need at least 2 rows to create train/validation split.")
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    indices = np.arange(int(n_rows), dtype=np.int64)
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(int(seed))
    perm = rng.permutation(indices)
    n_val = max(1, int(float(val_fraction) * int(n_rows)))
    if n_val >= int(n_rows):
        n_val = int(n_rows) - 1
    return perm, int(n_val)


def _rows_to_arrays(
    rows: list[dict[str, float | str]],
    *,
    input_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([[float(row[column]) for column in input_cols] for row in rows], dtype=np.float64)
    y = np.asarray([float(row[target_col]) for row in rows], dtype=np.float64)
    return X, y


def _train_smoke_surrogate(
    rows: list[dict[str, float | str]],
    *,
    input_cols: list[str],
    target_col: str,
    out_path: Path,
    width: int,
    depth: int,
    batch_size: int,
    max_epoch: int,
    seed: int | None,
    val_fraction: float,
    report_path: Path,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for DNN smoke training.")
    random.seed(int(seed) if seed is not None else 0)
    np.random.seed(int(seed) if seed is not None else 0)
    torch.manual_seed(int(seed) if seed is not None else 0)

    X_raw, y_raw = _rows_to_arrays(rows, input_cols=input_cols, target_col=target_col)
    x_mu = X_raw.mean(axis=0)
    x_sd = X_raw.std(axis=0)
    x_sd[x_sd == 0.0] = 1.0
    y_mu = np.asarray([y_raw.mean()], dtype=np.float64)
    y_sd = np.asarray([y_raw.std()], dtype=np.float64)
    y_sd[y_sd == 0.0] = 1.0

    X_norm = (X_raw - x_mu) / x_sd
    y_norm = ((y_raw.reshape(-1, 1) - y_mu) / y_sd).astype(np.float32)

    perm, n_val = _split_row_indices(len(rows), val_fraction=val_fraction, seed=seed)
    X_norm = X_norm[perm]
    y_norm = y_norm[perm]
    y_phys = y_raw[perm]

    Xv = torch.as_tensor(X_norm[:n_val], dtype=torch.float32)
    yv = torch.as_tensor(y_norm[:n_val], dtype=torch.float32)
    Xt = torch.as_tensor(X_norm[n_val:], dtype=torch.float32)
    yt = torch.as_tensor(y_norm[n_val:], dtype=torch.float32)
    yv_phys = y_phys[:n_val]

    loader_generator = None
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=min(batch_size, len(Xt)),
        shuffle=True,
        generator=loader_generator,
    )

    model = MLP(input_dims=len(input_cols), output_dims=1, hl_dims=[width] * depth)
    model.apply(init_weights)
    model, train_losses, valid_losses = train_model(
        model,
        loader,
        Xv,
        yv,
        lr=5e-4,
        max_epoch=max_epoch,
        info_every=max(max_epoch, 1),
    )
    model.eval()
    with torch.inference_mode():
        yv_pred_norm = model(Xv).detach().cpu().numpy().reshape(-1)
    yv_pred_phys = np.maximum(0.0, (yv_pred_norm * y_sd[0]) + y_mu[0])
    val_rmse_phys = float(np.sqrt(np.mean(np.square(yv_pred_phys - yv_phys))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_states(
        model,
        xshift=x_mu.tolist(),
        xscale=x_sd.tolist(),
        yshift=y_mu.tolist(),
        yscale=y_sd.tolist(),
        path=str(out_path),
    )
    result = {
        "status": "trained",
        "train_loss": float(train_losses[-1]),
        "val_loss": float(valid_losses[-1]),
        "out": str(out_path),
        "n_train": int(len(Xt)),
        "n_val": int(len(Xv)),
        "val_rmse_phys": val_rmse_phys,
        "seed": None if seed is None else int(seed),
        "val_fraction": float(val_fraction),
        "width": int(width),
        "depth": int(depth),
        "batch_size": int(min(batch_size, len(Xt))),
        "lr": 5e-4,
        "max_epoch": int(max_epoch),
    }
    _write_json(report_path, result)
    return result


def _predict_rows(
    model_path: Path,
    rows: list[dict[str, float | str]],
    *,
    input_cols: list[str],
    target_col: str,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required for DNN smoke reload evaluation.")
    model, xshift, xscale, yshift, yscale = load_model_states(str(model_path))
    model.eval()
    xshift_arr = np.asarray(xshift, dtype=np.float64)
    xscale_arr = np.asarray(xscale, dtype=np.float64)
    yshift_arr = np.asarray(yshift, dtype=np.float64)
    yscale_arr = np.asarray(yscale, dtype=np.float64)
    X_raw = np.asarray([[float(row[column]) for column in input_cols] for row in rows], dtype=np.float64)
    X_norm = (X_raw - xshift_arr) / xscale_arr
    with torch.inference_mode():
        pred_norm = model(torch.as_tensor(X_norm, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    return np.maximum(0.0, (pred_norm * yscale_arr[0]) + yshift_arr[0])


def _dry_run_training_payload(rows: list[dict[str, float | str]], *, val_fraction: float) -> dict[str, Any]:
    _perm, n_val = _split_row_indices(len(rows), val_fraction=val_fraction, seed=DEFAULT_SEED)
    return {
        "status": "skipped_missing_dependency",
        "missing_dependency": "torch",
        "reason": "Torch is unavailable in the current workspace, so DNN train/load/evaluate was not executed.",
        "n_train": int(len(rows) - n_val),
        "n_val": int(n_val),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        reference_manifest = _build_reference_manifest(args)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    axis_col, target_col = _axis_target_columns(str(reference_manifest["experiment"]))
    input_cols = list(PARAMETER_NAMES) + list(GEOMETRY_COLUMNS) + _control_columns(reference_manifest) + [axis_col]
    smoke_rows = _build_smoke_rows(
        reference_manifest,
        num_curves=args.num_curves,
        points_per_curve=args.points_per_curve,
    )
    dataset_columns = _ordered_dataset_columns(reference_manifest)

    reference_manifest_path = output_root / "gv_reference_manifest.json"
    dataset_path = output_root / "gv_surrogate_smoke_dataset.csv"
    split_manifest_path = output_root / "gv_surrogate_smoke_split_manifest.csv"
    model_path = output_root / "gv_surrogate_dnn_smoke.pkl"
    training_report_path = output_root / "gv_surrogate_dnn_training_report.json"
    workflow_manifest_path = output_root / "gv_dnn_surrogate_smoke_manifest.json"
    smoke_report_path = output_root / "gv_dnn_surrogate_smoke_report.json"

    _write_json(reference_manifest_path, reference_manifest)
    _write_dataset_csv(dataset_path, smoke_rows, dataset_columns)
    split_n_val = _write_split_manifest_csv(
        split_manifest_path,
        n_rows=len(smoke_rows),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    if torch is not None:
        training_result = _train_smoke_surrogate(
            smoke_rows,
            input_cols=input_cols,
            target_col=target_col,
            out_path=model_path,
            width=args.width,
            depth=args.depth,
            batch_size=args.batch_size,
            max_epoch=args.max_epoch,
            seed=args.seed,
            val_fraction=args.val_fraction,
            report_path=training_report_path,
        )
        perm, n_val = _split_row_indices(len(smoke_rows), val_fraction=args.val_fraction, seed=args.seed)
        val_rows = [smoke_rows[int(index)] for index in perm[:n_val]]
        val_predictions = _predict_rows(model_path, val_rows, input_cols=input_cols, target_col=target_col)
        val_observed = np.asarray([float(row[target_col]) for row in val_rows], dtype=np.float64)
        val_rmse_reload = float(np.sqrt(np.mean(np.square(val_predictions - val_observed))))
        reload_validation = {
            "status": "passed",
            "val_rmse_phys": val_rmse_reload,
            "passed_val_replay_tol_1e_6": abs(val_rmse_reload - training_result["val_rmse_phys"]) <= 1e-6,
            "observed_sample": val_observed[:3].round(6).tolist(),
            "predicted_sample": np.round(val_predictions[:3], 6).tolist(),
        }
        execution_mode = "train_load_evaluate"
    else:
        training_result = _dry_run_training_payload(smoke_rows, val_fraction=args.val_fraction)
        _write_json(training_report_path, training_result)
        n_val = training_result["n_val"]
        reload_validation = {
            "status": "not_run",
            "passed_val_replay_tol_1e_6": False,
            "reason": "Torch is unavailable in the current workspace.",
        }
        execution_mode = "dry_run_manifest_only"

    runtime_manifest = reference_manifest.get("runtime_manifest")
    runtime_stage = {}
    input_manifests = dict(reference_manifest.get("input_manifests", {}))
    if isinstance(runtime_manifest, dict):
        runtime_stage = {
            "dataset_id": runtime_manifest.get("dataset_id"),
            "control_id": runtime_manifest.get("control_id"),
            "runtime_package": runtime_manifest.get("runtime_package"),
            "experimental": runtime_manifest.get("experimental", False),
            "known_issues": runtime_manifest.get("known_issues", []),
            "output_root": runtime_manifest.get("output_root"),
            "work_dir": runtime_manifest.get("work_dir"),
        }
        manifest_path = runtime_manifest.get("manifest_path")
        if manifest_path is not None:
            input_manifests.setdefault("runtime_dry_run", str(Path(str(manifest_path)).resolve()))

    workflow_manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "workflow": "gv_dnn_surrogate_smoke",
        "surrogate_family": args.backend,
        "execution_mode": execution_mode,
        "structure": reference_manifest["structure"],
        "experiment": reference_manifest["experiment"],
        "geometry": reference_manifest["geometry"],
        "controls": reference_manifest["controls"],
        "reference_kind": reference_manifest["reference_kind"],
        "backend": args.backend,
        "dataset_id": reference_manifest.get("dataset_id"),
        "input_columns": input_cols,
        "target_column": target_col,
        "input_manifests": input_manifests,
        "runtime_stage": runtime_stage,
        "synthetic_fixture": {
            "num_curves": int(args.num_curves),
            "points_per_curve": int(args.points_per_curve),
            "num_rows": int(len(smoke_rows)),
            "dataset_csv": str(dataset_path),
            "split_manifest": str(split_manifest_path),
        },
        "training": {"report": training_result},
        "reload": reload_validation,
        "artifacts": {
            "reference_manifest": str(reference_manifest_path),
            "dataset_csv": str(dataset_path),
            "split_manifest": str(split_manifest_path),
            "artifact_path": str(model_path) if model_path.exists() else None,
            "model_path": str(model_path) if model_path.exists() else None,
            "training_report": str(training_report_path),
            "smoke_report": str(smoke_report_path),
        },
        "upstream_contract": reference_manifest.get("upstream_contract", _upstream_contract()),
    }
    _write_json(workflow_manifest_path, workflow_manifest)

    smoke_report = {
        "status": "passed" if torch is not None else "dry-run",
        "execution_mode": execution_mode,
        "structure": reference_manifest["structure"],
        "experiment": reference_manifest["experiment"],
        "geometry": reference_manifest["geometry"],
        "controls": reference_manifest["controls"],
        "reference_kind": reference_manifest["reference_kind"],
        "backend": args.backend,
        "dataset_id": reference_manifest.get("dataset_id"),
        "row_count": int(len(smoke_rows)),
        "validation_row_count": int(n_val),
        "split_validation_row_count": int(split_n_val),
        "training": training_result,
        "reload_validation": reload_validation,
        "artifacts": workflow_manifest["artifacts"],
        "upstream_contract": workflow_manifest["upstream_contract"],
    }
    _write_json(smoke_report_path, smoke_report)
    print(f"GV DNN smoke manifest: {workflow_manifest_path}")
    print(f"GV DNN smoke report: {smoke_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
