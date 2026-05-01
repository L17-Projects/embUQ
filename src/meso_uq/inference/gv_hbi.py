from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from meso_uq.experiments import ExperimentSpec
from meso_uq.structures import get_structure
from meso_uq.workflow_acceleration import (
    active_hierarchical_variable_names,
    phase1_prior_specs,
    phase2_hyperprior_specs,
)

GV_HBI_EXPERIMENTAL_FLAG = "MESOUQ_ENABLE_EXPERIMENTAL_GV_HBI"
GV_PHASE1_SETUP_MANIFEST = "gv_phase1_setup_manifest.json"
GV_PHASE1_EXECUTION_MANIFEST = "gv_phase1_execution_manifest.json"
GV_HBI_SETUP_SCHEMA_VERSION = 1
GV_PHASE1_STRUCTURE = "gv"
GV_PHASE1_CALIBRATED_PARAMETERS = ("ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c")
GV_PHASE1_NUISANCE_PARAMETERS = ("sigma",)
GV_PHASE1_NOISE_MODEL = {"kind": "multiplicative", "parameter": "sigma"}


def _require_structure_qualified_dataset_id(dataset_id: str) -> None:
    parts = str(dataset_id).split(":")
    if len(parts) < 4 or parts[0] != GV_PHASE1_STRUCTURE or any(not part for part in parts[:4]):
        raise ValueError(
            "GV dataset identifiers must be structure-qualified and follow '<gv>:<experiment>:<geometry>:<control>'."
        )


def _require_gv_calibrated_contract() -> tuple[str, ...]:
    calibrated = tuple(get_structure("gv").parameter_contract.calibrated_names)
    if calibrated != GV_PHASE1_CALIBRATED_PARAMETERS:
        raise ValueError(
            "GV calibrated contract mismatch. Expected exact order: "
            + ", ".join(GV_PHASE1_CALIBRATED_PARAMETERS)
        )
    return calibrated


def _require_gv_noisy_model(payload: Mapping[str, Any], *, label: str) -> dict[str, str]:
    noise_model = payload.get("noise_model")
    if noise_model is None:
        return dict(GV_PHASE1_NOISE_MODEL)
    if not isinstance(noise_model, Mapping):
        raise ValueError(
            f"{label} requires multiplicative sigma noise semantics: {GV_PHASE1_NOISE_MODEL}"
        )
    if (
        noise_model.get("kind") != GV_PHASE1_NOISE_MODEL["kind"]
        or noise_model.get("parameter") != GV_PHASE1_NOISE_MODEL["parameter"]
    ):
        raise ValueError(
            f"{label} requires multiplicative sigma noise semantics: {GV_PHASE1_NOISE_MODEL}"
        )
    return dict(GV_PHASE1_NOISE_MODEL)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _enabled_by(config: Mapping[str, Any]) -> str | None:
    if _as_bool(os.environ.get(GV_HBI_EXPERIMENTAL_FLAG)):
        return f"env:{GV_HBI_EXPERIMENTAL_FLAG}"
    experimental = config.get("experimental")
    if isinstance(experimental, Mapping) and _as_bool(experimental.get("gv_hbi")):
        return "config:experimental.gv_hbi"
    if _as_bool(config.get("experimental_gv_hbi")):
        return "config:experimental_gv_hbi"
    return None


def _require_experimental_flag(config: Mapping[str, Any]) -> str:
    enabled_by = _enabled_by(config)
    if enabled_by is None:
        raise PermissionError(
            "GV hierarchical inference setup is experimental. Set "
            f"`experimental_gv_hbi: true` in the config or {GV_HBI_EXPERIMENTAL_FLAG}=1."
        )
    return enabled_by


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GV {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"GV {label} must be a JSON object: {path}")
    return payload


def _enabled_experiments(experiments: Iterable[ExperimentSpec]) -> list[ExperimentSpec]:
    return [experiment for experiment in experiments if experiment.enabled]


def _require_gv_only(experiments: Sequence[ExperimentSpec]) -> None:
    structures = sorted({experiment.structure for experiment in experiments})
    if structures != ["gv"]:
        raise ValueError(
            "GV hierarchical inference setup requires only GV experiments. "
            f"Mixed or unsupported structures were selected: {structures}."
        )


def _surrogate_manifest_map(config: Mapping[str, Any]) -> dict[str, str]:
    raw = config.get("gv_surrogate_manifests") or config.get("surrogate_manifests") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("gv_surrogate_manifests must be a mapping from dataset id to manifest path.")
    manifest_map: dict[str, str] = {}
    for key, value in raw.items():
        dataset_id = str(key)
        try:
            _require_structure_qualified_dataset_id(dataset_id)
        except ValueError:
            continue
        manifest_map[dataset_id] = str(value)
    return manifest_map


def _single_surrogate_manifest(config: Mapping[str, Any]) -> str | None:
    value = config.get("gv_surrogate_manifest") or config.get("surrogate_manifest")
    if value is None:
        return None
    return str(value)


def _inline_surrogate_manifest_map(
    config: Mapping[str, Any],
    experiments: Sequence[ExperimentSpec],
) -> dict[str, str]:
    raw_experiments = config.get("experiments")
    if not isinstance(raw_experiments, (list, tuple)):
        return {}
    inline: dict[str, str] = {}
    for experiment in experiments:
        for raw in raw_experiments:
            if not isinstance(raw, Mapping):
                continue
            raw_structure = str(raw.get("structure", config.get("structure", "")))
            if raw_structure != experiment.structure or raw.get("name") != experiment.name:
                continue
            raw_manifest = raw.get("surrogate_manifest") or raw.get("gv_setup_manifest")
            if raw_manifest is None:
                continue
            raw_geometries = raw.get("geometries") or experiment.geometries
            raw_controls = raw.get("controls") or experiment.controls
            for geometry in raw_geometries:
                for control in raw_controls:
                    dataset_id = experiment.dataset_name(str(geometry), control=str(control))
                    inline[dataset_id] = str(raw_manifest)
    return inline


def _dataset_manifest_path(
    config: Mapping[str, Any],
    *,
    dataset_id: str,
    dataset_count: int,
    inline_manifest_map: Mapping[str, str],
) -> str:
    _require_structure_qualified_dataset_id(dataset_id)
    if dataset_id in inline_manifest_map:
        return inline_manifest_map[dataset_id]
    manifest_map = _surrogate_manifest_map(config)
    if dataset_id in manifest_map:
        return manifest_map[dataset_id]
    single_manifest = _single_surrogate_manifest(config)
    if single_manifest is not None and dataset_count == 1:
        return single_manifest
    raise FileNotFoundError(
        "Missing GV surrogate manifest selection for dataset "
        f"'{dataset_id}'. Provide gv_surrogate_manifests[dataset_id] or a single "
        "gv_surrogate_manifest for one-dataset setup."
    )


def _validate_gv_control_values(
    *,
    controls: Any,
    experiment_name: str,
    label: str,
) -> dict[str, float]:
    if not isinstance(controls, Mapping):
        raise ValueError(f"{label} controls must be a mapping.")
    experiment_spec = get_structure("gv").get_experiment(experiment_name, include_experimental=True)
    expected = experiment_spec.control_names
    unknown = sorted(set(controls) - set(expected))
    if unknown:
        raise ValueError(
            f"{label} declares unknown GV controls for experiment '{experiment_name}': "
            + ", ".join(unknown)
        )
    missing = [name for name in expected if name not in controls]
    if missing:
        raise ValueError(
            f"{label} is missing GV controls for experiment '{experiment_name}': "
            + ", ".join(missing)
        )
    return {name: float(controls[name]) for name in expected}


def _validate_surrogate_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    experiment: ExperimentSpec,
    geometry: str,
    control: str,
    dataset_id: str,
) -> dict[str, Any]:
    if payload.get("structure") != "gv":
        raise ValueError(f"GV surrogate manifest must declare structure='gv': {manifest_path}")
    if payload.get("experiment") != experiment.name:
        raise ValueError(
            f"GV surrogate manifest experiment mismatch for {dataset_id}: "
            f"{payload.get('experiment')!r} != {experiment.name!r}"
        )
    if payload.get("geometry") != geometry:
        raise ValueError(
            f"GV surrogate manifest geometry mismatch for {dataset_id}: "
            f"{payload.get('geometry')!r} != {geometry!r}"
        )
    manifest_dataset_id = payload.get("dataset_id")
    if manifest_dataset_id is not None and manifest_dataset_id != dataset_id:
        _require_structure_qualified_dataset_id(str(manifest_dataset_id))
        raise ValueError(
            f"GV surrogate manifest dataset mismatch: {manifest_dataset_id!r} != {dataset_id!r}"
        )
    backend = payload.get("backend") or payload.get("surrogate_family")
    if backend != "dnn":
        raise ValueError(f"GV hierarchical inference requires a DNN surrogate manifest: {manifest_path}")
    validated_controls = _validate_gv_control_values(
        controls=payload.get("controls"),
        experiment_name=experiment.name,
        label="GV surrogate manifest",
    )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"GV surrogate manifest is missing artifacts: {manifest_path}")
    artifact_value = artifacts.get("artifact_path") or artifacts.get("model_path")
    if not artifact_value:
        raise FileNotFoundError(f"Missing GV surrogate artifact path in manifest: {manifest_path}")
    artifact_path = _resolve_repo_path(manifest_path.parent, str(artifact_value))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing GV surrogate artifact for dataset '{dataset_id}': {artifact_path}")
    reference_manifest_value = artifacts.get("reference_manifest")
    if not reference_manifest_value:
        raise FileNotFoundError(f"Missing GV reference manifest path in surrogate manifest: {manifest_path}")
    reference_manifest_path = _resolve_repo_path(manifest_path.parent, str(reference_manifest_value))
    if not reference_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing GV reference manifest for dataset '{dataset_id}': {reference_manifest_path}"
        )

    return {
        "manifest": str(manifest_path),
        "artifact": str(artifact_path),
        "backend": "dnn",
        "workflow": payload.get("workflow"),
        "control_values": validated_controls,
        "reference_manifest": str(reference_manifest_path),
        "dataset_csv": artifacts.get("dataset_csv"),
        "training_report": artifacts.get("training_report"),
        "control": control,
    }


def _resolve_optional_manifest_artifact(
    manifest_path: Path,
    artifact_value: Any,
) -> str | None:
    if artifact_value in (None, ""):
        return None
    return str(_resolve_repo_path(manifest_path.parent, str(artifact_value)))


def _validate_reference_manifest(
    payload: Mapping[str, Any],
    *,
    experiment: str,
    geometry: str,
    dataset_id: str,
    control: str,
    expected_controls: Mapping[str, float],
) -> dict[str, Any]:
    _require_structure_qualified_dataset_id(dataset_id)
    noise_model = _require_gv_noisy_model(payload, label="GV reference manifest")
    if payload.get("structure") != "gv":
        raise ValueError("GV reference manifest must declare structure='gv'.")
    if payload.get("experiment") != experiment:
        raise ValueError(
            f"GV reference manifest experiment mismatch for {dataset_id}: "
            f"{payload.get('experiment')!r} != {experiment!r}"
        )
    if payload.get("geometry") != geometry:
        raise ValueError(
            f"GV reference manifest geometry mismatch for {dataset_id}: "
            f"{payload.get('geometry')!r} != {geometry!r}"
        )
    manifest_dataset_id = payload.get("dataset_id")
    if manifest_dataset_id is not None:
        _require_structure_qualified_dataset_id(manifest_dataset_id)
        if manifest_dataset_id != dataset_id:
            raise ValueError(f"GV reference manifest dataset mismatch: {manifest_dataset_id!r} != {dataset_id!r}")
    controls = _validate_gv_control_values(
        controls=payload.get("controls"),
        experiment_name=experiment,
        label="GV reference manifest",
    )
    if controls != {name: float(value) for name, value in expected_controls.items()}:
        raise ValueError(
            f"GV reference manifest controls mismatch for {dataset_id}: {controls!r} != {dict(expected_controls)!r}"
        )
    calibrated = payload.get("calibrated_parameter_names")
    if calibrated is not None:
        expected_calibrated = list(_require_gv_calibrated_contract())
        if list(calibrated) != expected_calibrated:
            raise ValueError(
                "GV reference manifest calibrated_parameter_names must match the GV parameter contract."
            )
        control_overlap = sorted(set(controls).intersection(calibrated))
        if control_overlap:
            raise ValueError(
                "GV reference manifest must keep controls separate from calibrated parameters: "
                + ", ".join(control_overlap)
            )
    return {
        "dataset_id": dataset_id,
        "reference_kind": payload.get("reference_kind"),
        "noise_model": noise_model,
        "control_label": control,
        "controls": controls,
    }


def _require_existing_path(path: str | None, *, label: str) -> str:
    if path is None:
        raise FileNotFoundError(f"Missing GV {label} path in surrogate manifest.")
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing GV {label}: {resolved}")
    return str(resolved)


def _build_execution_artifact_probe(artifact_path: Path) -> dict[str, Any]:
    try:
        from meso_uq.surrogate.model import load_model_states
    except ModuleNotFoundError as exc:
        return {
            "status": "skipped_missing_dependency",
            "reason": str(exc),
        }

    model, xshift, _xscale, yshift, _yscale = load_model_states(str(artifact_path))
    return {
        "status": "loaded",
        "model_class": type(model).__name__,
        "input_dim": int(len(xshift)),
        "output_dim": int(len(yshift)),
    }


def _read_dataset_rows(dataset_csv_path: Path) -> list[dict[str, str]]:
    with dataset_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"GV surrogate dataset CSV has no rows: {dataset_csv_path}")
    return rows


def _select_reference_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    first = rows[0]
    curve_id = first.get("source_curve_id")
    if curve_id is None:
        return list(rows)
    selected = [row for row in rows if row.get("source_curve_id") == curve_id]
    return selected


def _target_column(surrogate_payload: Mapping[str, Any], rows: Sequence[Mapping[str, str]]) -> str:
    target = surrogate_payload.get("target_column")
    if isinstance(target, str) and target:
        return target
    columns = set(rows[0])
    for candidate in ("response", "torsion_response", "shear_response", "buckling_response"):
        if candidate in columns:
            return candidate
    response_columns = sorted(column for column in columns if column.endswith("_response"))
    if response_columns:
        return response_columns[0]
    raise ValueError("GV surrogate manifest must declare target_column or dataset CSV must include a response column.")


def _input_columns(
    surrogate_payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    *,
    target_column: str,
) -> list[str]:
    configured = surrogate_payload.get("input_columns")
    if isinstance(configured, (list, tuple)):
        columns = [str(column) for column in configured]
    else:
        excluded = {target_column, "source_curve_id"}
        columns = [column for column in rows[0] if column not in excluded]
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError("GV surrogate dataset CSV is missing input columns: " + ", ".join(missing))
    return columns


def _axis_column(
    input_columns: Sequence[str],
    *,
    controls: Mapping[str, float],
) -> str:
    calibrated = set(get_structure("gv").parameter_contract.calibrated_names)
    geometry = {"radius", "height"}
    control_names = set(controls)
    candidates = [name for name in input_columns if name not in calibrated | geometry | control_names]
    if len(candidates) != 1:
        raise ValueError(
            "GV Phase 1 DNN execution needs exactly one observable-axis input column; "
            f"found {candidates!r}."
        )
    return candidates[0]


def _reference_series_from_dataset(
    surrogate_payload: Mapping[str, Any],
    dataset_csv_path: Path,
    *,
    controls: Mapping[str, float],
) -> dict[str, Any]:
    rows = _select_reference_rows(_read_dataset_rows(dataset_csv_path))
    target = _target_column(surrogate_payload, rows)
    inputs = _input_columns(surrogate_payload, rows, target_column=target)
    axis = _axis_column(inputs, controls=controls)
    return {
        "rows": rows,
        "input_columns": inputs,
        "axis_column": axis,
        "target_column": target,
        "reference_points": [float(row[axis]) for row in rows],
        "reference_data": [float(row[target]) for row in rows],
    }


def _geometry_parameters(reference_manifest_payload: Mapping[str, Any]) -> dict[str, float]:
    geometry_spec = reference_manifest_payload.get("geometry_spec")
    parameters = geometry_spec.get("parameters") if isinstance(geometry_spec, Mapping) else None
    if not isinstance(parameters, Mapping):
        raise ValueError("GV reference manifest must include geometry_spec.parameters for DNN execution.")
    missing = [name for name in ("radius", "height") if name not in parameters]
    if missing:
        raise ValueError("GV reference manifest geometry parameters are missing: " + ", ".join(missing))
    return {name: float(parameters[name]) for name in ("radius", "height")}


def _load_gv_dnn_model_state(artifact_path: Path) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from meso_uq.surrogate.model import load_model_states

    model, xshift, xscale, yshift, yscale = load_model_states(str(artifact_path))
    return (
        model,
        np.asarray(xshift, dtype=np.float64),
        np.asarray(xscale, dtype=np.float64),
        np.asarray(yshift, dtype=np.float64),
        np.asarray(yscale, dtype=np.float64),
    )


def _predict_gv_dnn(model_state: tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray], x_raw: np.ndarray) -> np.ndarray:
    import torch

    model, xshift, xscale, yshift, yscale = model_state
    model.eval()
    x_norm = (x_raw - xshift) / xscale
    with torch.inference_mode():
        pred_norm = model(torch.as_tensor(x_norm, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    return np.maximum(0.0, pred_norm * yscale[0] + yshift[0])


def _build_gv_dnn_input_matrix(
    *,
    sample_parameters: Mapping[str, float],
    reference_rows: Sequence[Mapping[str, str]],
    input_columns: Sequence[str],
    controls: Mapping[str, float],
    geometry_parameters: Mapping[str, float],
) -> np.ndarray:
    calibrated = set(get_structure("gv").parameter_contract.calibrated_names)
    rows: list[list[float]] = []
    for reference_row in reference_rows:
        values: list[float] = []
        for column in input_columns:
            if column in calibrated:
                values.append(float(sample_parameters[column]))
            elif column == "d0" or column == "sigma":
                raise ValueError(f"GV surrogate input column '{column}' is not a calibrated material/control input.")
            elif column in geometry_parameters:
                values.append(float(geometry_parameters[column]))
            elif column in controls:
                values.append(float(controls[column]))
            elif column in reference_row:
                values.append(float(reference_row[column]))
            else:
                raise ValueError(f"Cannot resolve GV surrogate input column '{column}'.")
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def _evaluate_gv_dnn_reference(sample_data: MutableMapping[str, Any], context: Mapping[str, Any]) -> None:
    variable_names = list(context["variable_names"])
    parameter_values = dict(zip(variable_names, sample_data["Parameters"]))
    x_raw = _build_gv_dnn_input_matrix(
        sample_parameters=parameter_values,
        reference_rows=context["reference_rows"],
        input_columns=context["input_columns"],
        controls=context["controls"],
        geometry_parameters=context["geometry_parameters"],
    )
    predictions = _predict_gv_dnn(context["model_state"], x_raw)
    if "d0" in parameter_values:
        predictions = predictions + float(parameter_values["d0"])
    sample_data["Reference Evaluations"] = predictions.tolist()


def _gv_dataset_entries(
    config: Mapping[str, Any],
    experiments: Sequence[ExperimentSpec],
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    requested: list[tuple[ExperimentSpec, str, str, str]] = []
    inline_manifest_map = _inline_surrogate_manifest_map(config, experiments)
    for experiment in experiments:
        structure_spec = get_structure("gv")
        structure_spec.get_experiment(
            experiment.name,
            include_experimental=bool(config.get("include_experimental", False)),
        )
        for geometry in experiment.geometries:
            for control in experiment.controls:
                requested.append(
                    (
                        experiment,
                        geometry,
                        control,
                        experiment.dataset_name(geometry, control=control),
                    )
                )

    entries: list[dict[str, Any]] = []
    for experiment, geometry, control, dataset_id in requested:
        _require_structure_qualified_dataset_id(dataset_id)
        manifest_path = _resolve_repo_path(
            repo_root,
            _dataset_manifest_path(
                config,
                dataset_id=dataset_id,
                dataset_count=len(requested),
                inline_manifest_map=inline_manifest_map,
            ),
        )
        surrogate_payload = _load_json(manifest_path, label="surrogate manifest")
        surrogate = _validate_surrogate_manifest(
            surrogate_payload,
            manifest_path=manifest_path,
            experiment=experiment,
            geometry=geometry,
            control=control,
            dataset_id=dataset_id,
        )
        entries.append(
            {
                "structure": "gv",
                "experiment": experiment.name,
                "experiment_id": experiment.experiment_id,
                "geometry": geometry,
                "control": control,
                "dataset_id": dataset_id,
                "surrogate": surrogate,
            }
        )
    return entries


def _parameter_contract_manifest() -> dict[str, Any]:
    calibrated = _require_gv_calibrated_contract()
    structure = get_structure("gv")
    nuisance = tuple(structure.parameter_contract.nuisance_names)
    for nuisance_name in GV_PHASE1_NUISANCE_PARAMETERS:
        if nuisance_name not in nuisance:
            raise ValueError(
                f"GV nuisance contract is missing required nuisance parameter: {nuisance_name}"
            )
    noise_model = structure.parameter_contract.noise_model
    return {
        "calibrated": list(calibrated),
        "nuisance": list(GV_PHASE1_NUISANCE_PARAMETERS),
        "noise_model": None
        if noise_model is None
        else {
            "kind": noise_model.kind,
            "parameter": noise_model.parameter,
            "description": noise_model.description,
        },
    }


def _bounds_payload(specs: Sequence[tuple[Any, ...]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if len(spec) == 2:
            name, bounds = spec
            rows.append({"name": str(name), "bounds": [float(bounds[0]), float(bounds[1])]})
        elif len(spec) == 3:
            name, mu_bounds, sigma_bounds = spec
            rows.append(
                {
                    "name": str(name),
                    "mu_bounds": [float(mu_bounds[0]), float(mu_bounds[1])],
                    "sigma_bounds": [float(sigma_bounds[0]), float(sigma_bounds[1])],
                }
            )
        else:
            raise ValueError(f"Unsupported bounds spec shape: {spec}")
    return rows


def build_gv_phase1_setup_manifest(
    config: Mapping[str, Any],
    experiments: Iterable[ExperimentSpec],
    *,
    repo_root: str | Path,
    output_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    enabled_by = _require_experimental_flag(config)
    selected = _enabled_experiments(experiments)
    if not selected:
        raise ValueError("No enabled GV experiments were selected for hierarchical inference setup.")
    _require_gv_only(selected)

    if str((config.get("surrogate") or {}).get("backend", "dnn")).lower() != "dnn":
        raise ValueError("GV hierarchical inference supports only the DNN surrogate backend in this tranche.")

    repo_root_path = Path(repo_root).resolve()
    output_root_path = Path(output_root).resolve()
    dataset_entries = _gv_dataset_entries(config, selected, repo_root=repo_root_path)
    phase1_specs = phase1_prior_specs(config)
    phase2_specs = phase2_hyperprior_specs(config)
    variable_names = [name for name, _bounds in phase1_specs]
    gv_structure = get_structure("gv")
    calibrated_names = list(_require_gv_calibrated_contract())
    nuisance_names = list(GV_PHASE1_NUISANCE_PARAMETERS)
    for nuisance_name in nuisance_names:
        if nuisance_name not in gv_structure.parameter_contract.nuisance_names:
            raise ValueError(
                f"GV nuisance contract is missing required nuisance parameter: {nuisance_name}"
            )
    registry_experiments = [
        gv_structure.get_experiment(
            experiment.name,
            include_experimental=bool(config.get("include_experimental", False)),
        )
        for experiment in selected
    ]
    control_names = sorted(
        {control_name for experiment in registry_experiments for control_name in experiment.control_names}
    )
    control_selections = sorted({control for experiment in selected for control in experiment.controls})
    contract_names = set(calibrated_names) | set(nuisance_names) | set(variable_names)
    control_overlap = sorted(set(control_names).intersection(contract_names))
    if control_overlap:
        raise ValueError(
            "GV controls must not appear in calibrated or nuisance variables: "
            + ", ".join(control_overlap)
        )

    return {
        "manifest_schema_version": GV_HBI_SETUP_SCHEMA_VERSION,
        "workflow": "gv_phase1_setup",
        "structure": "gv",
        "experimental": True,
        "enabled_by": enabled_by,
        "config_path": None if config_path is None else str(Path(config_path).resolve()),
        "output_root": str(output_root_path),
        "status": "setup_validated",
        "chain": ["Mirheo", "DNN surrogate", "hierarchical inference"],
        "parameter_contract": _parameter_contract_manifest(),
        "phase1": {
            "problem_type": "Bayesian/Reference",
            "likelihood_model": "Normal",
            "noise_model": dict(GV_PHASE1_NOISE_MODEL),
            "variable_names": variable_names,
            "calibrated_parameter_names": calibrated_names,
            "prior_specs": _bounds_payload(phase1_specs),
        },
        "phase2": {
            "problem_type": "Hierarchical/Psi",
            "pooled_variable_names": active_hierarchical_variable_names(config),
            "hyperprior_specs": _bounds_payload(phase2_specs),
            "pooling_axes": ["experiment", "geometry", "control", "dataset_id"],
        },
        "datasets": dataset_entries,
        "controls": {
            "names": control_names,
            "configured": control_selections,
            "policy": "GV controls are design inputs and are excluded from calibrated vectors.",
        },
    }


def write_gv_phase1_setup_manifest(
    config: Mapping[str, Any],
    experiments: Iterable[ExperimentSpec],
    *,
    repo_root: str | Path,
    output_root: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    output_root_path = Path(output_root).resolve()
    manifest = build_gv_phase1_setup_manifest(
        config,
        experiments,
        repo_root=repo_root,
        output_root=output_root_path,
        config_path=config_path,
    )
    manifest_path = output_root_path / "results_phase_1" / GV_PHASE1_SETUP_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def write_gv_phase1_execution_manifest(
    config: Mapping[str, Any],
    experiments: Iterable[ExperimentSpec],
    *,
    repo_root: str | Path,
    output_root: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    if not _as_bool(config.get("use_surrogate", True)):
        raise ValueError("GV Phase 1 DNN execution seam requires use_surrogate=true.")

    output_root_path = Path(output_root).resolve()
    setup_manifest_path = write_gv_phase1_setup_manifest(
        config,
        experiments,
        repo_root=repo_root,
        output_root=output_root_path,
        config_path=config_path,
    )
    setup_manifest = _load_json(setup_manifest_path, label="phase1 setup manifest")
    datasets = setup_manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise NotImplementedError("GV Phase 1 DNN execution seam currently supports exactly one GV dataset lane.")

    dataset = datasets[0]
    _require_structure_qualified_dataset_id(str(dataset["dataset_id"]))
    surrogate = dataset.get("surrogate")
    if not isinstance(surrogate, Mapping):
        raise ValueError("GV Phase 1 setup manifest is missing surrogate metadata for the selected dataset.")

    setup_manifest_file = Path(str(surrogate["manifest"])).resolve()
    surrogate_payload = _load_json(setup_manifest_file, label="surrogate manifest")
    artifacts = surrogate_payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"GV surrogate manifest is missing artifacts: {setup_manifest_file}")

    reference_manifest_path = Path(_require_existing_path(str(surrogate["reference_manifest"]), label="reference manifest"))
    artifact_path = Path(_require_existing_path(str(surrogate["artifact"]), label="surrogate artifact"))
    dataset_csv_path = _require_existing_path(
        _resolve_optional_manifest_artifact(setup_manifest_file, artifacts.get("dataset_csv")),
        label="surrogate dataset CSV",
    )
    training_report_path = _require_existing_path(
        _resolve_optional_manifest_artifact(setup_manifest_file, artifacts.get("training_report")),
        label="surrogate training report",
    )
    reference_manifest = _validate_reference_manifest(
        _load_json(reference_manifest_path, label="reference manifest"),
        experiment=str(dataset["experiment"]),
        geometry=str(dataset["geometry"]),
        dataset_id=str(dataset["dataset_id"]),
        control=str(dataset["control"]),
        expected_controls=surrogate["control_values"],
    )
    training_report = _load_json(Path(training_report_path), label="training report")
    if not isinstance(training_report, Mapping):
        raise ValueError("GV surrogate training report must be a JSON object.")

    phase1 = setup_manifest["phase1"]
    phase1_noise_model = _require_gv_noisy_model(phase1, label="GV setup phase1")
    execution_manifest = {
        "manifest_schema_version": GV_HBI_SETUP_SCHEMA_VERSION,
        "workflow": "gv_phase1_execution",
        "structure": "gv",
        "experimental": True,
        "enabled_by": setup_manifest["enabled_by"],
        "status": "execution_seam_emitted",
        "config_path": setup_manifest.get("config_path"),
        "output_root": setup_manifest["output_root"],
        "execution_model": "single_lane_dnn_seam",
        "phase1": {
            "variable_names": list(phase1["variable_names"]),
            "noise_model": phase1_noise_model,
        },
        "controls": {
            "names": list(setup_manifest["controls"]["names"]),
            "configured": list(setup_manifest["controls"]["configured"]),
            "policy": setup_manifest["controls"]["policy"],
            "fixed_outside_inferred_variables": True,
        },
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "experiment": dataset["experiment"],
            "experiment_id": dataset["experiment_id"],
            "geometry": dataset["geometry"],
            "control": dataset["control"],
            "control_values": dict(surrogate.get("control_values", {})),
            "surrogate_backend": surrogate["backend"],
        },
        "provenance": {
            "setup_manifest": str(setup_manifest_path),
            "surrogate_manifest": str(setup_manifest_file),
            "reference_manifest": str(reference_manifest_path),
            "surrogate_artifact": str(artifact_path),
            "dataset_csv": dataset_csv_path,
            "training_report": training_report_path,
            "noise_model": reference_manifest["noise_model"],
        },
        "artifact_probe": _build_execution_artifact_probe(artifact_path),
        "runtime": {
            "korali_invoked": False,
            "execution_scope": "adapter_only",
            "full_hbi_completed": False,
            "reason": "This tranche emits a bounded GV Phase 1 DNN execution seam without entering the broader Korali HBI runtime.",
        },
    }
    execution_manifest_path = output_root_path / "results_phase_1" / GV_PHASE1_EXECUTION_MANIFEST
    execution_manifest_path.write_text(json.dumps(execution_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return execution_manifest_path


def run_gv_phase1_dnn_execution(
    config: Mapping[str, Any],
    experiments: Iterable[ExperimentSpec],
    *,
    repo_root: str | Path,
    output_root: str | Path,
    korali_module: Any,
    engine: Any,
    config_path: str | Path | None = None,
) -> Path:
    """Run the first bounded GV Phase 1 DNN-backed Korali lane.

    This intentionally remains single-lane and DNN-only. Phase 2/3b pooling is a
    downstream slice; this function only proves that the validated GV surrogate
    and artificial reference can enter a real Bayesian/Reference Korali problem.
    """

    execution_manifest_path = write_gv_phase1_execution_manifest(
        config,
        experiments,
        repo_root=repo_root,
        output_root=output_root,
        config_path=config_path,
    )
    output_root_path = Path(output_root).resolve()
    execution_manifest = _load_json(execution_manifest_path, label="phase1 execution manifest")
    provenance = execution_manifest["provenance"]
    surrogate_payload = _load_json(Path(provenance["surrogate_manifest"]), label="surrogate manifest")
    reference_payload = _load_json(Path(provenance["reference_manifest"]), label="reference manifest")
    dataset = execution_manifest["dataset"]
    controls = {str(name): float(value) for name, value in dataset["control_values"].items()}
    variable_names = list(execution_manifest["phase1"]["variable_names"])
    control_overlap = sorted(set(controls).intersection(variable_names))
    if control_overlap:
        raise ValueError("GV controls must not appear in Phase 1 Korali variables: " + ", ".join(control_overlap))

    reference_series = _reference_series_from_dataset(
        surrogate_payload,
        Path(provenance["dataset_csv"]),
        controls=controls,
    )
    geometry_parameters = _geometry_parameters(reference_payload)
    model_state = _load_gv_dnn_model_state(Path(provenance["surrogate_artifact"]))
    context = {
        "variable_names": variable_names,
        "reference_rows": reference_series["rows"],
        "input_columns": reference_series["input_columns"],
        "controls": controls,
        "geometry_parameters": geometry_parameters,
        "model_state": model_state,
    }

    experiment = korali_module.Experiment()
    experiment["Problem"]["Type"] = "Bayesian/Reference"
    experiment["Problem"]["Likelihood Model"] = "Normal"
    experiment["Problem"]["Reference Data"] = reference_series["reference_data"]
    experiment["Problem"]["Computational Model"] = (
        lambda sample_data, ctx=context: _evaluate_gv_dnn_reference(sample_data, ctx)
    )
    experiment["Solver"]["Type"] = "Sampler/TMCMC"
    experiment["Solver"]["Population Size"] = int(config["pop_size"])
    burn_in_value = config.get("phase1_burn_in")
    if burn_in_value is None:
        burn_in_value = config.get("hbi_burn_in", 0)
    experiment["Solver"]["Burn In"] = int(burn_in_value)
    experiment["Solver"]["Target Coefficient Of Variation"] = float(config["target_cov"])
    experiment["Solver"]["Covariance Scaling"] = float(config["covariance_scaling"])
    if int(config["max_gen"]) > 0:
        experiment["Solver"]["Termination Criteria"]["Max Generations"] = int(config["max_gen"])

    for idx, (name, bounds) in enumerate(phase1_prior_specs(config)):
        experiment["Distributions"][idx]["Name"] = f"Prior {name}"
        experiment["Distributions"][idx]["Type"] = "Univariate/Uniform"
        experiment["Distributions"][idx]["Minimum"] = float(bounds[0])
        experiment["Distributions"][idx]["Maximum"] = float(bounds[1])
        experiment["Variables"][idx]["Name"] = "[Sigma]" if name == "sigma" else name
        experiment["Variables"][idx]["Prior Distribution"] = f"Prior {name}"

    experiment_root = output_root_path / "results_phase_1" / str(dataset["dataset_id"])
    experiment["File Output"]["Frequency"] = 1
    experiment["File Output"]["Use Multiple Files"] = False
    experiment["File Output"]["Path"] = str(experiment_root)
    experiment["Console Output"]["Frequency"] = 1
    experiment["Console Output"]["Verbosity"] = "Detailed"
    experiment["Store Sample Information"] = True

    engine.run([experiment])

    latest_path = experiment_root / "latest"
    execution_manifest["status"] = "phase1_korali_completed"
    execution_manifest["execution_model"] = "single_lane_dnn_korali"
    execution_manifest["reference"] = {
        "axis_column": reference_series["axis_column"],
        "target_column": reference_series["target_column"],
        "num_points": len(reference_series["reference_data"]),
    }
    execution_manifest["runtime"] = {
        "korali_invoked": True,
        "execution_scope": "single_lane_phase1",
        "phase1_completed": True,
        "full_hbi_completed": False,
        "state_path": str(experiment_root),
        "latest_state": str(latest_path),
        "latest_state_exists": latest_path.exists(),
    }
    execution_manifest_path.write_text(json.dumps(execution_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return execution_manifest_path


__all__ = [
    "GV_HBI_EXPERIMENTAL_FLAG",
    "GV_PHASE1_EXECUTION_MANIFEST",
    "GV_HBI_SETUP_SCHEMA_VERSION",
    "GV_PHASE1_SETUP_MANIFEST",
    "build_gv_phase1_setup_manifest",
    "run_gv_phase1_dnn_execution",
    "write_gv_phase1_execution_manifest",
    "write_gv_phase1_setup_manifest",
]
