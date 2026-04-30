from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    return {str(key): str(value) for key, value in raw.items()}


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
    if manifest_dataset_id is not None and manifest_dataset_id != dataset_id:
        raise ValueError(f"GV reference manifest dataset mismatch: {manifest_dataset_id!r} != {dataset_id!r}")
    noise_model = payload.get("noise_model")
    if noise_model is not None and noise_model != {"kind": "multiplicative", "parameter": "sigma"}:
        raise ValueError(
            "GV Phase 1 DNN execution seam requires multiplicative sigma noise semantics in the reference manifest."
        )
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
        expected_calibrated = list(get_structure("gv").parameter_contract.calibrated_names)
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
        "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
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
    structure = get_structure("gv")
    noise_model = structure.parameter_contract.noise_model
    return {
        "calibrated": list(structure.parameter_contract.calibrated_names),
        "nuisance": list(structure.parameter_contract.nuisance_names),
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
    calibrated_names = list(gv_structure.parameter_contract.calibrated_names)
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
    control_overlap = sorted(set(control_names).intersection(variable_names))
    if control_overlap:
        raise ValueError(
            "GV controls must not appear in calibrated or nuisance inference variables: "
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
            "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
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
    _load_json(Path(training_report_path), label="training report")

    phase1 = setup_manifest["phase1"]
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
            "noise_model": {"kind": "multiplicative", "parameter": "sigma"},
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


__all__ = [
    "GV_HBI_EXPERIMENTAL_FLAG",
    "GV_PHASE1_EXECUTION_MANIFEST",
    "GV_HBI_SETUP_SCHEMA_VERSION",
    "GV_PHASE1_SETUP_MANIFEST",
    "build_gv_phase1_setup_manifest",
    "write_gv_phase1_execution_manifest",
    "write_gv_phase1_setup_manifest",
]
