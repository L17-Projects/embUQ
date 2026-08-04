#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.config.models import InferenceConfig  # noqa: E402
from meso_uq.experiments import ExperimentSpec, load_experiments  # noqa: E402
from meso_uq.inference.emb_parameterization import (  # noqa: E402
    DIRECT_KA_KB_SURROGATE_PARAMETERIZATION,
    adapt_batch_sample_for_legacy_surrogate,
    resolve_direct_compression_surrogate_surface,
    surrogate_parameterization_for_experiment,
)
from meso_uq.inference.emb_resonance import (  # noqa: E402
    compute_emb_resonance_batch,
    preflight_emb_resonance_config,
    preload_emb_resonance,
)
from meso_uq.platforms.site_selector import resolve_hpc_site  # noqa: E402
from replay_provenance import (  # noqa: E402
    load_materialization_binding,
    runtime_provenance,
)

SCHEMA_VERSION = "mesouq.uq_emb.forward_canary.v1"
_BATCH_FRACTIONS = (0.4, 0.5, 0.6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bounds(
    config: Mapping[str, Any],
    experiment: ExperimentSpec,
    diameter_um: float,
    name: str,
) -> tuple[float, float]:
    if name in {"ka", "kb"}:
        override = experiment.phase1_prior_overrides(diameter_um).get(name)
        values = override if override is not None else config.get(f"prior_{name}")
    elif name == "d0":
        values = experiment.prior_d0 or config.get("prior_d0")
    elif name == "sigma":
        values = experiment.prior_sigma or config.get("prior_sigma")
    else:  # pragma: no cover - internal contract
        raise ValueError(f"Unsupported parameter {name!r}.")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(
            f"Missing two-value prior_{name} for {experiment.dataset_name(diameter_um)}."
        )
    lower, upper = (float(value) for value in values)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(
            f"Invalid prior_{name}={values!r} for {experiment.dataset_name(diameter_um)}."
        )
    return lower, upper


def _parameter_batch(
    config: Mapping[str, Any],
    experiment: ExperimentSpec,
    diameter_um: float,
) -> np.ndarray:
    parameter_bounds = [
        _bounds(config, experiment, diameter_um, name)
        for name in ("ka", "kb", "d0", "sigma")
    ]
    return np.asarray(
        [
            [lower + fraction * (upper - lower) for lower, upper in parameter_bounds]
            for fraction in _BATCH_FRACTIONS
        ],
        dtype=np.float32,
    )


def _summarize_batch(
    sample: Mapping[str, Any],
    *,
    expected_rows: int,
    expected_columns: int,
) -> dict[str, Any]:
    predictions = np.asarray(sample.get("Batch Reference Evaluations"), dtype=np.float64)
    standard_deviations = np.asarray(sample.get("Batch Standard Deviation"), dtype=np.float64)
    expected_shape = (expected_rows, expected_columns)
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match expected {expected_shape}."
        )
    if standard_deviations.shape != expected_shape:
        raise ValueError(
            "Standard-deviation shape "
            f"{standard_deviations.shape} does not match expected {expected_shape}."
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Forward predictions contain non-finite values.")
    if not np.all(np.isfinite(standard_deviations)):
        raise ValueError("Forward standard deviations contain non-finite values.")
    if np.any(standard_deviations <= 0.0):
        raise ValueError("Forward standard deviations must be strictly positive.")
    return {
        "shape": list(expected_shape),
        "predictions": predictions.tolist(),
        "standard_deviations": standard_deviations.tolist(),
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "standard_deviation_min": float(np.min(standard_deviations)),
        "standard_deviation_max": float(np.max(standard_deviations)),
    }


def _mechanical_artifacts(experiment: ExperimentSpec, diameter_um: float) -> list[dict[str, Any]]:
    diameter_label = experiment._lookup_diameter_mapping(
        experiment.diameter_labels, diameter_um
    ) or str(diameter_um)
    trained_dir = experiment.surrogate_dir / f"{diameter_label}um" / "trained"
    model_files = sorted(trained_dir.glob("*_BEST.pkl"))
    if len(model_files) != 1:
        raise FileNotFoundError(
            f"Expected one DNN model under {trained_dir}; found {len(model_files)}."
        )
    return [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in model_files
    ]


def _acoustic_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Acoustic artifact is missing: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Acoustic artifact hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": path.stat().st_size,
    }


def _evaluate_dataset(
    config: Mapping[str, Any],
    experiment: ExperimentSpec,
    diameter_um: float,
    *,
    device: str,
) -> dict[str, Any]:
    reference_points = experiment.get_reference_points(diameter_um)
    reference_data = experiment.get_reference_data(diameter_um)
    if len(reference_points) != len(reference_data) or not reference_points:
        raise ValueError(
            f"Invalid reference data for {experiment.dataset_name(diameter_um)}: "
            f"points={len(reference_points)}, data={len(reference_data)}."
        )
    parameters = _parameter_batch(config, experiment, diameter_um)
    sample: dict[str, Any] = {"Batch Parameters": parameters}
    artifacts: list[dict[str, Any]] = []

    if experiment.name == "compression":
        parameterization = surrogate_parameterization_for_experiment(experiment)
        if parameterization != DIRECT_KA_KB_SURROGATE_PARAMETERIZATION:
            raise ValueError(
                "The frozen UQ_EMB compression replay requires direct_ka_kb parameterization."
            )
        preload, _scalar, batch_model = resolve_direct_compression_surrogate_surface()
        preload(diameter_um, device=device, backend="dnn")
        batch_model(sample, reference_points, diameter_um, device=device, backend="dnn")
        artifacts = _mechanical_artifacts(experiment, diameter_um)
    elif experiment.name == "indentation":
        from emb.indentation.evalkit.posterior_indentation import (
            compute_indentation_surrogate_batch,
            preload_indentation_surrogate,
        )

        preload_indentation_surrogate(diameter_um, device=device, backend="dnn")
        adapted = adapt_batch_sample_for_legacy_surrogate(
            sample,
            config=config,
            modality="indentation",
            project_root=REPO_ROOT,
        )
        compute_indentation_surrogate_batch(
            adapted,
            reference_points,
            diameter_um,
            device=device,
        )
        sample.update(
            {
                key: value
                for key, value in adapted.items()
                if key != "Batch Parameters"
            }
        )
        artifacts = _mechanical_artifacts(experiment, diameter_um)
    elif experiment.name == "resonance":
        preload_emb_resonance(diameter_um, device=device, backend="direct")
        compute_emb_resonance_batch(
            sample,
            reference_points,
            diameter_um,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported UQ_EMB experiment {experiment.name!r}.")

    summary = _summarize_batch(
        sample,
        expected_rows=len(_BATCH_FRACTIONS),
        expected_columns=len(reference_points),
    )
    reference_points_payload = np.asarray(reference_points, dtype=np.float64).tolist()
    reference_data_payload = np.asarray(reference_data, dtype=np.float64).tolist()
    parameter_payload = parameters.astype(np.float64).tolist()
    return {
        "dataset_name": experiment.dataset_name(diameter_um),
        "experiment": experiment.name,
        "lane": experiment.lane,
        "configured_diameter_um": float(diameter_um),
        "grouped_reference_data": bool(experiment.grouped_reference_data),
        "reference_rows": len(reference_points),
        "reference_diameters_um": sorted({float(value) for value in reference_points})
        if experiment.name == "resonance"
        else [float(diameter_um)],
        "parameter_order": ["ka", "kb", "d0", "sigma"],
        "parameter_batch": parameter_payload,
        "parameter_batch_sha256": _canonical_sha256(parameter_payload),
        "reference_points": reference_points_payload,
        "reference_data": reference_data_payload,
        "reference_input_sha256": _canonical_sha256(
            {
                "points": reference_points_payload,
                "data": reference_data_payload,
            }
        ),
        "artifacts": artifacts,
        **summary,
    }


def run_forward_canary(
    *,
    config_path: Path,
    output_path: Path,
    device: str,
    site: str,
) -> dict[str, Any]:
    started = time.monotonic()
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    InferenceConfig.model_validate(config)
    os.environ["HUQ_INFERENCE_CONFIG"] = str(config_path)
    config_binding = load_materialization_binding(config_path, repo_root=REPO_ROOT)

    preflight = preflight_emb_resonance_config(config, project_root=REPO_ROOT)
    if preflight.get("status") != "passed":
        raise ValueError(f"UQ_EMB preflight did not pass: {preflight.get('status')!r}.")

    experiments = [
        experiment
        for experiment in load_experiments(config, REPO_ROOT)
        if experiment.enabled
    ]
    datasets = [
        _evaluate_dataset(config, experiment, diameter_um, device=device)
        for experiment in experiments
        for diameter_um in experiment.diameters
    ]
    mechanical_count = sum(row["experiment"] in {"compression", "indentation"} for row in datasets)
    acoustic_count = sum(row["experiment"] == "resonance" for row in datasets)
    if mechanical_count != 3:
        raise ValueError(f"Expected three mechanical datasets; evaluated {mechanical_count}.")
    expected_acoustic_count = 3 if preflight.get("agent") == "sonovue" else 1
    if acoustic_count != expected_acoustic_count:
        raise ValueError(
            f"Expected {expected_acoustic_count} configured acoustic datasets; evaluated {acoustic_count}."
        )

    evaluator = config.get("resonance", {}).get("evaluator", {})
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "site": site,
        "device": device,
        "agent": preflight.get("agent"),
        "config_path": str(config_path),
        **config_binding,
        "provenance": runtime_provenance(
            repo_root=REPO_ROOT,
            site=site,
        ),
        "phase1_contract_mode": config.get("phase1_contract_mode"),
        "acoustic_forward_model": preflight.get("forward_model"),
        "acoustic_artifacts": {
            path_key: _acoustic_artifact(
                Path(str(evaluator[path_key])),
                str(evaluator[sha_key]),
            )
            for path_key, sha_key in (
                ("artifact_path", "artifact_sha256"),
                ("bank_build_report_path", "bank_build_report_sha256"),
                ("independent_go_path", "independent_go_sha256"),
                ("promotion_contract_path", "promotion_contract_sha256"),
            )
        },
        "preflight": preflight,
        "datasets": datasets,
        "dataset_counts": {
            "mechanical": mechanical_count,
            "configured_acoustic": acoustic_count,
            "acoustic_reference_rows": sum(
                row["reference_rows"] for row in datasets if row["experiment"] == "resonance"
            ),
        },
        "wall_seconds": time.monotonic() - started,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exercise frozen UQ_EMB mechanical and acoustic forward artifacts."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--site", choices=("karolina", "vega"), default=None)
    args = parser.parse_args(argv)
    site = resolve_hpc_site(
        cli_site=args.site,
        env=os.environ,
        allow_hostname=False,
        default=None,
    )
    report = run_forward_canary(
        config_path=args.config,
        output_path=args.output,
        device=args.device,
        site=site,
    )
    print(
        f"UQ_EMB forward canary passed for {report['agent']} on {site} "
        f"in {report['wall_seconds']:.2f} s: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
