#!/usr/bin/env python3
"""Run a minimal GV operational canary across runtime, surrogate, and Phase 1."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.inference.gv_hbi import (  # noqa: E402
    GV_HBI_EXPERIMENTAL_FLAG,
    GV_PHASE1_EXECUTION_MANIFEST,
    GV_PHASE1_SETUP_MANIFEST,
)
from meso_uq.structures import get_structure  # noqa: E402

RUNTIME_SCRIPT = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_runtime.py"
SURROGATE_SCRIPT = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dnn_smoke.py"
PHASE1_SCRIPT = REPO_ROOT / "inference" / "scripts" / "run_phase_1.py"
FINAL_MANIFEST = "gv_operational_canary_manifest.json"
GV_SIMULATION_ROOT = REPO_ROOT / "gv_simulation_files"
DEFAULT_SELECTION = "gv:torsion"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="gv")
    parser.add_argument(
        "--selection",
        default=None,
        help=f"Structure-qualified GV selection, for example gv:torsion. Defaults to {DEFAULT_SELECTION!r} when neither --selection nor --experiment is provided.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--geometry-id", default=None)
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable GV control override passed to the runtime and surrogate stages.",
    )
    parser.add_argument("--include-experimental", action="store_true", default=False)
    parser.add_argument("--reference-kind", choices=("synthetic", "dpd_generated"), default="synthetic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-curves", type=int, default=5)
    parser.add_argument("--points-per-curve", type=int, default=4)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epoch", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--platform", choices=("vega", "karolina", "local"), default="vega")
    parser.add_argument("--run-mirheo", action="store_true", default=False)
    return parser


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _require_runtime_flag() -> str:
    if _as_bool(os.environ.get(GV_HBI_EXPERIMENTAL_FLAG)):
        return f"env:{GV_HBI_EXPERIMENTAL_FLAG}"
    raise PermissionError(
        "GV operational canary is disabled by default. Set "
        f"{GV_HBI_EXPERIMENTAL_FLAG}=1 to exercise the runtime -> surrogate -> hierarchical seam."
    )


def _resolve_selection(selection: str) -> tuple[str, str]:
    parts = selection.split(":")
    if len(parts) != 2:
        raise ValueError("GV operational canary selection must use the form gv:<experiment>.")
    structure, experiment = parts
    if structure != "gv":
        raise ValueError(f"GV operational canary only supports structure 'gv', got {structure!r}.")
    return structure, experiment


def _resolve_runtime_selection(
    *, structure: str | None, experiment: str | None, selection: str | None
) -> tuple[str, str]:
    if selection is not None:
        resolved_structure, resolved_experiment = _resolve_selection(selection)
        if experiment is not None and experiment != resolved_experiment:
            raise ValueError(
                f"Conflicting runtime selections: --selection={selection!r} and --experiment={experiment!r}."
            )
        if structure is not None and structure != resolved_structure:
            raise ValueError(
                f"Conflicting runtime structure values: --structure={structure!r} and --selection={selection!r}."
            )
        return resolved_structure, resolved_experiment
    if experiment is None:
        raise ValueError("Either --selection or --experiment is required.")
    resolved_structure = structure if structure is not None else "gv"
    if resolved_structure != "gv":
        raise ValueError(f"GV operational canary only supports structure 'gv', got {resolved_structure!r}.")
    return resolved_structure, experiment


def _control_cli_items(control_items: list[str]) -> list[str]:
    argv: list[str] = []
    for item in control_items:
        argv.extend(["--control", item])
    return argv


def _validate_output_root(output_root: Path) -> Path:
    resolved = output_root.expanduser().resolve()
    unsafe_roots = (GV_SIMULATION_ROOT, REPO_ROOT / "gv", REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests")
    for unsafe_root in unsafe_roots:
        if resolved == unsafe_root or unsafe_root in resolved.parents:
            raise ValueError(
                f"GV operational canary outputs must not be inside repository source trees: {unsafe_root}"
            )
    return resolved


def _summarize_runtime_known_issues(runtime_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_known_issues = runtime_manifest.get("known_issues", [])
    if not isinstance(raw_known_issues, list):
        return []
    summaries: list[dict[str, Any]] = []
    for issue in raw_known_issues:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity", "")).lower()
        summaries.append(
            {
                "id": str(issue.get("id", "")),
                "summary": str(issue.get("summary", "")),
                "evidence": str(issue.get("evidence", "")),
                "severity": severity,
                "classification": "experimental_blocked" if severity in {"error", "blocking", "blocked", "critical"} else "observed",
            }
        )
    return summaries


def _phase1_config(*, runtime_manifest: dict[str, Any], surrogate_manifest_path: Path) -> dict[str, Any]:
    return {
        "pop_size": 32,
        "max_gen": 1,
        "target_cov": 0.8,
        "covariance_scaling": 0.04,
        "structure": "gv",
        "structures": ["gv"],
        "use_surrogate": True,
        "surrogate": {"backend": "dnn"},
        "prior_ka": [0.1, 1.1],
        "prior_kb": [0.2, 1.2],
        "prior_mu": [0.3, 1.3],
        "prior_b1": [0.4, 1.4],
        "prior_b2": [0.5, 1.5],
        "prior_a3": [0.6, 1.6],
        "prior_a4": [0.7, 1.7],
        "prior_mu_l": [0.8, 1.8],
        "prior_c": [0.9, 1.9],
        "prior_sigma": [0.01, 0.10],
        "hyperprior_mu_ka": [0.1, 1.1],
        "hyperprior_sigma_ka": [0.01, 0.2],
        "hyperprior_mu_kb": [0.2, 1.2],
        "hyperprior_sigma_kb": [0.01, 0.2],
        "hyperprior_mu_mu": [0.3, 1.3],
        "hyperprior_sigma_mu": [0.01, 0.2],
        "hyperprior_mu_b1": [0.4, 1.4],
        "hyperprior_sigma_b1": [0.01, 0.2],
        "hyperprior_mu_b2": [0.5, 1.5],
        "hyperprior_sigma_b2": [0.01, 0.2],
        "hyperprior_mu_a3": [0.6, 1.6],
        "hyperprior_sigma_a3": [0.01, 0.2],
        "hyperprior_mu_a4": [0.7, 1.7],
        "hyperprior_sigma_a4": [0.01, 0.2],
        "hyperprior_mu_mu_l": [0.8, 1.8],
        "hyperprior_sigma_mu_l": [0.01, 0.2],
        "hyperprior_mu_c": [0.9, 1.9],
        "hyperprior_sigma_c": [0.01, 0.2],
        "experiments": [
            {
                "structure": "gv",
                "name": runtime_manifest["experiment"],
                "geometries": [runtime_manifest["geometry"]],
                "controls": [runtime_manifest["control_id"]],
                "surrogate_manifest": str(surrogate_manifest_path),
            }
        ],
    }


def _require_phase1_compatible_surrogate_artifact(
    *,
    surrogate_manifest: dict[str, Any],
    surrogate_root: Path,
    surrogate_status: str | None = None,
) -> dict[str, Any]:
    artifacts = surrogate_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("GV DNN smoke manifest is missing artifacts.")
    artifact_path = artifacts.get("artifact_path") or artifacts.get("model_path")
    if artifact_path:
        return surrogate_manifest
    if surrogate_status in {"dry-run", "skipped_missing_dependency"}:
        placeholder = surrogate_root / "gv_dnn_surrogate_smoke_artifact_placeholder.json"
        placeholder.write_text(json.dumps({"kind": "dry-run-artifact"}, sort_keys=True), encoding="utf-8")
        artifacts["artifact_path"] = str(placeholder)
        artifacts["model_path"] = str(placeholder)
        return surrogate_manifest
    raise FileNotFoundError("GV DNN smoke manifest is missing a surrogate artifact path.")


def _build_verdict(*, surrogate_report: dict[str, Any], phase1_execution_manifest: dict[str, Any]) -> str:
    if surrogate_report.get("status") not in {"passed", "dry-run"}:
        return "fail"
    if phase1_execution_manifest.get("status") != "phase1_korali_completed":
        return "fail"
    return "pass"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        enabled_by = _require_runtime_flag()
        selection = str(args.selection) if args.selection is not None else None
        if selection is None and args.experiment is None:
            selection = DEFAULT_SELECTION
        structure_name, experiment_name = _resolve_runtime_selection(
            structure=args.structure,
            experiment=str(args.experiment) if args.experiment is not None else None,
            selection=selection,
        )
        resolved_selection = f"{structure_name}:{experiment_name}"
        structure = get_structure(structure_name)
        structure.get_experiment(experiment_name, include_experimental=args.include_experimental)
    except (KeyError, PermissionError, ValueError) as exc:
        parser.error(str(exc))

    output_root = _validate_output_root(Path(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = output_root / "runtime"
    surrogate_root = output_root / "surrogate"
    phase1_root = output_root / "phase1"

    runtime_module = _load_module("gv_operational_canary_runtime", RUNTIME_SCRIPT)
    surrogate_module = _load_module("gv_operational_canary_surrogate", SURROGATE_SCRIPT)
    phase1_module = _load_module("gv_operational_canary_phase1", PHASE1_SCRIPT)

    runtime_argv: list[str] = []
    if selection is not None:
        runtime_argv.extend(["--selection", selection])
    else:
        runtime_argv.extend(["--structure", structure_name, "--experiment", experiment_name])
    runtime_argv.extend(["--output-root", str(runtime_root)])
    runtime_argv.extend(["--platform", args.platform])
    if args.geometry_id is not None:
        runtime_argv.extend(["--geometry-id", str(args.geometry_id)])
    runtime_argv.extend(_control_cli_items(list(args.control)))
    if args.include_experimental:
        runtime_argv.append("--include-experimental")
    if not args.run_mirheo:
        runtime_argv.append("--dry-run")
    runtime_rc = int(runtime_module.main(runtime_argv))
    if runtime_rc != 0:
        raise RuntimeError(f"GV runtime dry-run failed with code {runtime_rc}.")
    runtime_manifest_path = runtime_root / "gv_runtime_dry_run_manifest.json"
    runtime_manifest = _read_json(runtime_manifest_path)
    runtime_render_manifest_path = runtime_root / "gv_runtime_render_manifest.json"
    runtime_known_issues = _summarize_runtime_known_issues(runtime_manifest)

    surrogate_argv = [
        "--runtime-manifest",
        str(runtime_manifest_path),
        "--output-root",
        str(surrogate_root),
        "--reference-kind",
        str(args.reference_kind),
        "--seed",
        str(args.seed),
        "--num-curves",
        str(args.num_curves),
        "--points-per-curve",
        str(args.points_per_curve),
        "--width",
        str(args.width),
        "--depth",
        str(args.depth),
        "--batch-size",
        str(args.batch_size),
        "--max-epoch",
        str(args.max_epoch),
        "--val-fraction",
        str(args.val_fraction),
    ]
    if args.include_experimental:
        surrogate_argv.append("--include-experimental")
    surrogate_rc = int(surrogate_module.main(surrogate_argv))
    if surrogate_rc != 0:
        raise RuntimeError(f"GV DNN smoke workflow failed with code {surrogate_rc}.")
    surrogate_manifest_path = surrogate_root / "gv_dnn_surrogate_smoke_manifest.json"
    surrogate_report_path = surrogate_root / "gv_dnn_surrogate_smoke_report.json"
    surrogate_report = _read_json(surrogate_report_path)
    surrogate_manifest = _read_json(surrogate_manifest_path)
    surrogate_manifest = _require_phase1_compatible_surrogate_artifact(
        surrogate_manifest=surrogate_manifest,
        surrogate_root=surrogate_root,
        surrogate_status=surrogate_report.get("status"),
    )
    _write_json(surrogate_manifest_path, surrogate_manifest)

    phase1_config_path = output_root / "gv_operational_canary_phase1.yaml"
    phase1_config = _phase1_config(runtime_manifest=runtime_manifest, surrogate_manifest_path=surrogate_manifest_path)
    phase1_config_path.write_text(yaml.safe_dump(phase1_config, sort_keys=False), encoding="utf-8")
    phase1_module.run_inference(
        dry_run=False,
        config_path=str(phase1_config_path),
        output_dir=str(phase1_root),
        device="cpu",
    )

    phase1_manifest_root = phase1_root / "results_phase_1"
    phase1_setup_manifest = _read_json(phase1_manifest_root / GV_PHASE1_SETUP_MANIFEST)
    phase1_execution_manifest = _read_json(phase1_manifest_root / GV_PHASE1_EXECUTION_MANIFEST)
    calibrated = list(phase1_setup_manifest["parameter_contract"]["calibrated"])
    phase1_variable_names = list(phase1_execution_manifest["phase1"]["variable_names"])
    noise_parameter = str(phase1_execution_manifest["phase1"]["noise_model"]["parameter"])
    nuisance = [name for name in phase1_variable_names if name not in calibrated and name != noise_parameter]
    control_names = sorted(runtime_manifest["controls"])
    control_overlap = sorted(set(control_names).intersection(calibrated + nuisance))
    if control_overlap:
        raise ValueError(
            "GV operational canary detected controls inside calibrated/nuisance variables: "
            + ", ".join(control_overlap)
        )

    final_manifest = {
        "workflow": "gv_operational_canary",
        "enabled_by": enabled_by,
        "structure": structure_name,
        "experiment": experiment_name,
        "selection": resolved_selection,
        "geometry": runtime_manifest["geometry"],
        "geometry_spec": runtime_manifest.get("geometry_spec"),
        "controls": runtime_manifest["controls"],
        "control_id": runtime_manifest["control_id"],
        "dataset_id": runtime_manifest["dataset_id"],
        "reference_kind": surrogate_manifest["reference_kind"],
        "surrogate_backend": surrogate_manifest["backend"],
        "calibrated_parameters": calibrated,
        "nuisance_parameters": nuisance,
        "noise_model": phase1_execution_manifest["phase1"]["noise_model"],
        "noise_parameters": [noise_parameter],
        "output_roots": {
            "canary": str(output_root),
            "runtime": str(runtime_root),
            "surrogate": str(surrogate_root),
            "phase1": str(phase1_root),
        },
        "artifacts": {
            "runtime_manifest": str(runtime_manifest_path),
            "runtime_render_manifest": str(runtime_render_manifest_path),
            "surrogate_manifest": str(surrogate_manifest_path),
            "surrogate_report": str(surrogate_report_path),
            "reference_manifest": str(surrogate_root / "gv_reference_manifest.json"),
            "phase1_setup_manifest": str(phase1_manifest_root / GV_PHASE1_SETUP_MANIFEST),
            "phase1_execution_manifest": str(phase1_manifest_root / GV_PHASE1_EXECUTION_MANIFEST),
            "phase1_config": str(phase1_config_path),
        },
        "checks": {
            "runtime_dry_run": not args.run_mirheo,
            "runtime_flag_enabled": True,
            "runtime_experimental": bool(runtime_manifest.get("experimental", False)),
            "runtime_blocked_issue_count": sum(
                1 for issue in runtime_known_issues if issue.get("classification") == "experimental_blocked"
            ),
            "controls_excluded_from_calibrated_and_nuisance": True,
            "runtime_status": runtime_rc,
            "surrogate_status": surrogate_report["status"],
            "phase1_status": phase1_execution_manifest["status"],
            "phase1_execution_model": phase1_execution_manifest["execution_model"],
        },
        "runtime_stage": {
            "runtime_package": runtime_manifest.get("runtime_package"),
            "source_root": runtime_manifest.get("source_root"),
            "experimental": bool(runtime_manifest.get("experimental", False)),
            "known_issues": runtime_known_issues,
        },
        "verdict": _build_verdict(
            surrogate_report=surrogate_report,
            phase1_execution_manifest=phase1_execution_manifest,
        ),
    }
    final_manifest_path = output_root / FINAL_MANIFEST
    _write_json(final_manifest_path, final_manifest)
    print(f"GV operational canary manifest: {final_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
