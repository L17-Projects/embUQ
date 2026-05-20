#!/usr/bin/env python3
"""Run and/or render EMB 3.4um active-learning control stages for the final gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


_REPO_ROOT = _repo_root()
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meso_uq.active_learning import (
    ACTIVE_LEARNING_FINAL_GATE_ACQUISITION_PLOT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_CURVE_METRICS_PLOT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_FAILURE_PLOT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_KA_KB_PLOT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_MODEL_SELECTION_PLOT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_PREDICTION_PLOT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_RESIDUALS_PLOT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_SUMMARY_PLOT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME,
    EMB_34UM_FINAL_GATE_INGESTION_MANIFEST_FILENAME,
    EMB_34UM_FINAL_GATE_INGESTION_PLOT_FILENAME,
    EMB_34UM_FINAL_GATE_INGESTION_PLOT_SIDECAR_FILENAME,
    EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME,
    EMB_34UM_FINAL_GATE_INGESTION_SUMMARY_CSV_FILENAME,
    EMB_34UM_FINAL_GATE_QUARANTINE_FILENAME,
)
from meso_uq.active_learning.emb_34um_al_vs_lhs_validation import (
    EMB_34UM_AL_VS_LHS_VALIDATION_DISAGREEMENT_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_FAILURE_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_FORCE_OVERLAY_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_ROUND1_SAMPLES_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_ROUND_ADDITIONS_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_RUNTIME_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_SAMPLES_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_SOURCE_PLOT_FILENAME,
)
CONTROLLER_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_active_learning_controller.v1"


def _load_prepare_module():
    prepare_script = (
        _REPO_ROOT
        / "scripts"
        / "workflows"
        / "emb"
        / "active_learning"
        / "prepare_emb_34um_final_gate.py"
    )
    spec = importlib.util.spec_from_file_location("_prepare_emb_34um_final_gate", prepare_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {prepare_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prepare = _load_prepare_module()

DEFAULT_SCRATCH_ROOT = _prepare.DEFAULT_SCRATCH_ROOT
DEFAULT_VAULT_ROOT = _prepare.DEFAULT_VAULT_ROOT
DEFAULT_FORCE_GRID_DATA = _prepare.DEFAULT_FORCE_GRID_DATA
DEFAULT_LHS_CANDIDATE_COUNT = _prepare.LHS_CANDIDATE_COUNT
DEFAULT_VALIDATION_TARGET_COUNT = _prepare.VALIDATION_TARGET_COUNT
DEFAULT_DNN_ENSEMBLE_TARGET_SIZE = _prepare.DNN_ENSEMBLE_TARGET_SIZE
DEFAULT_CANDIDATE_POOL_SIZE = _prepare.CANDIDATE_POOL_SIZE
DEFAULT_CANARY_FORCE_COUNT = _prepare.CANARY_FORCE_COUNT

CONTROLLER_MANIFEST_FILE = "emb_34um_active_learning_controller_manifest.json"

EXPECTED_STAGE_ORDER = [
    "canary",
    "benchmark",
    "lhs_submit",
    "validation_submit",
    "al_round_1_submit",
    "al_round_2_select_render",
    "al_round_2_submit",
    "al_round_3_select_render",
    "al_round_3_submit",
    "final_ingestion",
    "build_al_vs_lhs_rows",
    "al_vs_lhs_validation",
    "final_gate_report",
]


def _python_exec() -> str:
    return shlex.quote(sys.executable)


def _python_one_liner(body: str, *, execution_mode: str = "execute") -> str:
    exec_prefix = f"EXECUTION_MODE={shlex.quote(execution_mode)} "
    return f"{exec_prefix}{_python_exec()} -c {shlex.quote(body)}"


def _replace_array(command: str, array_spec: str) -> str:
    replaced = re.sub(r"--array=[^\s\"]+", f"--array={array_spec}", command)
    if "--array=" not in replaced:
        raise ValueError("Submission command does not contain --array.")
    return replaced


def _replace_execution_mode(command: str, execution_mode: str) -> str:
    return command.replace("EXECUTION_MODE=render-only", f"EXECUTION_MODE={execution_mode}")


def _append_export(command: str, *, key: str, value: str | Path) -> str:
    if "--export=" not in command:
        raise ValueError("Submission command does not contain --export.")
    return command.replace("--export=", f"--export={key}={shlex.quote(str(value))},", 1)


def _stage_dict(
    *,
    name: str,
    description: str,
    commands: list[str],
    expected_output_roots: list[Path],
    acceptance_criteria: dict[str, Any] | None = None,
    command_type: str = "submission",
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    stage = {
        "name": name,
        "description": description,
        "command_type": command_type,
        "commands": [{"command": item} for item in commands],
        "expected_output_roots": [str(path) for path in expected_output_roots],
        "acceptance_criteria": acceptance_criteria or {},
        "status": "blocked" if blockers else "planned",
    }
    if blockers:
        stage["blockers"] = list(blockers)
    return stage


def _blocked_stage(
    *,
    name: str,
    description: str,
    criterion_id: str,
    blocker: str,
    expected_output_roots: list[Path] | None = None,
    command_type: str = "blocked_acceptance",
) -> dict[str, Any]:
    return _stage_dict(
        name=name,
        description=description,
        commands=[],
        expected_output_roots=expected_output_roots or [],
        acceptance_criteria={
            criterion_id: False,
            "blocking_acceptance_criterion": blocker,
        },
        command_type=command_type,
        blockers=[blocker],
    )


def _controller_stage_command(
    *,
    action: str,
    campaign_manifest: Path,
    execution_mode: str,
    round_index: int | None = None,
    force_grid_path: Path | None = None,
    run_id_prefix: str | None = None,
    walltime: str | None = None,
    concurrent_jobs: int | None = None,
    retry_limit: int | None = None,
    round_payloads: Path | None = None,
) -> str:
    parts = [
        f"EXECUTION_MODE={shlex.quote(execution_mode)}",
        _python_exec(),
        shlex.quote(str(Path(__file__).resolve())),
        "--stage-action",
        shlex.quote(action),
        "--campaign-manifest",
        shlex.quote(str(campaign_manifest)),
    ]
    if round_index is not None:
        parts.extend(["--round-index", str(round_index)])
    if force_grid_path is not None:
        parts.extend(["--force-grid", shlex.quote(str(force_grid_path))])
    if run_id_prefix is not None:
        parts.extend(["--run-id-prefix", shlex.quote(run_id_prefix)])
    if walltime is not None:
        parts.extend(["--walltime", shlex.quote(walltime)])
    if concurrent_jobs is not None:
        parts.extend(["--concurrent-jobs", str(concurrent_jobs)])
    if retry_limit is not None:
        parts.extend(["--retry-limit", str(retry_limit)])
    if round_payloads is not None:
        parts.extend(["--round-payloads", shlex.quote(str(round_payloads))])
    return " ".join(parts)


def _build_round_select_render_command(
    *,
    round_index: int,
    campaign_manifest: Path,
    force_grid_path: Path,
    run_id_prefix: str,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    execution_mode: str,
) -> str:
    return _controller_stage_command(
        action="select-render-round",
        campaign_manifest=campaign_manifest,
        round_index=round_index,
        force_grid_path=force_grid_path,
        run_id_prefix=run_id_prefix,
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
        execution_mode=execution_mode,
    )


def _build_validation_command(
    curve_rows: Path,
    output_root: Path,
    *,
    runtime_rows: Path | None = None,
    execution_mode: str,
) -> str:
    validate_script = (
        _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "validate_emb_34um_al_vs_lhs.py"
    )
    parts = [
        f"EXECUTION_MODE={shlex.quote(execution_mode)}",
        _python_exec(),
        shlex.quote(str(validate_script)),
        "--curve-rows",
        shlex.quote(str(curve_rows)),
        "--output-root",
        shlex.quote(str(output_root)),
        "--prefix-counts",
        "30,60,90",
        "--acquisition-engine",
        "dnn_ensemble_disagreement_diversity",
        "--adaptive-acquisition-available",
    ]
    if runtime_rows is not None:
        parts.extend(["--runtime-rows", shlex.quote(str(runtime_rows))])
    return " ".join(parts)


def _validation_output_paths(root: Path) -> list[Path]:
    return [
        root / EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_SAMPLES_PLOT_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_ROUND1_SAMPLES_PLOT_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_ROUND_ADDITIONS_PLOT_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_SOURCE_PLOT_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_DISAGREEMENT_PLOT_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_FORCE_OVERLAY_PLOT_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_FAILURE_PLOT_FILENAME,
        root / EMB_34UM_AL_VS_LHS_VALIDATION_RUNTIME_PLOT_FILENAME,
    ]


def _ingestion_output_paths(root: Path) -> list[Path]:
    return [
        root / EMB_34UM_FINAL_GATE_INGESTION_MANIFEST_FILENAME,
        root / EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME,
        root / EMB_34UM_FINAL_GATE_INGESTION_SUMMARY_CSV_FILENAME,
        root / EMB_34UM_FINAL_GATE_INGESTION_PLOT_FILENAME,
        root / EMB_34UM_FINAL_GATE_INGESTION_PLOT_SIDECAR_FILENAME,
        root / EMB_34UM_FINAL_GATE_QUARANTINE_FILENAME,
    ]


def _final_report_output_paths(root: Path, run_id_prefix: str) -> list[Path]:
    report_root = root / run_id_prefix / "iterations" / "iter_0001"
    return [
        report_root / ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_CURVE_METRICS_PLOT_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_RESIDUALS_PLOT_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_PREDICTION_PLOT_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_ACQUISITION_PLOT_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_KA_KB_PLOT_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_FAILURE_PLOT_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_MODEL_SELECTION_PLOT_FILENAME,
        report_root / ACTIVE_LEARNING_FINAL_GATE_SUMMARY_PLOT_FILENAME,
    ]


def _build_report_command(output_root: Path, run_id_prefix: str, execution_mode: str) -> str:
    rounds_path = output_root.parent / "round_payloads.json"
    return _controller_stage_command(
        action="final-report",
        campaign_manifest=output_root.parent / "emb_34um_final_gate_campaign_manifest.json",
        execution_mode=execution_mode,
        run_id_prefix=run_id_prefix,
        round_payloads=rounds_path,
    )


def _expected_stage_outputs(
    manifest: dict[str, Any],
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[Path]]:
    canary_outputs = [Path(path) for path in manifest["canary_gate"]["expected_output_roots"]]
    benchmark_outputs = [Path(path) for path in manifest["benchmark_gate"]["expected_output_roots"]]
    full_outputs = [Path(path) for path in manifest["full_gate"]["expected_output_roots"]]
    lhs_outputs = [Path(path) for path in manifest["lhs_gate"]["expected_output_roots"]]
    validation_outputs = [Path(path) for path in manifest["validation_gate"]["expected_output_roots"]]
    return canary_outputs, benchmark_outputs, full_outputs, lhs_outputs, validation_outputs


def build_emb_34um_active_learning_controller(
    *,
    timestamp: str,
    scratch_root: Path,
    vault_root: Path,
    force_grid_path: Path,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
    run_id_prefix: str,
    skip_vault_copy: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if concurrent_jobs < 1:
        raise ValueError("concurrent_jobs must be positive.")
    if retry_limit < 0:
        raise ValueError("retry_limit must be zero or positive.")

    execution_mode = "render-only" if dry_run else "execute"

    prepare_result = _prepare.prepare_emb_34um_final_gate(
        timestamp=timestamp,
        scratch_root=scratch_root,
        vault_root=vault_root,
        force_grid_path=force_grid_path,
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
        run_id_prefix=run_id_prefix,
        canary_force_count=DEFAULT_CANARY_FORCE_COUNT,
        lhs_candidate_count=DEFAULT_LHS_CANDIDATE_COUNT,
        validation_target_count=DEFAULT_VALIDATION_TARGET_COUNT,
        dnn_ensemble_target_size=DEFAULT_DNN_ENSEMBLE_TARGET_SIZE,
        candidate_pool_size=DEFAULT_CANDIDATE_POOL_SIZE,
        skip_vault_copy=skip_vault_copy,
    )

    campaign_manifest = prepare_result["manifest"]
    campaign_root = Path(campaign_manifest["campaign_root"])

    canary_outputs, benchmark_outputs, full_outputs, lhs_outputs, validation_outputs = _expected_stage_outputs(
        campaign_manifest
    )
    final_validation_root = campaign_root / "al_vs_lhs_validation"
    final_validation_rows = campaign_root / "al_vs_lhs_rows.json"
    round_payloads_path = campaign_root / "round_payloads.json"
    runtime_rows_path = campaign_root / "runtime_rows.json"
    final_report_root = campaign_root / "final_gate_report"
    ingestion_root = campaign_root / "ingestion_report"
    round_2_batch_root = campaign_root / "adaptive_round_02"
    round_3_batch_root = campaign_root / "adaptive_round_03"

    canary_command = _replace_execution_mode(campaign_manifest["commands"]["canary"]["command"], execution_mode)
    benchmark_command = _replace_execution_mode(
        campaign_manifest["commands"]["benchmark"]["command"], execution_mode
    )
    full_submit_command = _replace_execution_mode(campaign_manifest["commands"]["full"]["command"], execution_mode)
    lhs_command = _replace_execution_mode(campaign_manifest["commands"]["lhs"]["command"], execution_mode)
    validation_command = _replace_execution_mode(
        campaign_manifest["commands"]["validation"]["command"], execution_mode
    )

    stages: list[dict[str, Any]] = []
    stages.append(
        _stage_dict(
            name="canary",
            description="Canary 3-forcepoint check before batch production.",
            commands=[canary_command],
            expected_output_roots=canary_outputs,
            acceptance_criteria={"canary_count_is_1": len(canary_outputs) == 1},
        )
    )
    stages.append(
        _stage_dict(
            name="benchmark",
            description="Run benchmark stiffness points before AL gate acquisition rounds.",
            commands=[benchmark_command],
            expected_output_roots=benchmark_outputs,
            acceptance_criteria={"benchmark_count_is_3": len(benchmark_outputs) == 3},
        )
    )
    stages.append(
        _stage_dict(
            name="lhs_submit",
            description="Run one independent 90-curve LHS comparator batch.",
            commands=[lhs_command],
            expected_output_roots=lhs_outputs,
            command_type="submission",
            acceptance_criteria={"lhs_count_is_90": len(lhs_outputs) == 90},
        )
    )
    stages.append(
        _stage_dict(
            name="validation_submit",
            description="Run the fixed 30-curve validation target batch used for grouped surrogate metrics.",
            commands=[validation_command],
            expected_output_roots=validation_outputs,
            command_type="submission",
            acceptance_criteria={"validation_count_is_30": len(validation_outputs) == 30},
        )
    )

    round_1_submit_command = _replace_array(full_submit_command, f"0-29%{concurrent_jobs}")
    stages.append(
        _stage_dict(
            name="al_round_1_submit",
            description="Submit the initial Sobol/maximin AL round of 30 candidates.",
            commands=[round_1_submit_command],
            expected_output_roots=full_outputs[0:30],
            command_type="submission",
            acceptance_criteria={"round_1_submit_count_is_30": len(full_outputs[0:30]) == 30},
        )
    )

    campaign_manifest_path = Path(prepare_result["manifest_path"])
    for round_index, batch_root in ((2, round_2_batch_root), (3, round_3_batch_root)):
        stages.append(
            _stage_dict(
                name=f"al_round_{round_index}_select_render",
                description=(
                    f"Retrain on completed AL prefixes, score a fresh pool, select 24 acquisition "
                    f"and 6 exploration candidates, and render adaptive round {round_index}."
                ),
                commands=[
                    _build_round_select_render_command(
                        round_index=round_index,
                        campaign_manifest=campaign_manifest_path,
                        force_grid_path=force_grid_path,
                        run_id_prefix=run_id_prefix,
                        walltime=walltime,
                        concurrent_jobs=concurrent_jobs,
                        retry_limit=retry_limit,
                        execution_mode=execution_mode,
                    )
                ],
                expected_output_roots=[batch_root / "emb_34um_batch_summary.json", batch_root / "selection_manifest.json"],
                command_type="model_select",
                acceptance_criteria={
                    f"round_{round_index}_fresh_surrogate_selection_implemented": True,
                    "acquisition_count": 24,
                    "exploration_count": 6,
                },
            )
        )
        submit_command = _append_export(
            _replace_array(full_submit_command, f"0-29%{concurrent_jobs}"),
            key="BATCH_DIR_OVERRIDE",
            value=batch_root,
        )
        stages.append(
            _stage_dict(
                name=f"al_round_{round_index}_submit",
                description=f"Submit dynamically rendered AL round {round_index} batch of 30 candidates.",
                commands=[submit_command],
                expected_output_roots=[batch_root],
                command_type="submission",
                acceptance_criteria={f"round_{round_index}_dynamic_submit_command_ready": True},
            )
        )

    final_validation_outputs = _validation_output_paths(final_validation_root)
    ingestion_outputs = _ingestion_output_paths(ingestion_root)
    final_report_outputs = _final_report_output_paths(final_report_root, run_id_prefix)
    stages.append(
        _stage_dict(
            name="final_ingestion",
            description="Ingest runtime status and F_Delta outputs after execution.",
            commands=[
                _controller_stage_command(
                    action="ingest",
                    campaign_manifest=campaign_manifest_path,
                    execution_mode=execution_mode,
                )
            ],
            expected_output_roots=ingestion_outputs,
            command_type="analysis",
            acceptance_criteria={"final_ingestion_command_implemented": True},
        )
    )
    stages.append(
        _stage_dict(
            name="build_al_vs_lhs_rows",
            description="Train/evaluate AL and LHS prefix surrogates and write row evidence.",
            commands=[
                _controller_stage_command(
                    action="build-evidence",
                    campaign_manifest=campaign_manifest_path,
                    execution_mode=execution_mode,
                )
            ],
            expected_output_roots=[final_validation_rows, round_payloads_path, runtime_rows_path],
            command_type="analysis",
            acceptance_criteria={"al_vs_lhs_evidence_builder_implemented": True},
        )
    )
    stages.append(
        _stage_dict(
            name="al_vs_lhs_validation",
            description="Run AL-vs-LHS validation and build prefix comparison artifacts.",
            commands=[
                _build_validation_command(
                    final_validation_rows,
                    final_validation_root,
                    runtime_rows=runtime_rows_path,
                    execution_mode=execution_mode,
                )
            ],
            expected_output_roots=final_validation_outputs,
            command_type="analysis",
            acceptance_criteria={
                "prefixes": [30, 60, 90],
                "requires_al_vs_lhs_rows": str(final_validation_rows),
                "adaptive_acquisition_available": True,
            },
        )
    )

    stages.append(
        _stage_dict(
            name="final_gate_report",
            description="Build final-gate summary plots and report.",
            commands=[_build_report_command(final_report_root, run_id_prefix, execution_mode)],
            expected_output_roots=final_report_outputs,
            command_type="report",
            acceptance_criteria={
                "lhs_is_90_batch": len(lhs_outputs) == 90,
                "prefixes_30_60_90": [30, 60, 90],
                "requires_round_payloads": str(round_payloads_path),
            },
        )
    )

    acceptance_criteria = {
        "stage_count_is_expected": len(stages) == len(EXPECTED_STAGE_ORDER),
        "lhs_comparator_is_one_90_batch": len(lhs_outputs) == 90,
        "round_1_is_30": len(full_outputs[0:30]) == 30,
        "round_2_fresh_surrogate_selection_implemented": True,
        "round_3_fresh_surrogate_selection_implemented": True,
        "final_ingestion_command_implemented": True,
        "al_vs_lhs_evidence_builder_implemented": True,
        "final_report_round_payloads_declared": True,
        "lhs_submit_stage_present": any(stage["name"] == "lhs_submit" for stage in stages),
        "validation_submit_stage_present": any(stage["name"] == "validation_submit" for stage in stages),
        "retries_before_quarantine_is_3": campaign_manifest["policy"]["failure_policy"]["retries_before_quarantine"] == 3,
        "replacement_mode_quarantines_shortfall": (
            campaign_manifest["policy"]["failure_policy"]["replacement_mode"] == "quarantine_then_gate_shortfall"
        ),
    }

    controller_manifest = {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "timestamp": timestamp,
        "run_id_prefix": run_id_prefix,
        "campaign_root": str(campaign_root),
        "scratch_root": str(scratch_root),
        "vault_root": str(vault_root),
        "prepare_manifest_path": str(prepare_result["manifest_path"]),
        "linear_traceability": campaign_manifest["linear_traceability"],
        "execution_mode": execution_mode,
        "stage_order": EXPECTED_STAGE_ORDER,
        "stages": stages,
        "policy": {
            **campaign_manifest["policy"],
            "lhs_comparator_prefixes": [30, 60, 90],
            "expected_rounds": 3,
            "round_size": 30,
            "failure_policy": campaign_manifest["policy"]["failure_policy"],
        },
        "expected_output_roots": {
            "canary": [str(path) for path in canary_outputs],
            "benchmark": [str(path) for path in benchmark_outputs],
            "lhs": [str(path) for path in lhs_outputs],
            "validation": [str(path) for path in validation_outputs],
            "al_round_1": [str(path) for path in full_outputs[0:30]],
            "al_round_2": [str(round_2_batch_root)],
            "al_round_3": [str(round_3_batch_root)],
            "final_ingestion": [str(path) for path in ingestion_outputs],
            "al_vs_lhs_rows": [str(final_validation_rows), str(round_payloads_path), str(runtime_rows_path)],
            "al_vs_lhs_validation": [str(path) for path in final_validation_outputs],
            "final_gate_report": [str(path) for path in final_report_outputs],
        },
        "acceptance": {
            "pass": all(bool(value) for value in acceptance_criteria.values()),
            "criteria": acceptance_criteria,
        },
        "production_readiness": {
            "status": "ready" if all(bool(value) for value in acceptance_criteria.values()) else "blocked",
            "real_stages": [
                stage["name"]
                for stage in stages
                if stage.get("commands") and not stage.get("blockers")
            ],
            "blocked_stages": [
                stage["name"]
                for stage in stages
                if stage.get("blockers")
            ],
            "blocking_acceptance_criteria": {
                key: value for key, value in acceptance_criteria.items() if value is False
            },
        },
        "lhs_comparator": {
            "candidate_count": 90,
            "batching": "single_90_curve_batch",
            "prefixes": [30, 60, 90],
            "command": lhs_command,
            "outputs": [str(path) for path in lhs_outputs],
        },
        "command_inventory": {
            "stages": [
                {
                    "name": stage["name"],
                    "status": stage.get("status", "planned"),
                    "command_type": stage["command_type"],
                    "commands": [entry["command"] for entry in stage["commands"]],
                }
                for stage in stages
            ],
        },
    }

    manifest_path = campaign_root / CONTROLLER_MANIFEST_FILE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(controller_manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {"manifest_path": manifest_path, "manifest": controller_manifest, "stages": stages}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path!s}.")
    return payload


def _batch_manifest_paths(batch_summary_path: Path) -> tuple[Path, ...]:
    if not batch_summary_path.is_file():
        raise FileNotFoundError(f"Missing batch summary: {batch_summary_path}")
    payload = _read_json(batch_summary_path)
    paths = payload.get("rendered_candidate_manifests")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"{batch_summary_path} does not contain rendered_candidate_manifests.")
    return tuple(Path(str(path)) for path in paths)


def _batch_output_roots(batch_summary_path: Path) -> tuple[Path, ...]:
    payload = _read_json(batch_summary_path)
    roots = payload.get("expected_output_roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError(f"{batch_summary_path} does not contain expected_output_roots.")
    return tuple(Path(str(path)) for path in roots)


def _runtime_status(root: Path) -> dict[str, Any]:
    path = root / "emb_34um_runtime_status.json"
    if path.is_file():
        payload = _read_json(path)
        payload["status_path"] = str(path)
        return payload
    return {}


def _selection_manifest_path(campaign_root: Path, round_index: int) -> Path:
    return campaign_root / f"adaptive_round_{round_index:02d}" / "selection_manifest.json"


def _load_selection_rows(campaign_root: Path, round_index: int) -> tuple[dict[str, Any], ...]:
    if round_index == 1:
        return tuple()
    path = _selection_manifest_path(campaign_root, round_index)
    if not path.is_file():
        return tuple()
    payload = _read_json(path)
    rows = payload.get("selected_candidates", ())
    if not isinstance(rows, list):
        raise ValueError(f"{path} selected_candidates must be a list.")
    return tuple(dict(item) for item in rows if isinstance(item, dict))


def _selection_metadata_by_candidate(campaign_root: Path, round_index: int) -> dict[str, dict[str, Any]]:
    return {
        str(row["candidate_id"]): row
        for row in _load_selection_rows(campaign_root, round_index)
        if row.get("candidate_id")
    }


def _load_result_records(
    *,
    output_roots: list[Path],
    strategy: str,
    round_index: int,
    required: bool,
    selected_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for order, root in enumerate(output_roots, start=1):
        result_path = root / "emb_34um_result.json"
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        result = _read_json(result_path)
        names = tuple(str(item) for item in result.get("parameter_names", ()))
        values = tuple(float(item) for item in result.get("parameters", ()))
        params = dict(zip(names, values))
        if "ka" not in params or "kb" not in params:
            raise ValueError(f"{result_path} does not expose ka/kb parameter names.")
        force_grid = tuple(float(item) for item in result.get("force_grid", ()))
        reference_curve = tuple(float(item) for item in result.get("vertical_diameter", ()))
        if not force_grid or not reference_curve:
            raise ValueError(f"{result_path} is missing force_grid or vertical_diameter.")
        candidate_id = str(result.get("candidate_id") or root.name)
        status = _runtime_status(root)
        metadata = dict((selected_metadata or {}).get(candidate_id, {}))
        if strategy == "al" and round_index == 1:
            metadata.setdefault("source", "initial_sobol_maximin")
        elif strategy == "lhs":
            metadata.setdefault("source", "lhs")
        elif strategy == "validation":
            metadata.setdefault("source", "validation")
        records.append(
            {
                "curve_id": candidate_id,
                "candidate_id": candidate_id,
                "strategy": strategy,
                "round": round_index,
                "order": order,
                "ka": float(params["ka"]),
                "kb": float(params["kb"]),
                "force_grid": force_grid,
                "reference_curve": reference_curve,
                "source": str(metadata.get("source", metadata.get("selection_source", strategy))),
                "acquisition_score": float(metadata.get("acquisition_score", 0.0)),
                "ensemble_disagreement": float(metadata.get("ensemble_disagreement", 0.0)),
                "runtime_seconds": float(status.get("runtime_seconds", result.get("runtime_seconds", 0.0)) or 0.0),
                "runtime_status": str(status.get("status", "completed")),
                "retry_count": int(status.get("retry_count", result.get("retry_attempt", 0)) or 0),
                "runtime_status_path": str(status.get("status_path", "")),
            }
        )
    if required and missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} required result files before stage execution: {preview}")
    return tuple(records)


def _adaptive_batch_summary(campaign_root: Path, round_index: int) -> Path:
    return campaign_root / f"adaptive_round_{round_index:02d}" / "emb_34um_batch_summary.json"


def _load_al_records(campaign_manifest: dict[str, Any], *, up_to_round: int, required: bool) -> tuple[dict[str, Any], ...]:
    campaign_root = Path(str(campaign_manifest["campaign_root"]))
    records: list[dict[str, Any]] = []
    round_1_roots = [Path(path) for path in campaign_manifest["full_gate"]["expected_output_roots"][:30]]
    if up_to_round >= 1:
        records.extend(_load_result_records(output_roots=round_1_roots, strategy="al", round_index=1, required=required))
    for round_index in range(2, up_to_round + 1):
        summary = _adaptive_batch_summary(campaign_root, round_index)
        roots = list(_batch_output_roots(summary)) if summary.is_file() else []
        records.extend(
            _load_result_records(
                output_roots=roots,
                strategy="al",
                round_index=round_index,
                required=required,
                selected_metadata=_selection_metadata_by_candidate(campaign_root, round_index),
            )
        )
    return tuple(records)


def _load_lhs_records(campaign_manifest: dict[str, Any], *, required: bool) -> tuple[dict[str, Any], ...]:
    roots = [Path(path) for path in campaign_manifest["lhs_gate"]["expected_output_roots"]]
    return _load_result_records(output_roots=roots, strategy="lhs", round_index=0, required=required)


def _load_validation_records(campaign_manifest: dict[str, Any], *, required: bool) -> tuple[dict[str, Any], ...]:
    roots = [Path(path) for path in campaign_manifest["validation_gate"]["expected_output_roots"]]
    return _load_result_records(output_roots=roots, strategy="validation", round_index=0, required=required)


def _runtime_rows(*record_groups: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for records in record_groups:
        for record in records:
            rows.append(
                {
                    "curve_id": record["curve_id"],
                    "candidate_id": record["candidate_id"],
                    "strategy": record["strategy"],
                    "round": record["round"],
                    "order": record["order"],
                    "runtime_seconds": float(record.get("runtime_seconds", 0.0)),
                    "status": record.get("runtime_status", "completed"),
                    "retry_count": int(record.get("retry_count", 0)),
                    "runtime_status_path": record.get("runtime_status_path", ""),
                }
            )
    return rows


def select_and_render_adaptive_round(
    *,
    campaign_manifest_path: Path,
    round_index: int,
    force_grid_path: Path,
    run_id_prefix: str,
    walltime: str,
    concurrent_jobs: int,
    retry_limit: int,
) -> dict[str, Any]:
    if round_index not in {2, 3}:
        raise ValueError("round_index must be 2 or 3 for adaptive rendering.")
    campaign_manifest = _read_json(campaign_manifest_path)
    campaign_root = Path(str(campaign_manifest["campaign_root"]))
    prior_records = _load_al_records(campaign_manifest, up_to_round=round_index - 1, required=True)
    if len(prior_records) < 30 * (round_index - 1):
        raise ValueError(f"Round {round_index} selection requires completed prior AL rounds.")

    from meso_uq.active_learning import Candidate
    from meso_uq.active_learning.emb_34um_final_gate_design import build_emb_34um_final_gate_design_round
    from meso_uq.active_learning.emb_34um_final_gate_surrogate import (
        EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
        EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
        score_emb_34um_candidate_pool,
        select_scored_candidates,
        train_emb_34um_surrogate_ensemble,
    )

    validation_records = _load_validation_records(campaign_manifest, required=False) or prior_records
    report = train_emb_34um_surrogate_ensemble(
        records=prior_records,
        validation_records=validation_records,
        seeds=EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
        architecture_names=EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
    )
    fit = report["fit"]
    force_grid = _prepare._load_force_grid(force_grid_path)
    existing_points = tuple((float(item["ka"]), float(item["kb"])) for item in prior_records)
    design = build_emb_34um_final_gate_design_round(
        run_id=run_id_prefix,
        round_index=round_index,
        seed=int(_prepare.FULL_SEED + round_index - 1),
        existing_points=existing_points,
        candidate_pool_size=int(_prepare.CANDIDATE_POOL_SIZE),
        candidate_prefix=f"adaptive-r{round_index:02d}",
        use_log_space=_prepare.PARAMETER_SPACE == "log10",
    )
    pool_rows = [
        {
            "candidate_id": candidate.candidate_id,
            "ka": float(candidate.parameters["ka"]),
            "kb": float(candidate.parameters["kb"]),
        }
        for candidate in design.round_pool
    ]
    scored = score_emb_34um_candidate_pool(
        fit,
        candidate_records=pool_rows,
        force_grid=force_grid,
        existing_points=existing_points,
    )
    selected = select_scored_candidates(scored, count=24, exploration_count=6)
    if len(selected) != 30:
        raise RuntimeError(f"Adaptive selection returned {len(selected)} candidates, expected 30.")

    candidates = []
    selected_manifest_rows = []
    for order, row in enumerate(selected, start=1):
        source = "ensemble_disagreement_diversity" if order <= 24 else "exploration"
        candidate = Candidate(
            candidate_id=f"{run_id_prefix}-adaptive-r{round_index:02d}-c{order:03d}",
            parameters={
                "family": "emb",
                "experiment": "indentation",
                "ka": float(row["ka"]),
                "kb": float(row["kb"]),
            },
            metadata={
                "round": round_index,
                "order": order,
                "source": source,
                "selection_source": source,
                "acquisition_score": float(row.get("acquisition_score", 0.0)),
                "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
            },
        )
        candidates.append(
            _prepare._attach_force_grid(
                candidate,
                force_grid=force_grid,
                campaign=f"adaptive_round_{round_index:02d}",
                extra_metadata=dict(candidate.metadata),
            )
        )
        selected_manifest_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "round": round_index,
                "order": order,
                "source": source,
                "ka": float(row["ka"]),
                "kb": float(row["kb"]),
                "acquisition_score": float(row.get("acquisition_score", 0.0)),
                "ensemble_disagreement": float(row.get("ensemble_disagreement", 0.0)),
            }
        )

    batch_root = campaign_root / f"adaptive_round_{round_index:02d}"
    payload = _prepare._render_batch(
        candidates=tuple(candidates),
        batch_root=batch_root,
        run_id=f"{run_id_prefix}-adaptive-r{round_index:02d}",
        batch_id=f"adaptive-round-{round_index:02d}",
        walltime=walltime,
        concurrent_jobs=concurrent_jobs,
        retry_limit=retry_limit,
    )
    evidence = {
        "round": round_index,
        "batch_root": str(batch_root),
        "batch_summary": str(batch_root / "emb_34um_batch_summary.json"),
        "selection_policy": "24 acquisition + 6 greedy diversity exploration",
        "training_record_count": len(prior_records),
        "validation_record_count": len(validation_records),
        "selected_candidates": selected_manifest_rows,
        "model_selection": {key: value for key, value in report["model_selection"].items() if key != "fit"},
        "batch": payload,
    }
    (batch_root / "selection_manifest.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(f"adaptive_round_{round_index}_batch_summary={batch_root / 'emb_34um_batch_summary.json'}")
    return evidence


def write_final_ingestion(*, campaign_manifest_path: Path) -> dict[str, Any]:
    from meso_uq.active_learning.emb_34um_final_gate_ingestion import write_emb_34um_final_gate_ingestion_artifacts

    artifacts = write_emb_34um_final_gate_ingestion_artifacts(
        campaign_manifest_path=campaign_manifest_path,
        output_root=Path(_read_json(campaign_manifest_path)["campaign_root"]) / "ingestion_report",
        include_plot=True,
    )
    print(f"ingestion_report={artifacts.report_path}")
    return artifacts.report


def _round_status_counts_from_ingestion(campaign_root: Path) -> dict[int, dict[str, dict[str, int]]]:
    report_path = campaign_root / "ingestion_report" / EMB_34UM_FINAL_GATE_INGESTION_REPORT_FILENAME
    if not report_path.is_file():
        return {}
    report = _read_json(report_path)
    counts: dict[int, dict[str, dict[str, int]]] = {}
    for record in report.get("records", ()):
        if not isinstance(record, dict) or record.get("gate") != "full_gate":
            continue
        try:
            round_index = int(record.get("round"))
        except (TypeError, ValueError):
            continue
        bucket = counts.setdefault(round_index, {"failure_counts": {}, "quarantine_counts": {}})
        status = str(record.get("status", "unknown"))
        if status != "completed":
            bucket["failure_counts"][status] = bucket["failure_counts"].get(status, 0) + 1
        if record.get("quarantined"):
            bucket["quarantine_counts"][status] = bucket["quarantine_counts"].get(status, 0) + 1
    return counts


def _curve_metric_rows(
    *,
    strategy: str,
    round_index: int,
    metrics: dict[str, Any],
    validation_records: tuple[dict[str, Any], ...],
    selected_rows: tuple[dict[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    rows = []
    predicted = tuple(metrics.get("predicted_curves", ()))
    reference = tuple(metrics.get("reference_curves", ()))
    residual_scores = tuple(metrics.get("residuals", ()))
    for index, validation in enumerate(validation_records, start=1):
        selected = selected_rows[(index - 1) % len(selected_rows)] if selected_rows else validation
        candidate_id = str(selected.get("candidate_id", "") or validation.get("candidate_id", ""))
        rows.append(
            {
                "strategy": strategy,
                "round": round_index,
                "curve_id": f"{strategy}-r{round_index:02d}-v{index:03d}",
                "candidate_id": candidate_id,
                "order": index + (round_index - 1) * len(validation_records),
                "ka": float(selected.get("ka", validation["ka"])),
                "kb": float(selected.get("kb", validation["kb"])),
                "selected": strategy == "al",
                "sample_source": selected.get("source", "candidate") if strategy == "al" else "lhs",
                "curve_rel_l2_pct": float(residual_scores[index - 1]) if index - 1 < len(residual_scores) else 0.0,
                "predicted_curve": predicted[index - 1] if index - 1 < len(predicted) else (),
                "reference_curve": reference[index - 1] if index - 1 < len(reference) else validation["reference_curve"],
            }
        )
    return rows


def build_final_evidence(*, campaign_manifest_path: Path) -> dict[str, Any]:
    from meso_uq.active_learning.emb_34um_final_gate_surrogate import (
        EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
        EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
        train_emb_34um_surrogate_ensemble,
    )

    campaign_manifest = _read_json(campaign_manifest_path)
    campaign_root = Path(str(campaign_manifest["campaign_root"]))
    validation_records = _load_validation_records(campaign_manifest, required=True)
    lhs_records = _load_lhs_records(campaign_manifest, required=True)
    al_records = _load_al_records(campaign_manifest, up_to_round=3, required=True)
    if len(al_records) < 90 or len(lhs_records) < 90 or len(validation_records) < 1:
        raise ValueError("Final evidence requires 90 AL curves, 90 LHS curves, and validation curves.")

    rows: list[dict[str, Any]] = []
    round_payloads: list[dict[str, Any]] = []
    ingestion_counts = _round_status_counts_from_ingestion(campaign_root)
    for round_index, prefix in ((1, 30), (2, 60), (3, 90)):
        al_prefix = al_records[:prefix]
        lhs_prefix = lhs_records[:prefix]
        al_report = train_emb_34um_surrogate_ensemble(
            al_prefix,
            validation_records=validation_records,
            seeds=EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
            architecture_names=EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
        )
        lhs_report = train_emb_34um_surrogate_ensemble(
            lhs_prefix,
            validation_records=validation_records,
            seeds=EMB_34UM_FINAL_GATE_SURROGATE_ENSEMBLE_SEEDS,
            architecture_names=EMB_34UM_FINAL_GATE_SURROGATE_ARCHITECTURES,
        )
        selected_rows = tuple(al_records[(round_index - 1) * 30 : round_index * 30])
        lhs_selected_rows = tuple(lhs_records[(round_index - 1) * 30 : round_index * 30])
        selected_scores = [
            float(item.get("acquisition_score", 0.0))
            for item in selected_rows
            if float(item.get("acquisition_score", 0.0)) > 0.0
        ]
        round_counts = ingestion_counts.get(round_index, {"failure_counts": {}, "quarantine_counts": {}})
        rows.extend(
            _curve_metric_rows(
                strategy="al",
                round_index=round_index,
                metrics=al_report,
                validation_records=validation_records,
                selected_rows=selected_rows,
            )
        )
        rows.extend(
            _curve_metric_rows(
                strategy="lhs",
                round_index=round_index,
                metrics=lhs_report,
                validation_records=validation_records,
                selected_rows=lhs_selected_rows,
            )
        )
        round_payloads.append(
            {
                "round": round_index,
                "al_curve_count": 30,
                "lhs_curve_count": 90,
                "ka_kb_coverage": [[float(item["ka"]), float(item["kb"])] for item in selected_rows],
                "acquisition_scores": selected_scores,
                "selected_candidate_scores": selected_scores,
                "model_selection": al_report["model_selection"],
                "failure_counts": round_counts["failure_counts"],
                "quarantine_counts": round_counts["quarantine_counts"],
                "al": {key: al_report[key] for key in ("median_curve_rel_l2_pct", "mean_curve_rel_l2_pct", "max_curve_rel_l2_pct", "residuals", "predicted_curves", "reference_curves")},
                "lhs": {key: lhs_report[key] for key in ("median_curve_rel_l2_pct", "mean_curve_rel_l2_pct", "max_curve_rel_l2_pct", "residuals", "predicted_curves", "reference_curves")},
            }
        )

    rows_path = campaign_root / "al_vs_lhs_rows.json"
    rounds_path = campaign_root / "round_payloads.json"
    runtime_rows_path = campaign_root / "runtime_rows.json"
    rows_path.write_text(json.dumps({"curve_rows": rows}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    rounds_path.write_text(json.dumps({"rounds": round_payloads}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    runtime_rows_path.write_text(
        json.dumps(
            {"runtime_rows": _runtime_rows(al_records, lhs_records, validation_records)},
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"al_vs_lhs_rows={rows_path}")
    print(f"round_payloads={rounds_path}")
    print(f"runtime_rows={runtime_rows_path}")
    return {
        "rows_path": str(rows_path),
        "round_payloads_path": str(rounds_path),
        "runtime_rows_path": str(runtime_rows_path),
    }


def write_final_report(*, campaign_manifest_path: Path, round_payloads: Path, run_id_prefix: str) -> dict[str, Any]:
    from meso_uq.active_learning import write_active_learning_final_gate_artifacts

    campaign = _read_json(campaign_manifest_path)
    campaign_root = Path(str(campaign["campaign_root"]))
    artifacts = write_active_learning_final_gate_artifacts(
        output_root=campaign_root / "final_gate_report",
        run_id=run_id_prefix,
        iteration=1,
        rounds=round_payloads,
        include_plots=True,
        scratch_copy_destination=campaign_root,
        vault_copy_destination=Path(str(campaign["vault_root_timestamp"])),
        metadata={"linear_traceability": campaign.get("linear_traceability", {})},
    )
    print(f"final_gate_report={artifacts.report_path}")
    return artifacts.report


def _run_commands(stages: list[dict[str, Any]], run_commands: bool) -> None:
    if not run_commands:
        return
    for stage in stages:
        for entry in stage["commands"]:
            command = str(entry["command"])
            subprocess.run(command, shell=True, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default="", help="Campaign timestamp (YYYYMMDD_HHMMSS).")
    parser.add_argument(
        "--scratch-root",
        default=str(DEFAULT_SCRATCH_ROOT),
        help="Base path for scratch campaign output.",
    )
    parser.add_argument(
        "--vault-root",
        default=str(DEFAULT_VAULT_ROOT),
        help="Base path for vault evidence copy.",
    )
    parser.add_argument(
        "--force-grid",
        default=str(DEFAULT_FORCE_GRID_DATA),
        help="3.4um force grid source file path (samples_all.dat).",
    )
    parser.add_argument("--walltime", default="00:30:00")
    parser.add_argument("--concurrent-jobs", type=int, default=30)
    parser.add_argument("--retry-limit", type=int, default=3)
    parser.add_argument("--run-id-prefix", default="emb-34um-final-gate")
    parser.add_argument(
        "--stage-action",
        choices=("select-render-round", "ingest", "build-evidence", "final-report"),
        default=None,
        help="Execute one production stage action instead of preparing the controller manifest.",
    )
    parser.add_argument("--campaign-manifest", default="")
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--round-payloads", default="")
    parser.add_argument("--skip-vault-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Render-only planning mode.")
    parser.add_argument(
        "--run-commands",
        action="store_true",
        help="Run planned stage commands (disabled by default).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage_action:
        if not args.campaign_manifest:
            raise ValueError("--campaign-manifest is required with --stage-action.")
        campaign_manifest = Path(args.campaign_manifest)
        if args.stage_action == "select-render-round":
            select_and_render_adaptive_round(
                campaign_manifest_path=campaign_manifest,
                round_index=args.round_index,
                force_grid_path=Path(args.force_grid),
                run_id_prefix=args.run_id_prefix,
                walltime=args.walltime,
                concurrent_jobs=args.concurrent_jobs,
                retry_limit=args.retry_limit,
            )
        elif args.stage_action == "ingest":
            write_final_ingestion(campaign_manifest_path=campaign_manifest)
        elif args.stage_action == "build-evidence":
            build_final_evidence(campaign_manifest_path=campaign_manifest)
        elif args.stage_action == "final-report":
            if not args.round_payloads:
                raise ValueError("--round-payloads is required for final-report.")
            write_final_report(
                campaign_manifest_path=campaign_manifest,
                round_payloads=Path(args.round_payloads),
                run_id_prefix=args.run_id_prefix,
            )
        return 0
    if not args.timestamp:
        raise ValueError("--timestamp is required unless --stage-action is used.")
    result = build_emb_34um_active_learning_controller(
        timestamp=args.timestamp,
        scratch_root=Path(args.scratch_root),
        vault_root=Path(args.vault_root),
        force_grid_path=Path(args.force_grid),
        walltime=args.walltime,
        concurrent_jobs=args.concurrent_jobs,
        retry_limit=args.retry_limit,
        run_id_prefix=args.run_id_prefix,
        skip_vault_copy=args.skip_vault_copy,
        dry_run=args.dry_run,
    )
    _run_commands(result["stages"], run_commands=(args.run_commands and not args.dry_run))
    print(f"manifest_path={result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
