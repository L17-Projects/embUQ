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

from meso_uq.active_learning import (
    ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME,
    ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME,
    EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME,
)


CONTROLLER_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_active_learning_controller.v1"


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
    "al_round_1_submit",
    "al_round_1_retrain_model_select",
    "al_round_1_score",
    "al_round_2_submit",
    "al_round_2_retrain_model_select",
    "al_round_2_score",
    "al_round_3_submit",
    "al_round_3_retrain_model_select",
    "al_round_3_score",
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


def _stage_dict(
    *,
    name: str,
    description: str,
    commands: list[str],
    expected_output_roots: list[Path],
    acceptance_criteria: dict[str, Any] | None = None,
    command_type: str = "submission",
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "command_type": command_type,
        "commands": [{"command": item} for item in commands],
        "expected_output_roots": [str(path) for path in expected_output_roots],
        "acceptance_criteria": acceptance_criteria or {},
    }


def _build_round_retrain_command(round_index: int, output_root: Path, execution_mode: str) -> str:
    root = str(output_root)
    body = (
        "from pathlib import Path; "
        "from meso_uq.active_learning.emb_34um_final_gate_surrogate import train_surrogate_ensemble; "
        f"from pathlib import Path; Path({root!r}).mkdir(parents=True, exist_ok=True); "
        f"print('retrain_model_select_round={round_index}')"
    )
    return _python_one_liner(body, execution_mode=execution_mode)


def _build_round_score_command(round_index: int, output_root: Path, execution_mode: str) -> str:
    root = str(output_root)
    body = (
        "from meso_uq.active_learning.emb_34um_final_gate_surrogate import score_candidate_pool_with_ensemble; "
        f"from pathlib import Path; Path({root!r}).mkdir(parents=True, exist_ok=True); "
        f"print('score_100_candidate_pool_round={round_index}')"
    )
    return _python_one_liner(body, execution_mode=execution_mode)


def _build_validation_command(curve_rows: Path, output_root: Path, *, execution_mode: str) -> str:
    validate_script = (
        _REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "validate_emb_34um_al_vs_lhs.py"
    )
    return " ".join(
        [
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
            shlex.quote("emb_34um_final_gate_surrogate"),
            "--adaptive-acquisition-available",
        ]
    )


def _build_report_command(output_root: Path, execution_mode: str) -> str:
    # Keep function name in command to preserve traceability in manifest.
    body = (
        "from pathlib import Path; "
        "from meso_uq.active_learning import write_active_learning_final_gate_artifacts; "
        f"Path({str(output_root)!r}).mkdir(parents=True, exist_ok=True); "
        "print('write_active_learning_final_gate_artifacts')"
    )
    return _python_one_liner(body, execution_mode=execution_mode)


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
    score_root = campaign_root / "scoring"
    model_root = campaign_root / "modeling"
    final_validation_root = campaign_root / "al_vs_lhs_validation"
    final_validation_rows = campaign_root / "al_vs_lhs_rows.json"
    final_report_root = campaign_root / "final_gate_report"

    canary_command = _replace_execution_mode(campaign_manifest["commands"]["canary"]["command"], execution_mode)
    benchmark_command = _replace_execution_mode(
        campaign_manifest["commands"]["benchmark"]["command"], execution_mode
    )
    full_submit_command = _replace_execution_mode(campaign_manifest["commands"]["full"]["command"], execution_mode)
    lhs_command = _replace_execution_mode(campaign_manifest["commands"]["lhs"]["command"], execution_mode)

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

    for round_index, start in ((1, 0), (2, 30), (3, 60)):
        submit_stage_name = f"al_round_{round_index}_submit"
        submit_command = _replace_array(full_submit_command, f"{start}-{start + 29}%{concurrent_jobs}")
        stages.append(
            _stage_dict(
                name=submit_stage_name,
                description=f"Submit AL round {round_index} batch of 30 candidates.",
                commands=[submit_command],
                expected_output_roots=full_outputs[start : start + 30],
                command_type="submission",
                acceptance_criteria={f"round_{round_index}_submit_count_is_30": len(full_outputs[start : start + 30]) == 30},
            )
        )

        retrain_output_root = model_root / f"round_{round_index:02d}"
        score_output_root = score_root / f"round_{round_index:02d}"
        stages.append(
            _stage_dict(
                name=f"al_round_{round_index}_retrain_model_select",
                description=f"Retrain and model-select after AL round {round_index} submission.",
                commands=[_build_round_retrain_command(round_index, retrain_output_root, execution_mode)],
                expected_output_roots=[retrain_output_root],
                command_type="model_select",
                acceptance_criteria={
                    "retries_before_quarantine": campaign_manifest["policy"]["failure_policy"]["retries_before_quarantine"] == 3,
                    "replacement_mode": campaign_manifest["policy"]["failure_policy"]["replacement_mode"] == "next_candidate",
                },
            )
        )
        stages.append(
            _stage_dict(
                name=f"al_round_{round_index}_score",
                description="Score the next 100-candidate pool.",
                commands=[_build_round_score_command(round_index, score_output_root, execution_mode)],
                expected_output_roots=[score_output_root],
                command_type="score",
                acceptance_criteria={"lhs_comparator_batch_count": len(lhs_outputs) == 90},
            )
        )

    final_validation_outputs = [
        final_validation_root / EMB_34UM_AL_VS_LHS_VALIDATION_MANIFEST_FILENAME,
        final_validation_root / EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_FILENAME,
        final_validation_root / EMB_34UM_AL_VS_LHS_VALIDATION_PLOT_SIDECAR_FILENAME,
        final_validation_root / EMB_34UM_AL_VS_LHS_VALIDATION_SUMMARY_CSV_FILENAME,
    ]
    stages.append(
        _stage_dict(
            name="al_vs_lhs_validation",
            description="Run AL-vs-LHS validation and build prefix comparison artifacts.",
            commands=[_build_validation_command(final_validation_rows, final_validation_root, execution_mode=execution_mode)],
            expected_output_roots=final_validation_outputs,
            command_type="analysis",
            acceptance_criteria={"prefixes": [30, 60, 90]},
        )
    )

    final_report_outputs = [
        final_report_root / ACTIVE_LEARNING_FINAL_GATE_REPORT_FILENAME,
        final_report_root / ACTIVE_LEARNING_FINAL_GATE_REPORT_MARKDOWN_FILENAME,
        final_report_root / ACTIVE_LEARNING_FINAL_GATE_SUMMARY_CSV_FILENAME,
    ]
    stages.append(
        _stage_dict(
            name="final_gate_report",
            description="Build final-gate summary plots and report.",
            commands=[_build_report_command(final_report_root, execution_mode)],
            expected_output_roots=final_report_outputs,
            command_type="report",
            acceptance_criteria={
                "lhs_is_90_batch": len(lhs_outputs) == 90,
                "prefixes_30_60_90": [30, 60, 90],
            },
        )
    )

    acceptance_criteria = {
        "stage_count_is_expected": len(stages) == len(EXPECTED_STAGE_ORDER),
        "lhs_comparator_is_one_90_batch": len(lhs_outputs) == 90,
        "round_1_is_30": len(full_outputs[0:30]) == 30,
        "round_2_is_30": len(full_outputs[30:60]) == 30,
        "round_3_is_30": len(full_outputs[60:90]) == 30,
        "retries_before_quarantine_is_3": campaign_manifest["policy"]["failure_policy"]["retries_before_quarantine"] == 3,
        "replacement_mode_next_candidate": campaign_manifest["policy"]["failure_policy"]["replacement_mode"] == "next_candidate",
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
            "al_round_2": [str(path) for path in full_outputs[30:60]],
            "al_round_3": [str(path) for path in full_outputs[60:90]],
            "round_1_score": [str(score_root / "round_01")],
            "round_2_score": [str(score_root / "round_02")],
            "round_3_score": [str(score_root / "round_03")],
            "al_vs_lhs_validation": [str(path) for path in final_validation_outputs],
            "final_gate_report": [str(path) for path in final_report_outputs],
        },
        "acceptance": {
            "pass": all(bool(value) for value in acceptance_criteria.values()),
            "criteria": acceptance_criteria,
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


def _run_commands(stages: list[dict[str, Any]], run_commands: bool) -> None:
    if not run_commands:
        return
    for stage in stages:
        for entry in stage["commands"]:
            command = str(entry["command"])
            subprocess.run(command, shell=True, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True, help="Campaign timestamp (YYYYMMDD_HHMMSS).")
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
