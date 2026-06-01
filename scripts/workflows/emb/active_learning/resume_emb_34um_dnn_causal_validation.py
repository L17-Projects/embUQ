#!/usr/bin/env python3
"""Inspect and resume EMB 3.4um DNN causal-validation submission stages.

The helper reads the controller manifest from an existing campaign root, inspects
submission-stage batch summaries, counts completed candidate outputs under each
batch, and optionally submits only the missing stage commands while honoring a
max-active-jobs throttle.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


REPO_ROOT = _repo_root()
CONTROLLER_MANIFEST_FILENAME = "emb_34um_dnn_causal_validation_controller_manifest.json"
RESULT_FILENAME = "emb_34um_result.json"
RUNTIME_STATUS_FILENAMES = (
    "emb_34um_runtime_status.json",
    "runtime_status.json",
    "result_status.json",
)
FAILED_RUNTIME_STATES = {"failed", "failure", "error", "timeout", "timed_out"}
RUNNING_RUNTIME_STATES = {"running", "started", "submitted"}
ACTIVE_SLURM_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
MANUAL_CANCELLATION_STATES = {"cancelled", "canceled"}
BLOCKING_RUNTIME_FAILURE_KINDS = {
    "failed",
    "slurm_term",
    "slurm_timeout",
    "timeout",
    "missing_final_output_after_completed_runtime",
    "invalid_runtime_status",
}
RESUME_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_causal_validation_resume.v1"
DEFAULT_MAX_ACTIVE_JOBS = None
REPLACEMENT_MANIFEST_FILENAME = "emb_34um_dnn_causal_validation_replacement_manifest.json"
REPLACEMENT_BATCH_SUMMARY_FILENAME = "emb_34um_dnn_causal_validation_replacement_batch_summary.json"
PILOT_SUBMIT_STAGE_NAME = "pilot_submit"
PILOT_VERIFY_STAGE_NAME = "pilot_verify"
PILOT_VALIDATION_SUMMARY_FILENAME = "emb_34um_dnn_causal_validation_pilot_summary.json"
RUNTIME_FAILURE_GUARD_RECORD_LIMIT = 50
_REPLICA_PATTERN = re.compile(r"^replica-(\d+)$")
_SLURM_JOB_ID_PATTERN = re.compile(r"\bjob\s+(?P<job_id>\d+)\b", re.IGNORECASE)
_SLURM_ACCOUNTING_CACHE: dict[str, dict[str, str] | None] = {}


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _as_path(value: object, *, relative_to: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute() or relative_to is None:
        return path
    return (relative_to / path).resolve()


def _result_path(output_root: Path) -> Path:
    if output_root.suffix == ".json" and output_root.name == RESULT_FILENAME:
        return output_root
    return output_root / RESULT_FILENAME


def _runtime_status_payload(output_root: Path) -> dict[str, Any]:
    for filename in RUNTIME_STATUS_FILENAMES:
        path = output_root / filename
        if not path.is_file():
            continue
        try:
            payload = _load_json_object(path)
        except Exception as exc:  # pragma: no cover - defensive corrupt-file path
            return {
                "status": "invalid",
                "status_path": str(path),
                "error_message": str(exc),
            }
        payload["status_path"] = str(path)
        return payload
    return {"status": "missing", "status_path": ""}


def _runtime_status_label(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status", "")).strip().lower()
    if status in {"completed", "complete", "success", "succeeded"}:
        return "completed"
    if status in MANUAL_CANCELLATION_STATES:
        return "cancelled"
    if status in FAILED_RUNTIME_STATES:
        return "failed"
    if status in RUNNING_RUNTIME_STATES:
        return "running"
    if status:
        return status
    return "missing"


def _extract_slurm_job_id(payload: Mapping[str, Any]) -> str:
    for key in ("slurm_job_id", "job_id", "slurm_job"):
        value = str(payload.get(key, "")).strip()
        if value.isdigit():
            return value
    message = str(payload.get("error_message", ""))
    match = _SLURM_JOB_ID_PATTERN.search(message)
    return match.group("job_id") if match is not None else ""


def _slurm_accounting_for_job(job_id: str) -> dict[str, str] | None:
    job_id = str(job_id).strip()
    if not job_id:
        return None
    if job_id in _SLURM_ACCOUNTING_CACHE:
        return _SLURM_ACCOUNTING_CACHE[job_id]
    if shutil.which("sacct") is None:
        _SLURM_ACCOUNTING_CACHE[job_id] = None
        return None

    result = _safe_slurm_run(
        ["sacct", "-j", job_id, "-X", "-n", "-o", "JobIDRaw,State,Reason,Elapsed,ExitCode", "-P"]
    )
    if int(result.get("returncode", 1)) != 0:
        _SLURM_ACCOUNTING_CACHE[job_id] = None
        return None
    for raw_line in str(result.get("stdout", "")).splitlines():
        parts = raw_line.split("|")
        if len(parts) < 5:
            continue
        raw_job_id, state, reason, elapsed, exit_code = parts[:5]
        if raw_job_id != job_id:
            continue
        record = {
            "job_id": raw_job_id,
            "state": state,
            "reason": reason,
            "elapsed": elapsed,
            "exit_code": exit_code,
        }
        _SLURM_ACCOUNTING_CACHE[job_id] = record
        return record

    _SLURM_ACCOUNTING_CACHE[job_id] = None
    return None


def _runtime_failure_classification(payload: Mapping[str, Any], *, result_exists: bool = True) -> dict[str, Any]:
    runtime_label = _runtime_status_label(payload)
    if runtime_label == "completed" and not result_exists:
        return {
            "kind": "missing_final_output_after_completed_runtime",
            "retry_same_candidate": False,
        }
    if runtime_label == "cancelled":
        return {"kind": "manual_cancellation", "retry_same_candidate": True}
    if runtime_label != "failed":
        return {"kind": runtime_label, "retry_same_candidate": False}

    raw_status = str(payload.get("status", "")).strip().lower()
    if raw_status in MANUAL_CANCELLATION_STATES:
        return {"kind": "manual_cancellation", "retry_same_candidate": True}
    if raw_status in {"timeout", "timed_out"}:
        return {"kind": "timeout", "retry_same_candidate": False}

    job_id = _extract_slurm_job_id(payload)
    accounting = _slurm_accounting_for_job(job_id) if job_id else None
    state = str((accounting or {}).get("state", "")).upper()
    reason = str((accounting or {}).get("reason", ""))
    if "TIMEOUT" in state:
        return {
            "kind": "slurm_timeout",
            "retry_same_candidate": False,
            "slurm_job_id": job_id,
            "slurm_state": str((accounting or {}).get("state", "")),
            "slurm_reason": reason,
        }
    if "CANCELLED" in state:
        return {
            "kind": "manual_cancellation",
            "retry_same_candidate": True,
            "slurm_job_id": job_id,
            "slurm_state": str((accounting or {}).get("state", "")),
            "slurm_reason": reason,
        }
    if any(marker in state for marker in ("PREEMPTED", "NODE_FAIL", "REVOKED")):
        return {
            "kind": "scheduler_interruption",
            "retry_same_candidate": True,
            "slurm_job_id": job_id,
            "slurm_state": str((accounting or {}).get("state", "")),
            "slurm_reason": reason,
        }

    message = str(payload.get("error_message", "")).strip().lower()
    if "time limit" in message or "timeout" in message or "timed out" in message:
        return {
            "kind": "timeout",
            "retry_same_candidate": False,
            "slurm_job_id": job_id,
            "slurm_state": str((accounting or {}).get("state", "")),
            "slurm_reason": reason,
        }
    if "signal term" in message or "terminated" in message:
        return {
            "kind": "slurm_term",
            "retry_same_candidate": False,
            "slurm_job_id": job_id,
            "slurm_state": str((accounting or {}).get("state", "")),
            "slurm_reason": reason,
        }

    return {
        "kind": "failed",
        "retry_same_candidate": False,
        "slurm_job_id": job_id,
        "slurm_state": str((accounting or {}).get("state", "")),
        "slurm_reason": reason,
    }


def _candidate_output_report(output_roots: list[Path]) -> tuple[int, list[int], list[int], list[int], list[dict[str, Any]]]:
    missing_indices: list[int] = []
    failed_indices: list[int] = []
    active_indices: list[int] = []
    statuses: list[dict[str, Any]] = []
    for index, candidate_root in enumerate(output_roots):
        result_exists = _result_path(candidate_root).is_file()
        runtime_payload = _runtime_status_payload(candidate_root)
        runtime_label = _runtime_status_label(runtime_payload)
        failure_classification = _runtime_failure_classification(
            runtime_payload,
            result_exists=result_exists,
        )
        retry_same_candidate = bool(failure_classification.get("retry_same_candidate", False))
        runtime_failure_kind = str(failure_classification.get("kind", ""))
        guard_blocking_failure = runtime_failure_kind in BLOCKING_RUNTIME_FAILURE_KINDS
        active_runtime = runtime_failure_kind == "running" and not result_exists
        if active_runtime:
            active_indices.append(index)
        elif guard_blocking_failure:
            if retry_same_candidate:
                missing_indices.append(index)
            else:
                failed_indices.append(index)
        elif not result_exists:
            missing_indices.append(index)
        statuses.append(
            {
                "candidate_index": index,
                "candidate_id": str(runtime_payload.get("candidate_id", "")),
                "output_root": str(candidate_root),
                "result_path": str(_result_path(candidate_root)),
                "result_exists": result_exists,
                "runtime_status": runtime_label,
                "runtime_status_path": str(runtime_payload.get("status_path", "")),
                "retry_count": runtime_payload.get("retry_count", runtime_payload.get("retry_attempt")),
                "retry_limit": runtime_payload.get("retry_limit"),
                "error_message": str(runtime_payload.get("error_message", "")),
                "runtime_failure_kind": runtime_failure_kind,
                "active_runtime": active_runtime,
                "guard_blocking_failure": guard_blocking_failure and not retry_same_candidate,
                "manual_cancellation": runtime_failure_kind == "manual_cancellation",
                "retry_same_candidate": retry_same_candidate,
                "slurm_job_id": str(failure_classification.get("slurm_job_id", "")),
                "slurm_state": str(failure_classification.get("slurm_state", "")),
                "slurm_reason": str(failure_classification.get("slurm_reason", "")),
            }
        )

    incomplete_indices = sorted(set(missing_indices) | set(failed_indices) | set(active_indices))
    return len(output_roots) - len(incomplete_indices), missing_indices, failed_indices, active_indices, statuses


def _completed_output_count(output_roots: list[Path]) -> tuple[int, list[int]]:
    completed_count, missing_indices, failed_indices, active_indices, _ = _candidate_output_report(output_roots)
    return completed_count, sorted(set(missing_indices) | set(failed_indices) | set(active_indices))


def _quote_export_value(value: object) -> str:
    return shlex.quote(str(value))


def _format_slurm_array_indices(indices: list[int], *, concurrency: int | None = None) -> str:
    unique_indices = sorted({int(index) for index in indices})
    if not unique_indices:
        raise ValueError("At least one Slurm array index is required.")

    ranges: list[str] = []
    start = unique_indices[0]
    previous = unique_indices[0]
    for index in unique_indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = index
        previous = index
    ranges.append(f"{start}-{previous}" if start != previous else str(start))

    value = ",".join(ranges)
    if concurrency is not None and concurrency > 0:
        value = f"{value}%{concurrency}"
    return value


def _parse_slurm_array_concurrency(value: str) -> int | None:
    if "%" not in value:
        return None
    _, concurrency_text = value.rsplit("%", 1)
    try:
        concurrency = int(concurrency_text)
    except ValueError:
        return None
    if concurrency < 1:
        return None
    return concurrency


def _replace_slurm_array_indices(command: str, indices: list[int]) -> str:
    equals_match = re.search(r"(?P<option>--array=)(?P<value>\S+)", command)
    space_match = None if equals_match is not None else re.search(r"(?P<option>--array\s+)(?P<value>\S+)", command)
    match = equals_match or space_match
    if match is None:
        return command

    original_value = str(match.group("value"))
    replacement_value = _format_slurm_array_indices(
        indices,
        concurrency=_parse_slurm_array_concurrency(original_value),
    )
    return f"{command[:match.start('value')]}{replacement_value}{command[match.end('value'):]}"


def _ensure_sbatch_export_all(command: str) -> str:
    match = re.search(r"(?P<option>--export=)(?P<value>\S+)", command)
    if match is None:
        return command
    export_value = str(match.group("value"))
    if export_value == "ALL" or export_value.startswith("ALL,"):
        return command
    replacement_value = f"ALL,{export_value}"
    return f"{command[:match.start('value')]}{replacement_value}{command[match.end('value'):]}"


def _is_sbatch_command(argv: list[str]) -> bool:
    if not argv:
        return False
    executable = Path(argv[0]).name
    return executable == "sbatch"


def _split_sbatch_export_items(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def _normalize_sbatch_export_for_launch(command: str) -> tuple[list[str] | None, dict[str, str]]:
    """Move explicit --export=ALL,KEY=VALUE pairs into the sbatch environment.

    Karolina can intermittently hold large arrays before the batch script starts
    when explicit variables are packed into --export=ALL,KEY=VALUE. Launching
    sbatch with plain --export=ALL while setting those variables in the caller
    environment preserves the job environment without exercising that fragile
    Slurm path.
    """

    try:
        argv = shlex.split(command)
    except ValueError:
        return None, {}
    if not _is_sbatch_command(argv):
        return None, {}

    export_value_index: int | None = None
    export_prefix_index: int | None = None
    export_value = ""
    for index, item in enumerate(argv):
        if item == "--export" and index + 1 < len(argv):
            export_value_index = index + 1
            export_value = argv[index + 1]
            break
        if item.startswith("--export="):
            export_prefix_index = index
            export_value = item.split("=", 1)[1]
            break

    if not export_value.startswith("ALL,"):
        return argv, {}

    environment: dict[str, str] = {}
    for item in _split_sbatch_export_items(export_value[len("ALL,") :]):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        environment[key] = value

    normalized = list(argv)
    if export_value_index is not None:
        normalized[export_value_index] = "ALL"
    elif export_prefix_index is not None:
        normalized[export_prefix_index] = "--export=ALL"

    return normalized, environment


def _synthesize_replacement_submission_commands(
    *,
    replacement_summary_path: Path,
    replacement_summary: Mapping[str, Any],
    expected_output_roots: list[Path],
) -> list[str]:
    if not expected_output_roots:
        return []
    campaign_root = _as_path(replacement_summary.get("campaign_root", ""), relative_to=replacement_summary_path.parent)
    design_manifest_path = campaign_root / "emb_34um_dnn_causal_validation_manifest.json"
    if not campaign_root.is_dir() or not design_manifest_path.is_file():
        return []
    design_manifest = _load_json_object(design_manifest_path)
    command_inventory = design_manifest.get("command_inventory", {})
    command_inventory = command_inventory if isinstance(command_inventory, Mapping) else {}
    walltime = str(command_inventory.get("walltime", "00:12:00"))
    concurrent_jobs = int(command_inventory.get("concurrent_jobs", 30) or 30)
    retry_limit = int(command_inventory.get("retry_limit", 0) or 0)
    candidate_count = len(expected_output_roots)
    array_limit = max(1, min(candidate_count, concurrent_jobs))
    batch_root = _as_path(
        replacement_summary.get("replacement_batch_root", replacement_summary_path.parent),
        relative_to=replacement_summary_path.parent,
    )
    batch_summary = batch_root / "emb_34um_batch_summary.json"
    if not batch_summary.is_file():
        return []

    timestamp = str(design_manifest.get("timestamp", campaign_root.name))
    scratch_root = str(design_manifest.get("scratch_root", campaign_root.parent))
    vault_root = str(design_manifest.get("vault_root_timestamp", design_manifest.get("vault_root", campaign_root.parent / "vault")))
    run_id_prefix = str(design_manifest.get("run_id_prefix", "emb-34um-dnn-causal-validation"))
    stage = str(replacement_summary.get("stage", replacement_summary.get("mode", "replacement")))
    replica = int(replacement_summary.get("replica", 0) or 0)
    cycle = int(replacement_summary.get("cycle", 0) or 0)
    wrapper = REPO_ROOT / "scripts" / "platforms" / "karolina" / "sbatch" / "emb_34um_dnn_causal_validation_array.sbatch"

    export_values = {
        "TIMESTAMP": timestamp,
        "SCRATCH_ROOT": scratch_root,
        "VAULT_ROOT": vault_root,
        "RUN_ID_PREFIX": run_id_prefix,
        "CAMPAIGN_ROOT": str(campaign_root),
        "REPO_ROOT": str(REPO_ROOT),
        "MODE": stage,
        "BATCH_DIR_OVERRIDE": str(batch_root),
        "BATCH_SUMMARY": str(batch_summary),
        "REPLICA": replica,
        "CYCLE": cycle,
        "EXECUTION_MODE": "execute",
        "CONCURRENT_JOBS": concurrent_jobs,
        "RETRY_LIMIT": retry_limit,
        "PYTHON_EXECUTABLE": "python",
    }
    export_arg = "ALL," + ",".join(f"{key}={_quote_export_value(value)}" for key, value in export_values.items())
    return [
        " ".join(
            [
                "sbatch --parsable",
                f"--time={shlex.quote(walltime)}",
                f"--array=0-{candidate_count - 1}%{array_limit}",
                f"--export={export_arg}",
                shlex.quote(str(wrapper)),
            ]
        )
    ]


def _replacement_batches_for_summary(summary_path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    batches_payload = payload.get("replacement_batches")
    if not isinstance(batches_payload, list):
        manifest_path = payload.get("replacement_manifest_path")
        if not manifest_path:
            fallback_manifest = summary_path.parent / REPLACEMENT_MANIFEST_FILENAME
            manifest_path = str(fallback_manifest) if fallback_manifest.is_file() else ""
        if manifest_path:
            manifest = _load_json_object(_as_path(manifest_path, relative_to=summary_path.parent))
            batches_payload = manifest.get("replacement_batches")
    raw_batches: list[dict[str, Any]] = []
    if isinstance(batches_payload, list):
        raw_batches.extend([dict(item) for item in batches_payload if isinstance(item, Mapping)])

    discovered_summary_paths = sorted((summary_path.parent / "replacement").glob(f"batch-*/{REPLACEMENT_BATCH_SUMMARY_FILENAME}"))
    known_batch_paths = {
        str(_as_path(batch.get("replacement_batch_summary_path", ""), relative_to=summary_path.parent))
        for batch in raw_batches
        if isinstance(batch, Mapping)
    }
    for discovered_summary_path in discovered_summary_paths:
        if str(discovered_summary_path) in known_batch_paths:
            continue
        raw_batches.append({"replacement_batch_summary_path": str(discovered_summary_path)})

    if not raw_batches:
        return []

    batches: list[dict[str, Any]] = []
    for raw in raw_batches:
        replacement_summary_path = _as_path(raw.get("replacement_batch_summary_path", ""), relative_to=summary_path.parent)
        raw_expected_output_roots_payload = raw.get("expected_output_roots", [])
        raw_expected_output_roots = [
            _as_path(item, relative_to=summary_path.parent)
            for item in raw_expected_output_roots_payload
            if isinstance(raw_expected_output_roots_payload, list) and str(item)
        ] if isinstance(raw_expected_output_roots_payload, list) else []
        if not replacement_summary_path.is_file():
            completed_count, raw_missing_indices, failed_indices, active_indices, runtime_statuses = _candidate_output_report(
                raw_expected_output_roots
            )
            missing_indices = sorted(set(raw_missing_indices) | set(failed_indices) | set(active_indices))
            original_failed_indices = [int(item) for item in raw.get("failed_candidate_indices", [])]
            batches.append(
                {
                    "replacement_batch_summary_path": str(replacement_summary_path),
                    "failed_candidate_indices": original_failed_indices,
                    "failed_candidate_ids": [str(item) for item in raw.get("failed_candidate_ids", [])],
                    "submission_commands": [str(item) for item in raw.get("replacement_submission_commands", []) if str(item).strip()],
                    "expected_output_roots": [str(path) for path in raw_expected_output_roots],
                    "completed_candidate_count": completed_count,
                    "missing_candidate_indices": missing_indices,
                    "raw_missing_candidate_indices": raw_missing_indices,
                    "failed_replacement_candidate_indices": failed_indices,
                    "active_replacement_candidate_indices": active_indices,
                    "failed_original_candidate_indices": [
                        original_failed_indices[index]
                        for index in failed_indices
                        if 0 <= index < len(original_failed_indices)
                    ],
                    "replacement_runtime_statuses": runtime_statuses,
                    "complete": not missing_indices and bool(raw_expected_output_roots),
                    "status": "missing_replacement_batch_summary",
                }
            )
            continue

        replacement_summary = _load_json_object(replacement_summary_path)
        replacement_expected_output_roots_payload = replacement_summary.get("expected_output_roots", [])
        replacement_expected_output_roots = [
            _as_path(item, relative_to=replacement_summary_path.parent)
            for item in replacement_expected_output_roots_payload
            if isinstance(replacement_expected_output_roots_payload, list) and str(item)
        ] if isinstance(replacement_expected_output_roots_payload, list) else []
        expected_output_roots = raw_expected_output_roots or replacement_expected_output_roots
        completed_count, raw_missing_indices, failed_indices, active_indices, runtime_statuses = _candidate_output_report(
            expected_output_roots
        )
        missing_indices = sorted(set(raw_missing_indices) | set(failed_indices) | set(active_indices))
        submission_commands = [
            str(item)
            for item in replacement_summary.get("replacement_submission_commands", [])
            if str(item).strip()
        ]
        if not submission_commands:
            batch_payload = replacement_summary.get("batch_payload", {})
            if isinstance(batch_payload, Mapping):
                manifest_path = batch_payload.get("manifest_path")
                if manifest_path:
                    batch_manifest = _load_json_object(_as_path(manifest_path, relative_to=replacement_summary_path.parent))
                    submission = batch_manifest.get("scheduler_boundary", {}).get("submission", {})
                    if isinstance(submission, Mapping):
                        submission_commands = [
                            str(item)
                            for item in submission.get("submission_commands", [])
                            if str(item).strip()
                        ]
        synthesized_submission_commands = False
        if not submission_commands:
            submission_commands = _synthesize_replacement_submission_commands(
                replacement_summary_path=replacement_summary_path,
                replacement_summary=replacement_summary,
                expected_output_roots=expected_output_roots,
            )
            synthesized_submission_commands = bool(submission_commands)

        original_failed_indices = [int(item) for item in replacement_summary.get("failed_candidate_indices", [])]
        batches.append(
            {
                "replacement_batch_summary_path": str(replacement_summary_path),
                "failed_candidate_indices": original_failed_indices,
                "failed_candidate_ids": [str(item) for item in replacement_summary.get("failed_candidate_ids", [])],
                "submission_commands": submission_commands,
                "synthesized_submission_commands": synthesized_submission_commands,
                "has_submission_commands": bool(submission_commands),
                "submission_ready": bool(submission_commands)
                and not failed_indices
                and not active_indices
                and str(replacement_summary.get("status", "")).strip() == "replacement_rendered",
                "expected_output_roots": [str(path) for path in expected_output_roots],
                "completed_candidate_count": completed_count,
                "missing_candidate_indices": missing_indices,
                "raw_missing_candidate_indices": raw_missing_indices,
                "failed_replacement_candidate_indices": failed_indices,
                "active_replacement_candidate_indices": active_indices,
                "failed_original_candidate_indices": [
                    original_failed_indices[index]
                    for index in failed_indices
                    if 0 <= index < len(original_failed_indices)
                ],
                "replacement_runtime_statuses": runtime_statuses,
                "complete": not missing_indices and bool(expected_output_roots),
                "status": str(replacement_summary.get("status", "")),
                "runtime_status": str(raw.get("runtime_status", replacement_summary.get("runtime_status", ""))),
            }
        )
    return batches


def _inspect_submission_batch(summary_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "batch_summary_path": str(summary_path),
        "batch_summary_exists": summary_path.is_file(),
        "candidate_count": None,
        "completed_candidate_count": 0,
        "missing_candidate_indices": [],
        "complete": False,
        "ready_for_submission": False,
    }
    if not summary_path.is_file():
        return summary

    payload = _load_json_object(summary_path)
    expected_output_roots = payload.get("expected_output_roots")
    if not isinstance(expected_output_roots, list) or not expected_output_roots:
        raise ValueError(f"{summary_path} must contain a non-empty expected_output_roots list.")

    candidate_count = int(payload.get("candidate_count", len(expected_output_roots)))
    status = str(payload.get("status", ""))
    rendered_candidate_manifests = payload.get("rendered_candidate_manifests")
    rendered_count = len(rendered_candidate_manifests) if isinstance(rendered_candidate_manifests, list) else None
    if (
        status == "adaptive_placeholder_not_rendered"
        or len(expected_output_roots) != candidate_count
        or (rendered_count is not None and rendered_count != candidate_count)
    ):
        summary.update(
            {
                "candidate_count": candidate_count,
                "completed_candidate_count": 0,
                "missing_candidate_indices": list(range(candidate_count)),
                "complete": False,
                "ready_for_submission": False,
                "expected_output_roots": [str(_as_path(item, relative_to=summary_path.parent)) for item in expected_output_roots],
                "status": payload.get("status"),
                "blocked_reason": "batch_not_fully_rendered",
                "rendered_candidate_manifest_count": rendered_count,
                "expected_output_root_count": len(expected_output_roots),
            }
        )
        return summary

    roots = [_as_path(raw_root, relative_to=summary_path.parent) for raw_root in expected_output_roots]
    replacement_batches = _replacement_batches_for_summary(summary_path, payload)
    (
        original_completed_candidate_count,
        original_raw_missing_indices,
        original_failed_indices,
        original_active_indices,
        original_runtime_statuses,
    ) = _candidate_output_report(roots)
    effective_roots = list(roots)
    covered_missing_indices: set[int] = set()
    rendered_replacement_indices: set[int] = set()
    replacement_state_by_original_index: dict[int, str] = {}
    for batch in replacement_batches:
        failed_original_indices = {int(item) for item in batch.get("failed_original_candidate_indices", [])}
        for failed_index in batch.get("failed_candidate_indices", []):
            original_index = int(failed_index)
            rendered_replacement_indices.add(original_index)
            if bool(batch.get("complete", False)):
                replacement_state_by_original_index[original_index] = "complete"
            elif original_index in failed_original_indices:
                replacement_state_by_original_index[original_index] = "failed"
            else:
                replacement_state_by_original_index[original_index] = "pending"
        if not bool(batch.get("complete", False)):
            continue
        batch_roots = [
            _as_path(item, relative_to=summary_path.parent)
            for item in batch.get("expected_output_roots", [])
            if str(item)
        ]
        for failed_index, replacement_root in zip(batch.get("failed_candidate_indices", []), batch_roots):
            index = int(failed_index)
            if 0 <= index < len(effective_roots):
                effective_roots[index] = replacement_root
                covered_missing_indices.add(index)

    completed_candidate_count, raw_missing_indices, failed_indices, active_indices, runtime_statuses = _candidate_output_report(
        effective_roots
    )
    missing_indices = sorted(set(raw_missing_indices) | set(failed_indices) | set(active_indices))
    failed_replacement_indices = {
        original_index
        for original_index, state in replacement_state_by_original_index.items()
        if state == "failed"
    }
    replacement_required_indices = sorted((set(failed_indices) - rendered_replacement_indices) | failed_replacement_indices)
    ready_for_submission = True
    blocked_reason = None
    submission_strategy = "original_batch"
    if active_indices:
        ready_for_submission = False
        blocked_reason = "candidate_runtime_in_progress"
        submission_strategy = "blocked"
    elif replacement_batches:
        has_separate_replacement_commands = any(bool(batch.get("has_submission_commands", False)) for batch in replacement_batches)
        if replacement_required_indices:
            ready_for_submission = False
            blocked_reason = "replacement_required_for_failed_candidates"
            submission_strategy = "replacement_required"
        elif not has_separate_replacement_commands:
            submission_strategy = "stage_batch_with_embedded_replacements"
        elif any(bool(batch.get("submission_ready", False)) for batch in replacement_batches):
            ready_for_submission = True
            submission_strategy = "replacement_batches"
        elif missing_indices:
            ready_for_submission = False
            blocked_reason = "replacement_batches_incomplete"
            submission_strategy = "blocked"
        elif any(bool(batch.get("complete", False)) for batch in replacement_batches):
            ready_for_submission = True
            submission_strategy = "replacement_batches"
    elif failed_indices:
        ready_for_submission = False
        blocked_reason = "replacement_required_for_failed_candidates"
        submission_strategy = "replacement_required"

    summary.update(
        {
            "candidate_count": candidate_count,
            "completed_candidate_count": completed_candidate_count,
            "missing_candidate_indices": missing_indices,
            "raw_missing_candidate_indices": raw_missing_indices,
            "failed_candidate_indices": failed_indices,
            "active_candidate_indices": active_indices,
            "quarantined_candidate_indices": failed_indices,
            "original_completed_candidate_count": original_completed_candidate_count,
            "original_raw_missing_candidate_indices": original_raw_missing_indices,
            "original_failed_candidate_indices": original_failed_indices,
            "original_active_candidate_indices": original_active_indices,
            "original_candidate_runtime_statuses": original_runtime_statuses,
            "replacement_rendered_candidate_indices": sorted(rendered_replacement_indices),
            "replacement_required_candidate_indices": replacement_required_indices,
            "candidate_runtime_statuses": runtime_statuses,
            "complete": not missing_indices,
            "ready_for_submission": ready_for_submission,
            "expected_output_roots": [str(path) for path in effective_roots],
            "status": payload.get("status"),
            "replacement_batches": replacement_batches,
            "covered_missing_candidate_indices": sorted(covered_missing_indices),
            "uncovered_missing_candidate_indices": sorted(set(missing_indices) - rendered_replacement_indices),
            "replacement_batch_count": len(replacement_batches),
            "submission_strategy": submission_strategy,
        }
    )
    if blocked_reason is not None:
        summary["blocked_reason"] = blocked_reason
    return summary


def _load_controller_manifest(campaign_root: Path, controller_manifest_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(controller_manifest_path or campaign_root / CONTROLLER_MANIFEST_FILENAME)
    payload = _load_json_object(manifest_path)
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"Controller manifest {manifest_path} must contain a stages list.")
    return manifest_path, payload


def _default_pilot_validation_summary_path(campaign_root: Path) -> Path:
    return campaign_root / "pilot" / "validation" / PILOT_VALIDATION_SUMMARY_FILENAME


def _pilot_validation_summary_path(campaign_root: Path, controller_manifest: Mapping[str, Any]) -> Path:
    pilot_validation = controller_manifest.get("pilot_validation")
    if isinstance(pilot_validation, Mapping):
        summary_path = str(pilot_validation.get("summary_path", "")).strip()
        if summary_path:
            return _as_path(summary_path, relative_to=campaign_root)

    stages = controller_manifest.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            if str(stage.get("name", "")) != PILOT_VERIFY_STAGE_NAME:
                continue
            expected_output_roots = stage.get("expected_output_roots")
            if isinstance(expected_output_roots, list):
                for output in expected_output_roots:
                    text = str(output).strip()
                    if text:
                        return _as_path(text, relative_to=campaign_root)
            break
    return _default_pilot_validation_summary_path(campaign_root)


def _inspect_pilot_verification(*, campaign_root: Path, controller_manifest: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = _pilot_validation_summary_path(campaign_root, controller_manifest)
    status = "missing"
    passed = False
    blocked_reason = "pilot_validation_summary_missing"

    if summary_path.is_file():
        try:
            payload = _load_json_object(summary_path)
        except Exception:
            status = "invalid_summary"
            blocked_reason = "pilot_validation_summary_invalid_json"
        else:
            decision = payload.get("decision")
            if isinstance(decision, Mapping):
                decision_passed = decision.get("passed")
                if isinstance(decision_passed, bool):
                    passed = decision_passed
                    status = "passed" if passed else "failed"
                    blocked_reason = "pilot_validation_not_passed" if not passed else ""
                else:
                    status = "invalid_decision"
                    blocked_reason = "pilot_validation_decision_missing_passed"
            else:
                status = "invalid_decision"
                blocked_reason = "pilot_validation_decision_missing"

    result = {
        "required": True,
        "summary_path": str(summary_path),
        "exists": summary_path.is_file(),
        "status": status,
        "passed": passed,
    }
    if not passed:
        result["blocked_reason"] = blocked_reason
    return result


def _is_production_submission_stage(stage_name: str) -> bool:
    return stage_name != PILOT_SUBMIT_STAGE_NAME


def _coerce_optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 1:
        return None
    return number


def _resolve_active_replicate_count(controller_manifest: Mapping[str, Any]) -> int | None:
    direct = _coerce_optional_positive_int(controller_manifest.get("active_replicate_count"))
    if direct is not None:
        return direct

    protocol = controller_manifest.get("protocol")
    if isinstance(protocol, Mapping):
        protocol_count = _coerce_optional_positive_int(protocol.get("active_replicate_count"))
        if protocol_count is not None:
            return protocol_count
        protocol_legacy = _coerce_optional_positive_int(protocol.get("replicate_count"))
        if protocol_legacy is not None:
            return protocol_legacy

    design_manifest_path = controller_manifest.get("design_manifest_path")
    if design_manifest_path:
        path = Path(str(design_manifest_path))
        if path.is_file():
            design_manifest = _load_json_object(path)
            policy = design_manifest.get("policy", {})
            protocol = design_manifest.get("protocol", {})
            if isinstance(policy, Mapping):
                policy_count = _coerce_optional_positive_int(policy.get("active_replicate_count"))
                if policy_count is not None:
                    return policy_count
                policy_legacy = _coerce_optional_positive_int(policy.get("replicate_count"))
                if policy_legacy is not None:
                    return policy_legacy
            if isinstance(protocol, Mapping):
                protocol_count = _coerce_optional_positive_int(protocol.get("active_replicate_count"))
                if protocol_count is not None:
                    return protocol_count
                protocol_legacy = _coerce_optional_positive_int(protocol.get("replicate_count"))
                if protocol_legacy is not None:
                    return protocol_legacy
    return None


def _replica_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = _REPLICA_PATTERN.fullmatch(part)
        if match is None:
            continue
        return int(match.group(1))
    return None


def _parse_slurm_elapsed_minutes(value: str) -> float:
    text = str(value).strip()
    if not text:
        return 0.0
    day_count = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        day_count = int(day_text or "0")
    fields = text.split(":")
    if len(fields) == 3:
        hours, minutes, seconds = (int(item) for item in fields)
    elif len(fields) == 2:
        hours = 0
        minutes, seconds = (int(item) for item in fields)
    else:
        hours = int(fields[0])
        minutes = 0
        seconds = 0
    total_seconds = day_count * 24 * 3600 + hours * 3600 + minutes * 60 + seconds
    return float(total_seconds) / 60.0


def _safe_slurm_run(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "command": command,
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
        "command": command,
    }


def audit_stale_jobs(
    *,
    campaign_root: Path,
    user: str,
    lookback_hours: int = 72,
) -> dict[str, Any]:
    campaign_token = campaign_root.name
    lookback_hours = max(1, int(lookback_hours))
    squeue_available = shutil.which("squeue") is not None
    sacct_available = shutil.which("sacct") is not None
    if not squeue_available and not sacct_available:
        return {
            "available": False,
            "status": "slurm_tools_unavailable",
            "campaign_root": str(campaign_root),
            "user": user,
            "lookback_hours": lookback_hours,
            "notes": ["squeue/sacct are not available in PATH; skipping stale-job audit."],
        }

    squeue_observed_jobs: list[dict[str, Any]] = []
    squeue_jobs: list[dict[str, Any]] = []
    stale_pending_jobs: list[dict[str, Any]] = []
    squeue_result: dict[str, Any] | None = None
    if squeue_available:
        squeue_result = _safe_slurm_run(["squeue", "-h", "-u", user, "-o", "%i|%T|%M|%j|%R"])
        if squeue_result["ok"]:
            for raw_line in squeue_result["stdout"].splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                job_id, state, elapsed, job_name, reason = (line.split("|", 4) + ["", "", "", "", ""])[:5]
                record = {
                    "job_id": job_id.strip(),
                    "state": state.strip(),
                    "elapsed": elapsed.strip(),
                    "elapsed_minutes": _parse_slurm_elapsed_minutes(elapsed),
                    "job_name": job_name.strip(),
                    "reason": reason.strip(),
                    "matches_campaign_token": campaign_token in job_name or campaign_token in reason,
                }
                squeue_observed_jobs.append(record)
                if record["state"] not in ACTIVE_SLURM_STATES:
                    continue
                squeue_jobs.append(record)
                if record["state"] in {"PENDING", "CONFIGURING"} and record["elapsed_minutes"] >= 120.0:
                    stale_pending_jobs.append(record)

    sacct_jobs: list[dict[str, Any]] = []
    sacct_result: dict[str, Any] | None = None
    if sacct_available:
        since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%S")
        sacct_result = _safe_slurm_run(
            [
                "sacct",
                "-n",
                "-X",
                "-u",
                user,
                "-S",
                since,
                "-o",
                "JobIDRaw,State,Elapsed,JobName",
                "-P",
            ]
        )
        if sacct_result["ok"]:
            for raw_line in sacct_result["stdout"].splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                job_id, state, elapsed, job_name = (line.split("|", 3) + ["", "", "", ""])[:4]
                sacct_jobs.append(
                    {
                        "job_id": job_id.strip(),
                        "state": state.strip(),
                        "elapsed": elapsed.strip(),
                        "job_name": job_name.strip(),
                        "matches_campaign_token": campaign_token in job_name,
                    }
                )

    notes: list[str] = []
    if squeue_result is not None and not squeue_result["ok"]:
        notes.append(f"squeue failed with returncode={squeue_result['returncode']}.")
    if sacct_result is not None and not sacct_result["ok"]:
        notes.append(f"sacct failed with returncode={sacct_result['returncode']}.")

    return {
        "available": True,
        "status": "ok" if not notes else "partial",
        "campaign_root": str(campaign_root),
        "campaign_token": campaign_token,
        "user": user,
        "lookback_hours": lookback_hours,
        "squeue_available": squeue_available,
        "sacct_available": sacct_available,
        "active_jobs": squeue_jobs,
        "squeue_observed_job_count": len(squeue_observed_jobs),
        "non_active_squeue_job_count": max(0, len(squeue_observed_jobs) - len(squeue_jobs)),
        "stale_pending_jobs": stale_pending_jobs,
        "recent_accounting_jobs": sacct_jobs,
        "notes": notes,
    }


def build_resume_plan(
    *,
    campaign_root: Path,
    controller_manifest_path: Path | None = None,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    manifest_path, controller_manifest = _load_controller_manifest(campaign_root, controller_manifest_path)
    active_replicate_count = _resolve_active_replicate_count(controller_manifest)
    pilot_verification = _inspect_pilot_verification(campaign_root=campaign_root, controller_manifest=controller_manifest)

    stage_reports: list[dict[str, Any]] = []
    commands_to_submit: list[dict[str, Any]] = []
    total_complete_batches = 0
    total_missing_batches = 0
    total_blocked_batches = 0
    total_completed_candidates = 0
    total_expected_candidates = 0

    for stage_index, stage in enumerate(controller_manifest["stages"]):
        if not isinstance(stage, Mapping):
            raise ValueError(f"Controller manifest stage #{stage_index} must be a mapping.")
        if str(stage.get("command_type", "")) != "submission":
            continue

        stage_name = str(stage.get("name", f"stage-{stage_index}"))
        stage_commands = [str(item.get("command", "")) for item in stage.get("commands", []) if isinstance(item, Mapping)]
        expected_output_roots = [Path(str(item)) for item in stage.get("expected_output_roots", []) if str(item)]
        if len(stage_commands) != len(expected_output_roots):
            raise ValueError(
                f"Submission stage {stage_name} has {len(stage_commands)} commands but "
                f"{len(expected_output_roots)} expected output roots."
            )

        command_reports: list[dict[str, Any]] = []
        skipped_inactive_replica_count = 0
        for command_index, (command, summary_path) in enumerate(zip(stage_commands, expected_output_roots)):
            command = _ensure_sbatch_export_all(command)
            replica = _replica_from_path(summary_path)
            if active_replicate_count is not None and replica is not None and replica > active_replicate_count:
                skipped_inactive_replica_count += 1
                continue
            batch_report = _inspect_submission_batch(summary_path)
            if _is_production_submission_stage(stage_name) and not bool(pilot_verification["passed"]) and not bool(batch_report["complete"]):
                batch_report["ready_for_submission"] = False
                batch_report["blocked_reason"] = str(pilot_verification.get("blocked_reason", "pilot_verification_not_passed"))
                batch_report["pilot_verification"] = {
                    "passed": bool(pilot_verification["passed"]),
                    "status": str(pilot_verification.get("status", "")),
                    "summary_path": str(pilot_verification["summary_path"]),
                }
            if batch_report["candidate_count"] is not None:
                total_expected_candidates += int(batch_report["candidate_count"])
                total_completed_candidates += int(batch_report["completed_candidate_count"])
            if batch_report["complete"]:
                total_complete_batches += 1
            elif not batch_report.get("ready_for_submission", False):
                total_blocked_batches += 1
            else:
                total_missing_batches += 1
                stage_candidate_indices = [
                    int(item)
                    for item in batch_report.get("uncovered_missing_candidate_indices", batch_report.get("missing_candidate_indices", []))
                ]
                stage_command = _replace_slurm_array_indices(command, stage_candidate_indices) if stage_candidate_indices else command
                if batch_report.get("submission_strategy") == "replacement_batches":
                    for batch in batch_report.get("replacement_batches", []):
                        if bool(batch.get("complete", False)) or not bool(batch.get("submission_ready", False)):
                            continue
                        for submission_index, replacement_command in enumerate(batch.get("submission_commands", [])):
                            replacement_command = _ensure_sbatch_export_all(str(replacement_command))
                            replacement_candidate_indices = [
                                int(item)
                                for item in batch.get(
                                    "missing_candidate_indices",
                                    batch.get("raw_missing_candidate_indices", []),
                                )
                            ]
                            if replacement_candidate_indices:
                                replacement_command = _replace_slurm_array_indices(
                                    replacement_command,
                                    replacement_candidate_indices,
                                )
                            commands_to_submit.append(
                                {
                                    "stage_name": stage_name,
                                    "command_index": command_index,
                                    "batch_summary_path": str(summary_path),
                                    "replacement_batch_summary_path": str(batch.get("replacement_batch_summary_path", "")),
                                    "replacement_submission_index": submission_index,
                                    "submission_kind": "replacement_batch",
                                    "replacement_array_subset": bool(replacement_candidate_indices),
                                    "replacement_array_candidate_indices": replacement_candidate_indices,
                                    "command": replacement_command,
                                    "batch_report": batch_report,
                                }
                            )
                    if stage_candidate_indices:
                        commands_to_submit.append(
                            {
                                "stage_name": stage_name,
                                "command_index": command_index,
                                "batch_summary_path": str(summary_path),
                                "submission_kind": "stage_batch",
                                "stage_array_subset": True,
                                "stage_array_candidate_indices": stage_candidate_indices,
                                "command": stage_command,
                                "batch_report": batch_report,
                            }
                        )
                else:
                    # Either the original stage is ready, or the replacement work was render-only and
                    # the main stage command must consume the swapped manifests/roots.
                    commands_to_submit.append(
                        {
                            "stage_name": stage_name,
                            "command_index": command_index,
                            "batch_summary_path": str(summary_path),
                            "submission_kind": "stage_batch",
                            "stage_array_subset": bool(stage_candidate_indices),
                            "stage_array_candidate_indices": stage_candidate_indices,
                            "command": stage_command,
                            "batch_report": batch_report,
                        }
                    )
            command_reports.append(
                {
                    "stage_name": stage_name,
                    "command_index": command_index,
                    "command": command,
                    "batch_report": batch_report,
                    "needs_submission": (not batch_report["complete"])
                    and bool(batch_report.get("ready_for_submission", False)),
                    "blocked": (not batch_report["complete"])
                    and not bool(batch_report.get("ready_for_submission", False)),
                }
            )

        stage_reports.append(
            {
                "stage_name": stage_name,
                "command_type": "submission",
                "command_count": len(stage_commands),
                "skipped_inactive_replica_count": skipped_inactive_replica_count,
                "complete_command_count": sum(1 for item in command_reports if item["batch_report"]["complete"]),
                "missing_command_count": sum(1 for item in command_reports if item["needs_submission"]),
                "blocked_command_count": sum(1 for item in command_reports if item["blocked"]),
                "needs_submission": any(item["needs_submission"] for item in command_reports),
                "commands": command_reports,
            }
        )

    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "campaign_root": str(campaign_root),
        "controller_manifest_path": str(manifest_path),
        "controller_execution_mode": str(controller_manifest.get("execution_mode", "")),
        "active_replicate_count": active_replicate_count,
        "pilot_verification": pilot_verification,
        "submission_stage_count": len(stage_reports),
        "complete_batch_count": total_complete_batches,
        "missing_batch_count": total_missing_batches,
        "blocked_batch_count": total_blocked_batches,
        "completed_candidate_count": total_completed_candidates,
        "expected_candidate_count": total_expected_candidates,
        "stages": stage_reports,
        "commands_to_submit": commands_to_submit,
    }


def _runtime_failure_guard_records(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def append_record(record: dict[str, Any]) -> None:
        dedupe_key = (
            str(record.get("stage_name", "")),
            str(record.get("source", "")),
            str(record.get("runtime_status_path", "")),
            str(record.get("output_root", "")),
        )
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        records.append(record)

    def append_status_record(
        *,
        stage_name: str,
        command_index: int,
        batch_report: Mapping[str, Any],
        status: Mapping[str, Any],
        source: str,
        replacement_batch: Mapping[str, Any] | None = None,
        original_indices: list[int] | None = None,
    ) -> None:
        runtime_failure_kind = str(status.get("runtime_failure_kind", ""))
        if not runtime_failure_kind and str(status.get("runtime_status", "")).strip().lower() == "failed":
            runtime_failure_kind = "failed"
        guard_blocking_failure = bool(status.get("guard_blocking_failure", False))
        if not guard_blocking_failure and runtime_failure_kind not in BLOCKING_RUNTIME_FAILURE_KINDS:
            return

        runtime_status_path = str(status.get("runtime_status_path", ""))
        output_root = str(status.get("output_root", ""))

        record = {
            "stage_name": stage_name,
            "command_index": command_index,
            "batch_summary_path": str(batch_report.get("batch_summary_path", "")),
            "source": source,
            "candidate_index": int(status.get("candidate_index", 0) or 0),
            "output_root": output_root,
            "runtime_status_path": runtime_status_path,
            "error_message": str(status.get("error_message", "")),
            "runtime_failure_kind": runtime_failure_kind,
            "candidate_id": str(status.get("candidate_id", "")),
            "slurm_job_id": str(status.get("slurm_job_id", "")),
            "slurm_state": str(status.get("slurm_state", "")),
            "slurm_reason": str(status.get("slurm_reason", "")),
        }
        if replacement_batch is not None:
            replacement_index = int(status.get("candidate_index", 0) or 0)
            record["replacement_batch_summary_path"] = str(
                replacement_batch.get("replacement_batch_summary_path", "")
            )
            record["replacement_candidate_index"] = replacement_index
            if original_indices is not None and 0 <= replacement_index < len(original_indices):
                record["candidate_index"] = original_indices[replacement_index]
        append_record(record)

    campaign_root_text = str(plan.get("campaign_root", "")).strip()
    campaign_root = Path(campaign_root_text) if campaign_root_text else None
    if campaign_root is not None and campaign_root.is_dir():
        status_paths: list[Path] = []
        for filename in RUNTIME_STATUS_FILENAMES:
            status_paths.extend(campaign_root.rglob(filename))
        for status_path in sorted(set(status_paths)):
            output_root = status_path.parent
            try:
                runtime_payload = _load_json_object(status_path)
            except Exception as exc:  # pragma: no cover - defensive corrupt-file path
                runtime_payload = {
                    "status": "invalid",
                    "error_message": str(exc),
                }
            runtime_payload["status_path"] = str(status_path)
            result_exists = _result_path(output_root).is_file()
            failure_classification = _runtime_failure_classification(
                runtime_payload,
                result_exists=result_exists,
            )
            runtime_failure_kind = str(failure_classification.get("kind", ""))
            if runtime_failure_kind not in BLOCKING_RUNTIME_FAILURE_KINDS:
                continue
            append_record(
                {
                    "stage_name": "",
                    "command_index": -1,
                    "batch_summary_path": "",
                    "source": "campaign_runtime_status_scan",
                    "candidate_index": -1,
                    "candidate_id": str(runtime_payload.get("candidate_id", "")),
                    "output_root": str(output_root),
                    "runtime_status_path": str(status_path),
                    "error_message": str(runtime_payload.get("error_message", "")),
                    "runtime_failure_kind": runtime_failure_kind,
                    "slurm_job_id": str(failure_classification.get("slurm_job_id", "")),
                    "slurm_state": str(failure_classification.get("slurm_state", "")),
                    "slurm_reason": str(failure_classification.get("slurm_reason", "")),
                }
            )

    for stage in plan.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        stage_name = str(stage.get("stage_name", ""))
        for command in stage.get("commands", []):
            if not isinstance(command, Mapping):
                continue
            command_index = int(command.get("command_index", 0) or 0)
            batch_report = command.get("batch_report")
            if not isinstance(batch_report, Mapping):
                continue
            for status in batch_report.get("original_candidate_runtime_statuses", []):
                if not isinstance(status, Mapping):
                    continue
                append_status_record(
                    stage_name=stage_name,
                    command_index=command_index,
                    batch_report=batch_report,
                    status=status,
                    source="original_stage_batch",
                )
            for status in batch_report.get("candidate_runtime_statuses", []):
                if not isinstance(status, Mapping):
                    continue
                append_status_record(
                    stage_name=stage_name,
                    command_index=command_index,
                    batch_report=batch_report,
                    status=status,
                    source="stage_batch",
                )
            for replacement_batch in batch_report.get("replacement_batches", []):
                if not isinstance(replacement_batch, Mapping):
                    continue
                original_indices = [
                    int(item)
                    for item in replacement_batch.get("failed_candidate_indices", [])
                ]
                for status in replacement_batch.get("replacement_runtime_statuses", []):
                    if not isinstance(status, Mapping):
                        continue
                    append_status_record(
                        stage_name=stage_name,
                        command_index=command_index,
                        batch_report=batch_report,
                        status=status,
                        source="replacement_batch",
                        replacement_batch=replacement_batch,
                        original_indices=original_indices,
                    )
    return records


def _apply_runtime_failure_guard(
    plan: dict[str, Any],
    *,
    allow_runtime_failure_resume: bool = False,
) -> dict[str, Any]:
    records = _runtime_failure_guard_records(plan)
    triggered = bool(records)
    status = "clear"
    if triggered:
        status = "override_allowed" if allow_runtime_failure_resume else "blocked"

    guard = {
        "status": status,
        "triggered": triggered,
        "allow_runtime_failure_resume": bool(allow_runtime_failure_resume),
        "failure_count": len(records),
        "record_limit": RUNTIME_FAILURE_GUARD_RECORD_LIMIT,
        "records": records[:RUNTIME_FAILURE_GUARD_RECORD_LIMIT],
        "truncated_failure_count": max(0, len(records) - RUNTIME_FAILURE_GUARD_RECORD_LIMIT),
    }
    if triggered:
        guard["blocked_reason"] = "runtime_failure_requires_investigation_before_resume"
        guard["message"] = (
            "Runtime failures are present in the campaign. Resume/replacement submission is blocked "
            "until the failure is investigated and an explicit override is supplied."
        )

    plan["runtime_failure_guard"] = guard
    if triggered and not allow_runtime_failure_resume:
        plan["commands_blocked_by_runtime_failure_guard"] = list(plan.get("commands_to_submit", []))
        plan["commands_to_submit"] = []
    else:
        plan["commands_blocked_by_runtime_failure_guard"] = []
    return guard


def _load_repair_module() -> Any:
    script_path = REPO_ROOT / "scripts" / "workflows" / "emb" / "active_learning" / "repair_emb_34um_dnn_causal_validation_resume.py"
    spec = importlib.util.spec_from_file_location("repair_emb_34um_dnn_causal_validation_resume", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load replacement helper from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repair_stage_name(*, controller_stage_name: str, batch_summary_path: Path) -> str | None:
    for part in reversed(batch_summary_path.parts):
        if part in {"shared_initial", "unseen_test"}:
            return part
        if re.fullmatch(r"(?:al|lhs)-step-\d{2}", part):
            return part
        if part == "pilot":
            return None

    normalized = controller_stage_name.replace("_submit", "").replace("-submit", "")
    if normalized in {"shared_initial", "unseen_test"} or re.fullmatch(r"(?:al|lhs)-step-\d{2}", normalized):
        return normalized
    return None


def _repair_cycle(stage_name: str) -> int | None:
    match = re.fullmatch(r"(?:al|lhs)-step-(\d{2})", stage_name)
    if match is None:
        return None
    return int(match.group(1))


def render_replacements_for_resume_plan(
    plan: Mapping[str, Any],
    *,
    repair_module: Any | None = None,
    stage_names: set[str] | None = None,
) -> dict[str, Any]:
    repair = repair_module or _load_repair_module()
    campaign_root = Path(str(plan["campaign_root"]))
    stage_filter = {str(item) for item in stage_names or set() if str(item)}
    rendered: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for stage in plan.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        controller_stage_name = str(stage.get("stage_name", ""))
        if stage_filter and controller_stage_name not in stage_filter:
            continue
        for command in stage.get("commands", []):
            if not isinstance(command, Mapping):
                continue
            batch_report = command.get("batch_report")
            if not isinstance(batch_report, Mapping):
                continue
            replacement_required_indices = batch_report.get("replacement_required_candidate_indices")
            replacement_index_source = (
                replacement_required_indices
                if isinstance(replacement_required_indices, list)
                else batch_report.get(
                    "quarantined_candidate_indices",
                    batch_report.get("failed_candidate_indices", []),
                )
            )
            failed_indices = [
                int(item)
                for item in replacement_index_source
            ]
            if not failed_indices or batch_report.get("submission_strategy") != "replacement_required":
                continue

            batch_summary_path = Path(str(batch_report.get("batch_summary_path", "")))
            repair_stage = _repair_stage_name(
                controller_stage_name=controller_stage_name,
                batch_summary_path=batch_summary_path,
            )
            if repair_stage is None:
                skipped.append(
                    {
                        "stage_name": controller_stage_name,
                        "batch_summary_path": str(batch_summary_path),
                        "failed_candidate_indices": failed_indices,
                        "reason": "replacement_stage_not_supported",
                    }
                )
                continue

            replica = _replica_from_path(batch_summary_path) or 1
            payload = repair.render_al_stage_replacements(
                campaign_root=campaign_root,
                replica=int(replica),
                cycle=_repair_cycle(repair_stage),
                stage=repair_stage,
                failed_candidate_indices=tuple(failed_indices),
                failure_reason="runtime_failure_or_quarantine",
            )
            rendered.append(
                {
                    "stage_name": controller_stage_name,
                    "repair_stage": repair_stage,
                    "batch_summary_path": str(batch_summary_path),
                    "failed_candidate_indices": failed_indices,
                    "replacement_manifest_path": str(payload.get("replacement_manifest_path", "")),
                    "replacement_batch_summary_path": str(payload.get("replacement_batch_summary_path", "")),
                    "replacement_count": len(payload.get("replacement_records", [])),
                    "replacement_submission_commands": list(payload.get("replacement_submission_commands", [])),
                }
            )

    return {
        "rendered_count": len(rendered),
        "skipped_count": len(skipped),
        "stage_name_filter": sorted(stage_filter),
        "rendered": rendered,
        "skipped": skipped,
    }


def _active_job_count(*, user: str) -> int:
    proc = subprocess.run(
        ["squeue", "-h", "-u", user, "-t", "PENDING,RUNNING,CONFIGURING", "-o", "%i"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"squeue failed: stdout={proc.stdout!r} stderr={proc.stderr!r}")
    return sum(1 for line in (proc.stdout or "").splitlines() if line.strip())


def _submit_missing_commands(
    commands_to_submit: list[dict[str, Any]],
    *,
    user: str,
    max_active_jobs: int,
) -> dict[str, Any]:
    if max_active_jobs < 1:
        raise ValueError("max_active_jobs must be a positive integer.")

    submitted: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    reserved_jobs = 0
    for entry in commands_to_submit:
        active_jobs = _active_job_count(user=user)
        if active_jobs + reserved_jobs >= max_active_jobs:
            deferred.append(
                {
                    **entry,
                    "deferred_reason": "active_job_limit_reached",
                    "active_job_count": active_jobs,
                    "reserved_jobs": reserved_jobs,
                }
            )
            continue

        argv, submission_environment = _normalize_sbatch_export_for_launch(str(entry["command"]))
        if argv is None:
            run_command: str | list[str] = str(entry["command"])
            run_shell = True
        else:
            run_command = argv
            run_shell = False

        run_environment = None
        if submission_environment:
            run_environment = dict(os.environ)
            run_environment.update(submission_environment)

        proc = subprocess.run(
            run_command,
            shell=run_shell,
            text=True,
            capture_output=True,
            check=False,
            env=run_environment,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Submission failed for {entry['stage_name']}[{entry['command_index']}]: "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        reserved_jobs += 1
        submitted.append(
            {
                **entry,
                "stdout": (proc.stdout or "").strip(),
                "stderr": (proc.stderr or "").strip(),
                "active_job_count": active_jobs,
                "reserved_jobs": reserved_jobs,
                "launch_command": shlex.join(argv) if argv is not None else str(entry["command"]),
                "launch_export_strategy": "environment_plus_export_all" if submission_environment else "original",
                "launch_environment_keys": sorted(submission_environment),
            }
        )

    return {
        "submitted_commands": submitted,
        "deferred_commands": deferred,
        "submitted_count": len(submitted),
        "deferred_count": len(deferred),
    }


def _filter_commands_to_submit(
    commands_to_submit: list[dict[str, Any]],
    *,
    stage_names: set[str] | None = None,
    submission_kinds: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_filter = {str(item) for item in stage_names or set() if str(item)}
    kind_filter = {str(item) for item in submission_kinds or set() if str(item)}
    selected: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for entry in commands_to_submit:
        stage_name = str(entry.get("stage_name", ""))
        submission_kind = str(entry.get("submission_kind", ""))
        if stage_filter and stage_name not in stage_filter:
            filtered.append({**entry, "filtered_reason": "stage_name_filter"})
            continue
        if kind_filter and submission_kind not in kind_filter:
            filtered.append({**entry, "filtered_reason": "submission_kind_filter"})
            continue
        selected.append(entry)
    return selected, filtered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--controller-manifest")
    parser.add_argument("--max-active-jobs", type=int, default=DEFAULT_MAX_ACTIVE_JOBS)
    parser.add_argument("--user", default=None, help="User name passed to squeue; defaults to the current user.")
    parser.add_argument(
        "--audit-stale-jobs",
        action="store_true",
        help="Audit active/pending Slurm jobs and recent accounting records before resume/submit.",
    )
    parser.add_argument("--stale-lookback-hours", type=int, default=72)
    parser.add_argument(
        "--render-replacements",
        action="store_true",
        help="Render replacement batches for failed/quarantined candidates before submitting missing work.",
    )
    parser.add_argument(
        "--allow-runtime-failure-resume",
        action="store_true",
        help=(
            "Allow replacement rendering/submission even when failed DPD runtime statuses are present. "
            "Use only after the failure has been investigated and resume is explicitly approved."
        ),
    )
    parser.add_argument(
        "--stage-name",
        action="append",
        default=[],
        help="Limit submitted commands to this controller stage name; can be supplied multiple times.",
    )
    parser.add_argument(
        "--submission-kind",
        action="append",
        default=[],
        choices=("stage_batch", "replacement_batch"),
        help="Limit submitted commands to this submission kind; can be supplied multiple times.",
    )
    parser.add_argument("--submit", action="store_true", help="Submit missing commands while respecting the job limit.")
    parser.add_argument("--dry-run", action="store_true", help="Render a plan without submitting commands.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign_root = Path(args.campaign_root)
    controller_manifest_path = Path(args.controller_manifest) if args.controller_manifest else None
    plan = build_resume_plan(campaign_root=campaign_root, controller_manifest_path=controller_manifest_path)
    user = str(args.user or os.environ.get("USER") or getpass.getuser())

    if args.audit_stale_jobs:
        plan["stale_job_audit"] = audit_stale_jobs(
            campaign_root=campaign_root,
            user=user,
            lookback_hours=int(args.stale_lookback_hours),
        )

    runtime_failure_guard = _apply_runtime_failure_guard(
        plan,
        allow_runtime_failure_resume=bool(args.allow_runtime_failure_resume),
    )

    if (
        args.render_replacements
        and not args.dry_run
        and (
            not bool(runtime_failure_guard["triggered"])
            or bool(args.allow_runtime_failure_resume)
        )
    ):
        stale_job_audit = plan.get("stale_job_audit")
        replacement_result = render_replacements_for_resume_plan(
            plan,
            stage_names=set(args.stage_name),
        )
        plan = build_resume_plan(campaign_root=campaign_root, controller_manifest_path=controller_manifest_path)
        if stale_job_audit is not None:
            plan["stale_job_audit"] = stale_job_audit
        plan["rendered_replacements"] = replacement_result
        runtime_failure_guard = _apply_runtime_failure_guard(
            plan,
            allow_runtime_failure_resume=bool(args.allow_runtime_failure_resume),
        )
    elif args.render_replacements:
        plan["rendered_replacements"] = {
            "rendered_count": 0,
            "skipped_count": 0,
            "dry_run": True,
            "note": "Replacement rendering skipped because --dry-run was supplied.",
        }
        if bool(runtime_failure_guard["triggered"]) and not bool(args.allow_runtime_failure_resume):
            plan["rendered_replacements"].update(
                {
                    "blocked_by_runtime_failure_guard": True,
                    "dry_run": bool(args.dry_run),
                    "note": "Replacement rendering skipped because runtime failures require investigation first.",
                }
            )
    else:
        plan["rendered_replacements"] = {"rendered_count": 0, "skipped_count": 0}

    filtered_commands: list[dict[str, Any]] = []
    if args.stage_name or args.submission_kind:
        selected_commands, filtered_commands = _filter_commands_to_submit(
            list(plan["commands_to_submit"]),
            stage_names=set(args.stage_name),
            submission_kinds=set(args.submission_kind),
        )
        plan["commands_to_submit"] = selected_commands
    plan["filtered_commands"] = filtered_commands
    plan["filtered_count"] = len(filtered_commands)

    if args.submit and not args.dry_run:
        manifest = _load_json_object(Path(plan["controller_manifest_path"]))
        max_active_jobs = args.max_active_jobs
        if max_active_jobs is None:
            max_active_jobs = int(manifest.get("command_inventory", {}).get("concurrent_jobs", 1))
        submission_result = _submit_missing_commands(
            plan["commands_to_submit"],
            user=user,
            max_active_jobs=int(max_active_jobs),
        )
        plan.update(submission_result)
        plan["active_job_user"] = user
        plan["max_active_jobs"] = int(max_active_jobs)
    else:
        plan["submitted_count"] = 0
        plan["deferred_count"] = 0
        plan["submitted_commands"] = []
        plan["deferred_commands"] = []

    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
