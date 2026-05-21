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
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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
RESUME_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_causal_validation_resume.v1"
DEFAULT_MAX_ACTIVE_JOBS = None


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

    roots: list[Path] = []
    missing_indices: list[int] = []
    for index, raw_root in enumerate(expected_output_roots):
        candidate_root = _as_path(raw_root, relative_to=summary_path.parent)
        roots.append(candidate_root)
        if not _result_path(candidate_root).is_file():
            missing_indices.append(index)

    completed_candidate_count = candidate_count - len(missing_indices)
    summary.update(
        {
            "candidate_count": candidate_count,
            "completed_candidate_count": completed_candidate_count,
            "missing_candidate_indices": missing_indices,
            "complete": not missing_indices,
            "ready_for_submission": True,
            "expected_output_roots": [str(path) for path in roots],
            "status": payload.get("status"),
        }
    )
    return summary


def _load_controller_manifest(campaign_root: Path, controller_manifest_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(controller_manifest_path or campaign_root / CONTROLLER_MANIFEST_FILENAME)
    payload = _load_json_object(manifest_path)
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"Controller manifest {manifest_path} must contain a stages list.")
    return manifest_path, payload


def build_resume_plan(
    *,
    campaign_root: Path,
    controller_manifest_path: Path | None = None,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    manifest_path, controller_manifest = _load_controller_manifest(campaign_root, controller_manifest_path)

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
        for command_index, (command, summary_path) in enumerate(zip(stage_commands, expected_output_roots)):
            batch_report = _inspect_submission_batch(summary_path)
            if batch_report["candidate_count"] is not None:
                total_expected_candidates += int(batch_report["candidate_count"])
                total_completed_candidates += int(batch_report["completed_candidate_count"])
            if batch_report["complete"]:
                total_complete_batches += 1
            elif not batch_report.get("ready_for_submission", False):
                total_blocked_batches += 1
            else:
                total_missing_batches += 1
                commands_to_submit.append(
                    {
                        "stage_name": stage_name,
                        "command_index": command_index,
                        "batch_summary_path": str(summary_path),
                        "command": command,
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
        "submission_stage_count": len(stage_reports),
        "complete_batch_count": total_complete_batches,
        "missing_batch_count": total_missing_batches,
        "blocked_batch_count": total_blocked_batches,
        "completed_candidate_count": total_completed_candidates,
        "expected_candidate_count": total_expected_candidates,
        "stages": stage_reports,
        "commands_to_submit": commands_to_submit,
    }


def _active_job_count(*, user: str) -> int:
    proc = subprocess.run(
        ["squeue", "-h", "-u", user, "-o", "%i"],
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

        proc = subprocess.run(
            entry["command"],
            shell=True,
            text=True,
            capture_output=True,
            check=False,
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
            }
        )

    return {
        "submitted_commands": submitted,
        "deferred_commands": deferred,
        "submitted_count": len(submitted),
        "deferred_count": len(deferred),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--controller-manifest")
    parser.add_argument("--max-active-jobs", type=int, default=DEFAULT_MAX_ACTIVE_JOBS)
    parser.add_argument("--user", default=None, help="User name passed to squeue; defaults to the current user.")
    parser.add_argument("--submit", action="store_true", help="Submit missing commands while respecting the job limit.")
    parser.add_argument("--dry-run", action="store_true", help="Render a plan without submitting commands.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign_root = Path(args.campaign_root)
    controller_manifest_path = Path(args.controller_manifest) if args.controller_manifest else None
    plan = build_resume_plan(campaign_root=campaign_root, controller_manifest_path=controller_manifest_path)

    if args.submit and not args.dry_run:
        user = str(args.user or os.environ.get("USER") or getpass.getuser())
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
