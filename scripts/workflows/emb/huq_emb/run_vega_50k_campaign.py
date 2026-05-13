#!/usr/bin/env python3
"""Launch the real Vega HUQ-EMB rebuild path.

This runner does not use the local `run_workflow_matrix.py` path for production.
Instead it submits the DNN rebuild and the 50k production lanes through the Vega
sbatch wrappers, then delegates paper asset generation to the existing
postprocess-only campaign runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.vega_workflows import (  # noqa: E402
    VegaWorkflowSelection,
    parse_selection,
    resolve_workflow_config_path,
    selection_key,
    selection_slug,
)

DEFAULT_SELECTIONS = (
    "compression:full-model:production",
    "compression:reduced-model:production",
    "indentation:full-model:production",
    "indentation:reduced-model:production",
)

COMPLETE_WRAPPERS = {
    ("compression", "full-model"): REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch" / "production" / "complete_inference_compression.sbatch",
    ("compression", "reduced-model"): REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch" / "production" / "complete_reduced_compression.sbatch",
    ("indentation", "full-model"): REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch" / "production" / "complete_inference_indentation.sbatch",
    ("indentation", "reduced-model"): REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch" / "production" / "complete_reduced_indentation.sbatch",
}

TRAINING_WRAPPER = REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch" / "train_dnn_surrogates.sbatch"
MAP_WRAPPER = REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch" / "workflow_map.sbatch"
MAP_MIRHEO_WRAPPER = REPO_ROOT / "scripts" / "platforms" / "vega" / "sbatch" / "workflow_map_mirheo.sbatch"
POSTPROCESS_RUNNER = REPO_ROOT / "scripts" / "workflows" / "emb" / "huq_emb" / "run_paper_data_campaign.py"
MAP_MIRHEO_TIME_LIMITS = {
    "compression": "02:00:00",
    "indentation": "00:45:00",
}

TRAINING_DATA = {
    "compression": (
        REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.1um" / "data" / "F_Delta.dat",
        REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.9um" / "data" / "F_Delta.dat",
        REPO_ROOT / "compression" / "surrogate" / "diameters" / "3.0um" / "data" / "F_Delta.dat",
    ),
    "indentation": (
        REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.2um" / "data" / "samples_all.dat",
        REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.4um" / "data" / "samples_all.dat",
        REPO_ROOT / "indentation" / "surrogate" / "diameters" / "5.8um" / "data" / "samples_all.dat",
    ),
}

REFERENCE_DATA = {
    "compression": (
        REPO_ROOT / "compression" / "evalkit" / "data" / "compression_data_2.1um.dat",
        REPO_ROOT / "compression" / "evalkit" / "data" / "compression_data_2.9um.dat",
        REPO_ROOT / "compression" / "evalkit" / "data" / "compression_data_3.0um.dat",
    ),
    "indentation": (
        REPO_ROOT / "indentation" / "evalkit" / "data" / "indentation_data_3.2um.dat",
        REPO_ROOT / "indentation" / "evalkit" / "data" / "indentation_data_3.4um.dat",
        REPO_ROOT / "indentation" / "evalkit" / "data" / "indentation_data_5.8um.dat",
    ),
}


def _sbatch_export_arg(env: dict[str, str]) -> str:
    explicit = sorted(key for key, value in env.items() if os.environ.get(key) != value)
    parts = ["ALL", *(f"{key}={env[key]}" for key in explicit)]
    return ",".join(parts)


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lane_root(workflow_root: Path, selection: VegaWorkflowSelection) -> Path:
    return workflow_root / "runs" / selection.experiment / selection.model_family / selection.profile


def _selection_env(
    *,
    selection: VegaWorkflowSelection,
    lane_root: Path,
    python_bin: str,
    phase2_cpu_ranks: int,
    campaign_id: str,
    logs_dir: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "EXPERIMENT": selection.experiment,
            "MODEL_FAMILY": selection.model_family,
            "PROFILE": selection.profile,
            "OUTPUT_DIR": str(lane_root),
            "CONFIG_PATH": str(resolve_workflow_config_path(REPO_ROOT, selection)),
            "PYTHON_BIN": python_bin,
            "PHASE2_BACKEND": "native-cuda",
            "PHASE2_CPU_RANKS": str(phase2_cpu_ranks),
            "RUN_TAG": campaign_id,
            "LOGS_DIR": str(logs_dir),
        }
    )
    return env


def _submit_sbatch(
    *,
    script: Path,
    env: dict[str, str],
    logs_root: Path,
    name: str,
    wait: bool = True,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    logs_root.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_root / f"{name}.stdout.log"
    stderr_log = logs_root / f"{name}.stderr.log"
    command = ["sbatch", "--parsable", "--output", str(logs_root / f"{name}_%j.out"), "--error", str(logs_root / f"{name}_%j.err")]
    if wait:
        command.insert(1, "--wait")
    command.extend(["--export", _sbatch_export_arg(env)])
    if extra_args:
        command.extend(extra_args)
    command.append(str(script))
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    payload: dict[str, Any] = {
        "name": name,
        "script": str(script),
        "command": command,
        "returncode": result.returncode,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "job_id": (result.stdout or "").strip() or None,
    }
    if result.returncode != 0:
        payload["stderr_tail"] = (result.stderr or "")[-2000:]
    return payload


def _wait_for_jobs(job_ids: list[str], *, timeout_seconds: int = 7 * 24 * 3600, poll_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    remaining = set(job_ids)
    while remaining and time.time() < deadline:
        proc = subprocess.run(
            ["squeue", "-h", "-j", ",".join(sorted(remaining)), "-o", "%i %T"],
            text=True,
            capture_output=True,
            check=False,
        )
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        active = {line.split()[0] for line in lines}
        remaining = {job_id for job_id in remaining if job_id in active}
        if remaining:
            time.sleep(poll_seconds)
    if remaining:
        raise TimeoutError(f"Timed out waiting for jobs: {', '.join(sorted(remaining))}")


def _job_status(job_id: str) -> dict[str, str]:
    proc = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "-n", "-P"],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        current_job_id, state, exit_code = line.split("|", 2)
        if current_job_id == job_id:
            return {"state": state, "exit_code": exit_code}
    return {"state": "UNKNOWN", "exit_code": "unknown"}


def _batch_wait_and_validate(submissions: list[dict[str, Any]], report: dict[str, Any], logs_root: Path) -> bool:
    if not submissions:
        return True
    _wait_for_jobs([str(item["job_id"]) for item in submissions if item.get("job_id")])
    ok = True
    for item in submissions:
        status = _job_status(str(item["job_id"]))
        item["job_state"] = status["state"]
        item["job_exit_code"] = status["exit_code"]
        if status["state"] != "COMPLETED" or status["exit_code"] != "0:0":
            ok = False
    report_path = logs_root / "vega_50k_campaign_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ok


def _snapshot_inputs(*, snapshot_root: Path, selections: list[VegaWorkflowSelection]) -> dict[str, Any]:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    seen: set[Path] = set()
    for selection in selections:
        for path in TRAINING_DATA[selection.experiment]:
            seen.add(path)
        for path in REFERENCE_DATA[selection.experiment]:
            seen.add(path)
        seen.add(resolve_workflow_config_path(REPO_ROOT, selection))
    seen.add(REPO_ROOT / "extern" / "mirheo.lock.json")

    for path in sorted(seen):
        relative = path.relative_to(REPO_ROOT)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(
            {
                "source": str(path),
                "snapshot": str(destination),
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "created_at_utc": _now_iso(),
        "repo_root": str(REPO_ROOT),
        "snapshot_root": str(snapshot_root),
        "files": copied,
    }
    manifest_path = snapshot_root / "inputs_snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "files": copied}


def _run_postprocess(*, paper_data_root: Path, campaign_id: str, python_bin: str, force_rebuild_assets: bool, logs_root: Path) -> dict[str, Any]:
    command = [
        python_bin,
        str(POSTPROCESS_RUNNER),
        "--paper-data-root",
        str(paper_data_root),
        "--campaign-id",
        campaign_id,
        "--python-bin",
        python_bin,
        "--site",
        "vega",
        "--skip-workflow",
    ]
    if force_rebuild_assets:
        command.append("--force-rebuild-assets")
    result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    stdout_log = logs_root / "postprocess.stdout.log"
    stderr_log = logs_root / "postprocess.stderr.log"
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    return {
        "name": "postprocess",
        "command": command,
        "returncode": result.returncode,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the real Vega 50k HUQ-EMB rebuild campaign.")
    parser.add_argument("--paper-data-root", required=True)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--selection", action="append", default=[])
    parser.add_argument("--phase2-cpu-ranks", type=int, default=64)
    parser.add_argument("--skip-surrogate-training", action="store_true", default=False)
    parser.add_argument("--skip-postprocess", action="store_true", default=False)
    parser.add_argument("--force-rebuild-assets", action="store_true", default=False)
    args = parser.parse_args(argv)

    campaign_id = args.campaign_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paper_data_root = _resolve_path(args.paper_data_root)
    runs_root = paper_data_root / "runs" / campaign_id
    workflow_root = runs_root / "workflow_matrix"
    training_root = runs_root / "surrogate_training"
    inputs_snapshot_root = paper_data_root / "inputs_snapshot" / campaign_id
    logs_root = paper_data_root / "logs" / campaign_id
    manifests_root = paper_data_root / "manifests"
    for path in (workflow_root, training_root, inputs_snapshot_root, logs_root, manifests_root):
        path.mkdir(parents=True, exist_ok=True)

    selections = [parse_selection(item) for item in (args.selection or list(DEFAULT_SELECTIONS))]
    for selection in selections:
        if selection.profile != "production":
            raise ValueError(f"This runner only supports production selections. Got {selection_key(selection)}")

    report: dict[str, Any] = {
        "created_at_utc": _now_iso(),
        "repo_root": str(REPO_ROOT),
        "paper_data_root": str(paper_data_root),
        "campaign_id": campaign_id,
        "workflow_root": str(workflow_root),
        "training_root": str(training_root),
        "inputs_snapshot_root": str(inputs_snapshot_root),
        "status": "running",
        "steps": [],
    }

    snapshot = _snapshot_inputs(snapshot_root=inputs_snapshot_root, selections=selections)
    report["inputs_snapshot"] = snapshot

    if not args.skip_surrogate_training:
        train_env = dict(os.environ)
        train_env.update(
            {
                "REPO_ROOT": str(REPO_ROOT),
                "OUTPUT_ROOT": str(training_root),
                "RUN_TAG": campaign_id,
                "PYTHON_BIN": args.python_bin,
                "MAX_EPOCH": "100",
                "RESUME": "true",
            }
        )
        step = _submit_sbatch(
            script=TRAINING_WRAPPER,
            env=train_env,
            logs_root=logs_root / "surrogate_training",
            name="surrogate_training",
        )
        report["steps"].append(step)
        if step["returncode"] != 0:
            report["status"] = "failed"
            (logs_root / "vega_50k_campaign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return 1

    complete_submissions: list[dict[str, Any]] = []
    for selection in selections:
        lane_root = _lane_root(workflow_root, selection)
        lane_logs = logs_root / "workflows" / selection_slug(selection)
        lane_logs.mkdir(parents=True, exist_ok=True)
        env = _selection_env(
            selection=selection,
            lane_root=lane_root,
            python_bin=args.python_bin,
            phase2_cpu_ranks=args.phase2_cpu_ranks,
            campaign_id=campaign_id,
            logs_dir=lane_logs,
        )
        complete_wrapper = COMPLETE_WRAPPERS[(selection.experiment, selection.model_family)]
        complete_step = _submit_sbatch(
            script=complete_wrapper,
            env=env,
            logs_root=lane_logs,
            name="workflow_complete",
            wait=False,
        )
        entry = {"selection": selection_key(selection), **complete_step}
        report["steps"].append(entry)
        complete_submissions.append(entry)
        if complete_step["returncode"] != 0:
            report["status"] = "failed"
            (logs_root / "vega_50k_campaign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return 1
    if not _batch_wait_and_validate(complete_submissions, report, logs_root):
        report["status"] = "failed"
        return 1

    map_phase1_submissions: list[dict[str, Any]] = []
    for stage in ("phase1", "phase3b"):
        current_batch: list[dict[str, Any]] = []
        for selection in selections:
            lane_logs = logs_root / "workflows" / selection_slug(selection)
            lane_root = _lane_root(workflow_root, selection)
            env = _selection_env(
                selection=selection,
                lane_root=lane_root,
                python_bin=args.python_bin,
                phase2_cpu_ranks=args.phase2_cpu_ranks,
                campaign_id=campaign_id,
                logs_dir=lane_logs,
            )
            map_env = dict(env)
            map_env["STAGE"] = stage
            map_env["GPU_TIME_LIMIT"] = "00:15:00"
            step = _submit_sbatch(
                script=MAP_WRAPPER,
                env=map_env,
                logs_root=lane_logs,
                name=f"map_{stage}",
                wait=False,
            )
            entry = {"selection": selection_key(selection), **step}
            report["steps"].append(entry)
            current_batch.append(entry)
            if step["returncode"] != 0:
                report["status"] = "failed"
                (logs_root / "vega_50k_campaign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                return 1
        if not _batch_wait_and_validate(current_batch, report, logs_root):
            report["status"] = "failed"
            return 1

    mirheo_submissions: list[dict[str, Any]] = []
    for selection in selections:
        lane_logs = logs_root / "workflows" / selection_slug(selection)
        lane_root = _lane_root(workflow_root, selection)
        env = _selection_env(
            selection=selection,
            lane_root=lane_root,
            python_bin=args.python_bin,
            phase2_cpu_ranks=args.phase2_cpu_ranks,
            campaign_id=campaign_id,
            logs_dir=lane_logs,
        )
        mirheo_env = dict(env)
        mirheo_env["GPU_TIME_LIMIT"] = MAP_MIRHEO_TIME_LIMITS.get(selection.experiment, "02:00:00")
        step = _submit_sbatch(
            script=MAP_MIRHEO_WRAPPER,
            env=mirheo_env,
            logs_root=lane_logs,
            name="map_mirheo",
            wait=False,
        )
        entry = {"selection": selection_key(selection), **step}
        report["steps"].append(entry)
        mirheo_submissions.append(entry)
        if step["returncode"] != 0:
            report["status"] = "failed"
            (logs_root / "vega_50k_campaign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return 1
    if not _batch_wait_and_validate(mirheo_submissions, report, logs_root):
        report["status"] = "failed"
        return 1

    if not args.skip_postprocess:
        step = _run_postprocess(
            paper_data_root=paper_data_root,
            campaign_id=campaign_id,
            python_bin=args.python_bin,
            force_rebuild_assets=args.force_rebuild_assets,
            logs_root=logs_root,
        )
        report["steps"].append(step)
        if step["returncode"] != 0:
            report["status"] = "failed"
            (logs_root / "vega_50k_campaign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return 1

    report["status"] = "passed"
    report["completed_at_utc"] = _now_iso()
    (logs_root / "vega_50k_campaign_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Vega 50k campaign report: {logs_root / 'vega_50k_campaign_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
