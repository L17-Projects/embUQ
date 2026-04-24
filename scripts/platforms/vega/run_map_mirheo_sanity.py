#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.production_sanity import resolve_production_sanity_selections  # noqa: E402
from meso_uq.vega_workflows import VegaWorkflowSelection, selection_key, selection_slug  # noqa: E402

DEFAULT_NUMSTEPS = 200
DEFAULT_NUMSTEPS_EQ = 200
DEFAULT_N_DISPLACEMENTS = 1
DEFAULT_TIME_LIMIT = "00:10:00"
DEFAULT_CANARY_ROOT = REPO_ROOT / "_runtime_validation" / "gpu_canaries"
SBATCH_TEMPLATE = REPO_ROOT / "scripts" / "vega" / "sbatch" / "workflow_map_mirheo.sbatch"


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _selection_run_root(canary_root: Path, selection: VegaWorkflowSelection) -> Path:
    if selection.experiment == "compression" and selection.model_family == "full-model":
        lane_dir = canary_root / "compression_full_retry"
    elif selection.experiment == "compression" and selection.model_family == "reduced-model":
        lane_dir = canary_root / "compression_reduced"
    elif selection.experiment == "indentation" and selection.model_family == "full-model":
        lane_dir = canary_root / "indentation_full"
    elif selection.experiment == "indentation" and selection.model_family == "reduced-model":
        lane_dir = canary_root / "indentation_reduced"
    else:  # pragma: no cover
        raise ValueError(f"Unsupported selection: {selection_key(selection)}")

    return lane_dir / "runs" / selection.experiment / selection.model_family / selection.profile


def _submit_job(
    *,
    selection: VegaWorkflowSelection,
    output_dir: Path,
    python_bin: str,
    n_displacements: int,
    numsteps: int,
    numsteps_eq: int,
    time_limit: str,
) -> str:
    env = {
        **os.environ,
        "REPO_ROOT": str(REPO_ROOT),
        "EXPERIMENT": selection.experiment,
        "MODEL_FAMILY": selection.model_family,
        "PROFILE": selection.profile,
        "OUTPUT_DIR": str(output_dir),
        "PYTHON_BIN": python_bin,
        "MAP_MIRHEO_N_DISPLACEMENTS": str(n_displacements),
        "MAP_MIRHEO_NUMSTEPS": str(numsteps),
        "MAP_MIRHEO_NUMSTEPS_EQ": str(numsteps_eq),
        "GPU_TIME_LIMIT": time_limit,
    }
    command = [
        "sbatch",
        "--parsable",
        "--partition=dev",
        f"--time={time_limit}",
        str(SBATCH_TEMPLATE),
    ]
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {selection_key(selection)}: "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    return (proc.stdout or "").strip()


def _wait_for_jobs(job_ids: list[str], *, timeout_seconds: int = 900, poll_seconds: int = 10) -> dict[str, dict[str, str]]:
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
        active_ids = {line.split()[0] for line in lines}
        remaining = {job_id for job_id in remaining if job_id in active_ids}
        if remaining:
            time.sleep(poll_seconds)

    sacct = subprocess.run(
        ["sacct", "-j", ",".join(job_ids), "--format=JobID,State,ExitCode", "-n", "-P"],
        text=True,
        capture_output=True,
        check=False,
    )
    records: dict[str, dict[str, str]] = {}
    for line in (sacct.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        job_id, state, exit_code = line.split("|", 2)
        if "." in job_id:
            continue
        records[job_id] = {"state": state, "exit_code": exit_code}
    return records


def _manifest_status(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "map_mirheo" / "map_mirheo_manifest.json"
    payload: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "exists": manifest_path.is_file(),
    }
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["status"] = data.get("status")
        payload["diameters"] = data.get("diameters", [])
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit and verify MAP Mirheo GPU sanity canaries against existing workflow outputs."
    )
    parser.add_argument("--selection", action="append", default=[])
    parser.add_argument("--all-lanes", action="store_true", default=False)
    parser.add_argument("--canary-root", type=str, default=str(DEFAULT_CANARY_ROOT))
    parser.add_argument("--report-root", type=str, default=None)
    parser.add_argument("--python-bin", type=str, default="python")
    parser.add_argument("--n-displacements", type=int, default=DEFAULT_N_DISPLACEMENTS)
    parser.add_argument("--numsteps", type=int, default=DEFAULT_NUMSTEPS)
    parser.add_argument("--numsteps-eq", type=int, default=DEFAULT_NUMSTEPS_EQ)
    parser.add_argument("--time-limit", type=str, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)

    selections = resolve_production_sanity_selections(args.selection, all_lanes=args.all_lanes)
    canary_root = _resolve_path(args.canary_root)
    report_root = (
        _resolve_path(args.report_root)
        if args.report_root is not None
        else REPO_ROOT / "_runtime_validation" / "map_mirheo_sanity"
    )
    report_root.mkdir(parents=True, exist_ok=True)

    submissions: list[dict[str, object]] = []
    for selection in selections:
        output_dir = _selection_run_root(canary_root, selection)
        phase3b_manifest = output_dir / "map_phase3b" / "phase3b_map_manifest.json"
        if not phase3b_manifest.is_file():
            raise FileNotFoundError(
                f"Missing phase3b MAP manifest for {selection_key(selection)}: {phase3b_manifest}"
            )
        job_id = _submit_job(
            selection=selection,
            output_dir=output_dir,
            python_bin=args.python_bin,
            n_displacements=args.n_displacements,
            numsteps=args.numsteps,
            numsteps_eq=args.numsteps_eq,
            time_limit=args.time_limit,
        )
        submissions.append(
            {
                "selection": selection,
                "job_id": job_id,
                "output_dir": output_dir,
            }
        )

    job_records = _wait_for_jobs(
        [entry["job_id"] for entry in submissions],
        timeout_seconds=args.timeout_seconds,
    )

    lane_reports: list[dict[str, object]] = []
    overall_pass = True
    for entry in submissions:
        selection = entry["selection"]
        job_id = entry["job_id"]
        output_dir = entry["output_dir"]
        manifest = _manifest_status(output_dir)
        job_record = job_records.get(job_id, {"state": "UNKNOWN", "exit_code": "unknown"})
        lane_pass = job_record["state"] == "COMPLETED" and manifest.get("status") == "passed"
        overall_pass = overall_pass and lane_pass
        lane_reports.append(
            {
                "selection": selection_key(selection),
                "slug": selection_slug(selection),
                "job_id": job_id,
                "job_state": job_record["state"],
                "job_exit_code": job_record["exit_code"],
                "output_dir": str(output_dir),
                "manifest": manifest,
                "status": "passed" if lane_pass else "failed",
            }
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "canary_root": str(canary_root),
        "n_displacements": args.n_displacements,
        "numsteps": args.numsteps,
        "numsteps_eq": args.numsteps_eq,
        "time_limit": args.time_limit,
        "status": "passed" if overall_pass else "failed",
        "lanes": lane_reports,
    }
    report_path = report_root / "map_mirheo_sanity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"MAP Mirheo sanity report: {report_path}")
    print(f"MAP Mirheo sanity status: {report['status']}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
