#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.hpc_paths import default_runs_root, detect_hpc_site  # noqa: E402

SBATCH_TEMPLATE = REPO_ROOT / "scripts" / "vega" / "sbatch" / "train_dnn_arch_array.sbatch"
ARRAY_SIZE = 12
MANIFEST_SCHEMA_VERSION = 1

SPECS = [
    {
        "name": "compression_2.1um",
        "experiment": "compression",
        "diameter": "2.1",
        "script": REPO_ROOT / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "best": REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.1um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "compression_2.9um",
        "experiment": "compression",
        "diameter": "2.9",
        "script": REPO_ROOT / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "best": REPO_ROOT / "compression" / "surrogate" / "diameters" / "2.9um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "compression_3.0um",
        "experiment": "compression",
        "diameter": "3.0",
        "script": REPO_ROOT / "compression" / "surrogate" / "scripts" / "train_multi_arch.py",
        "best": REPO_ROOT / "compression" / "surrogate" / "diameters" / "3.0um" / "trained" / "microbubble_force_BEST.pkl",
    },
    {
        "name": "indentation_3.2um",
        "experiment": "indentation",
        "diameter": "3.2",
        "script": REPO_ROOT / "indentation" / "surrogate" / "scripts" / "train_multi_arch.py",
        "best": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.2um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
    {
        "name": "indentation_3.4um",
        "experiment": "indentation",
        "diameter": "3.4",
        "script": REPO_ROOT / "indentation" / "surrogate" / "scripts" / "train_multi_arch.py",
        "best": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "3.4um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
    {
        "name": "indentation_5.8um",
        "experiment": "indentation",
        "diameter": "5.8",
        "script": REPO_ROOT / "indentation" / "surrogate" / "scripts" / "train_multi_arch.py",
        "best": REPO_ROOT / "indentation" / "surrogate" / "diameters" / "5.8um" / "trained" / "microbubble_displacement_BEST.pkl",
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    payload: dict[str, Any] = {"path": str(resolved)}
    if resolved.exists() and resolved.is_file():
        payload["sha256"] = _sha256_path(resolved)
        payload["size_bytes"] = resolved.stat().st_size
    else:
        payload["missing"] = True
        payload["sha256"] = None
    return payload


def _write_report(path: Path, payload: dict[str, object]) -> None:
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_map(report: dict[str, object]) -> dict[str, dict[str, object]]:
    runs = report.get("runs", [])
    if not isinstance(runs, list):
        return {}
    mapping: dict[str, dict[str, object]] = {}
    for item in runs:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            mapping[item["name"]] = item
    return mapping


def _load_report(path: Path, output_root: Path, args: argparse.Namespace) -> dict[str, object]:
    if path.exists():
        report = _read_json(path)
        if "runs" not in report or not isinstance(report["runs"], list):
            report["runs"] = []
    else:
        report = {"schema_version": 2, "started_at": _now_iso(), "runs": []}
    report["output_root"] = str(output_root)
    report["config"] = {
        "max_epoch": args.max_epoch,
        "compression_batch_size": args.compression_batch_size,
        "compression_lr": args.compression_lr,
        "indentation_batch_size": args.indentation_batch_size,
        "indentation_lr": args.indentation_lr,
        "array_size": ARRAY_SIZE,
        "timeout_seconds": args.timeout_seconds,
    }
    return report


def _select_specs(specs: list[dict[str, Path | str]], *, start_from: str | None, only: list[str]) -> list[dict[str, Path | str]]:
    selected = list(specs)
    if only:
        wanted = set(only)
        selected = [spec for spec in selected if str(spec["name"]) in wanted]
    if start_from is not None:
        names = [str(spec["name"]) for spec in selected]
        if start_from not in names:
            raise ValueError(f"--start-from={start_from!r} not found in selected specs. Available: {', '.join(names)}")
        selected = selected[names.index(start_from):]
    return selected


def _spec_params(spec: dict[str, Path | str], args: argparse.Namespace) -> tuple[int, float]:
    name = str(spec["name"])
    if name.startswith("compression_"):
        return args.compression_batch_size, args.compression_lr
    return args.indentation_batch_size, args.indentation_lr


def _spec_paths(output_root: Path, spec_name: str) -> tuple[Path, Path, Path]:
    spec_root = output_root / spec_name
    models_dir = spec_root / "models"
    report_path = spec_root / "training_report.json"
    manifest_path = spec_root / "dnn_training_manifest.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, report_path, manifest_path


def _spec_data_path(spec: dict[str, Path | str]) -> Path:
    experiment = str(spec["experiment"])
    diameter = str(spec["diameter"])
    if experiment == "compression":
        return (
            REPO_ROOT
            / "compression"
            / "surrogate"
            / "diameters"
            / f"{diameter}um"
            / "data"
            / "F_Delta.dat"
        )
    return (
        REPO_ROOT
        / "indentation"
        / "surrogate"
        / "diameters"
        / f"{diameter}um"
        / "data"
        / "samples_all.dat"
    )


def _build_spec_manifest(
    *,
    spec: dict[str, Path | str],
    args: argparse.Namespace,
    models_dir: Path,
    report_path: Path,
    run_entry: dict[str, Any],
) -> dict[str, Any]:
    batch_size, lr = _spec_params(spec, args)
    finalize = run_entry.get("finalize")
    finalize_logs: dict[str, str] = {}
    if isinstance(finalize, dict):
        if isinstance(finalize.get("stdout_log"), str):
            finalize_logs["stdout"] = str(finalize["stdout_log"])
        if isinstance(finalize.get("stderr_log"), str):
            finalize_logs["stderr"] = str(finalize["stderr_log"])

    report_payload: dict[str, Any] = {}
    if report_path.exists():
        try:
            report_payload = _read_json(report_path)
        except Exception:
            report_payload = {}

    best_artifact_path = Path(str(report_payload.get("best_dest", spec["best"])))
    training_results_path = report_payload.get("training_results_csv")
    summary_plot_path = report_payload.get("summary_plot")

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "name": str(spec["name"]),
        "experiment": str(spec["experiment"]),
        "diameter_um": str(spec["diameter"]),
        "status": run_entry.get("status"),
        "timestamps": {
            "started_at": run_entry.get("started_at"),
            "completed_at": run_entry.get("completed_at"),
        },
        "config": {
            "train_script": str(spec["script"]),
            "data_path": str(_spec_data_path(spec)),
            "output_root": str(models_dir.parent),
            "models_dir": str(models_dir),
            "report_path": str(report_path),
            "best_artifact_path": str(Path(spec["best"]).resolve()),
            "python_bin": args.python_bin,
            "batch_size": batch_size,
            "lr": lr,
            "max_epoch": args.max_epoch,
            "array_size": ARRAY_SIZE,
        },
        "slurm": {
            "array_job_id": run_entry.get("array_job_id"),
            "submit_stdout_log": run_entry.get("submit_stdout_log"),
            "submit_stderr_log": run_entry.get("submit_stderr_log"),
            "array_stdout_pattern": run_entry.get("array_stdout_pattern"),
            "array_stderr_pattern": run_entry.get("array_stderr_pattern"),
            "sacct": run_entry.get("sacct", []),
        },
        "logs": {
            "submit": {
                "stdout": run_entry.get("submit_stdout_log"),
                "stderr": run_entry.get("submit_stderr_log"),
            },
            "array": {
                "stdout_pattern": run_entry.get("array_stdout_pattern"),
                "stderr_pattern": run_entry.get("array_stderr_pattern"),
            },
            "finalize": finalize_logs,
        },
        "artifacts": {
            "best_artifact": _file_entry(best_artifact_path),
            "report": _file_entry(report_path),
            "training_results_csv": _file_entry(Path(training_results_path)) if training_results_path else None,
            "summary_plot": _file_entry(Path(summary_plot_path)) if summary_plot_path else None,
        },
    }


def _write_spec_manifest(
    *,
    spec: dict[str, Path | str],
    args: argparse.Namespace,
    output_root: Path,
    run_entry: dict[str, Any],
) -> Path:
    models_dir, report_path, manifest_path = _spec_paths(output_root, str(spec["name"]))
    manifest = _build_spec_manifest(
        spec=spec,
        args=args,
        models_dir=models_dir,
        report_path=report_path,
        run_entry=run_entry,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _report_passed(item_report: Path, best_path: Path) -> bool:
    if not item_report.exists() or not best_path.exists():
        return False
    try:
        payload = _read_json(item_report)
    except Exception:
        return False
    return Path(str(payload.get("best_dest", ""))).resolve() == best_path.resolve()


def _has_complete_result_set(models_dir: Path) -> bool:
    return len(list(models_dir.glob("result_*.json"))) == ARRAY_SIZE


def _submit_array_job(*, spec: dict[str, Path | str], output_root: Path, args: argparse.Namespace) -> tuple[str, dict[str, str], Path, Path]:
    models_dir, report_path, _manifest_path = _spec_paths(output_root, str(spec["name"]))
    stdout_log = output_root / str(spec["name"]) / "submit.stdout.log"
    stderr_log = output_root / str(spec["name"]) / "submit.stderr.log"
    array_stdout = output_root / str(spec["name"]) / "array_%A_%a.out"
    array_stderr = output_root / str(spec["name"]) / "array_%A_%a.err"
    batch_size, lr = _spec_params(spec, args)
    env = {
        **dict(os.environ),
        "REPO_ROOT": str(REPO_ROOT),
        "TRAIN_SCRIPT": str(spec["script"]),
        "EXPERIMENT": str(spec["experiment"]),
        "DIAMETER": str(spec["diameter"]),
        "OUTPUT_DIR": str(models_dir),
        "PYTHON_BIN": args.python_bin,
        "BATCH_SIZE": str(batch_size),
        "LR": str(lr),
        "MAX_EPOCH": str(args.max_epoch),
    }
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--array=0-{ARRAY_SIZE - 1}",
            "--output",
            str(array_stdout),
            "--error",
            str(array_stderr),
            str(SBATCH_TEMPLATE),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch array submission failed for {spec['name']}: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    raw_job_id = (result.stdout or "").strip()
    job_id = raw_job_id.split(";", 1)[0]
    return job_id, {
        "submit_stdout_log": str(stdout_log),
        "submit_stderr_log": str(stderr_log),
        "array_stdout_pattern": str(array_stdout),
        "array_stderr_pattern": str(array_stderr),
    }, models_dir, report_path


def _wait_for_jobs(job_ids: list[str], *, timeout_seconds: int, poll_seconds: int = 15) -> None:
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
        active = {line.split()[0].split("_")[0] for line in lines}
        remaining = {job_id for job_id in remaining if job_id in active}
        if remaining:
            time.sleep(poll_seconds)
    if remaining:
        raise TimeoutError(f"Timed out waiting for DNN array jobs: {', '.join(sorted(remaining))}")


def _job_children(job_id: str) -> list[dict[str, str]]:
    proc = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "-n", "-P"],
        text=True,
        capture_output=True,
        check=False,
    )
    rows: list[dict[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        current_job_id, state, exit_code = line.split("|", 2)
        if "." in current_job_id:
            continue
        rows.append({"job_id": current_job_id, "state": state, "exit_code": exit_code})
    return rows


def _array_passed(job_id: str) -> tuple[bool, list[dict[str, str]]]:
    rows = _job_children(job_id)
    child_rows = [row for row in rows if row["job_id"].startswith(f"{job_id}_")]
    if len(child_rows) != ARRAY_SIZE:
        return False, rows
    return all(row["state"] == "COMPLETED" and row["exit_code"] == "0:0" for row in child_rows), rows


def _finalize_spec(*, spec: dict[str, Path | str], output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    models_dir, report_path, _manifest_path = _spec_paths(output_root, str(spec["name"]))
    stdout_log = output_root / str(spec["name"]) / "finalize.stdout.log"
    stderr_log = output_root / str(spec["name"]) / "finalize.stderr.log"
    command = [
        args.python_bin,
        str(spec["script"]),
        "--diameter",
        str(spec["diameter"]),
        "--output-dir",
        str(models_dir),
        "--report-json",
        str(report_path),
        "--collect-only",
    ]
    result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "report": str(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 12-architecture DNN surrogate sweep for all EMB diameters in parallel arrays.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--compression-batch-size", type=int, default=128)
    parser.add_argument("--compression-lr", type=float, default=5e-4)
    parser.add_argument("--indentation-batch-size", type=int, default=1024)
    parser.add_argument("--indentation-lr", type=float, default=1e-3)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=8 * 3600)
    args = parser.parse_args(argv)

    resolved_site = args.site if args.site is not None else detect_hpc_site()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else default_runs_root(REPO_ROOT, "dnn_training", site=resolved_site, run_tag=args.run_tag)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.state_path).resolve() if args.state_path is not None else output_root / "dnn_training_matrix_report.json"

    report = _load_report(report_path, output_root, args)
    runs_by_name = _run_map(report)
    selected_specs = _select_specs(SPECS, start_from=args.start_from, only=list(args.only))
    if not selected_specs:
        raise ValueError("No specs selected to run.")

    report["status"] = "running"
    report["selected"] = [str(spec["name"]) for spec in selected_specs]
    _write_report(report_path, report)

    submissions: list[tuple[dict[str, Path | str], str]] = []
    for spec in selected_specs:
        name = str(spec["name"])
        models_dir, item_report, _manifest_path = _spec_paths(output_root, name)
        if args.resume and _report_passed(item_report, Path(spec["best"])):
            runs_by_name[name] = {
                "name": name,
                "status": "skipped",
                "reason": "existing_artifacts",
                "best": str(spec["best"]),
                "report": str(item_report),
                "started_at": _now_iso(),
                "completed_at": _now_iso(),
            }
            runs_by_name[name]["manifest"] = str(
                _write_spec_manifest(spec=spec, args=args, output_root=output_root, run_entry=runs_by_name[name])
            )
            continue
        if args.resume and _has_complete_result_set(models_dir):
            runs_by_name[name] = {
                "name": name,
                "status": "ready_to_finalize",
                "experiment": str(spec["experiment"]),
                "diameter": str(spec["diameter"]),
                "best": str(spec["best"]),
                "report": str(item_report),
                "models_dir": str(models_dir),
                "started_at": _now_iso(),
                "resumed_from_existing_results": True,
            }
            runs_by_name[name]["manifest"] = str(
                _write_spec_manifest(spec=spec, args=args, output_root=output_root, run_entry=runs_by_name[name])
            )
            continue

        job_id, submit_logs, _models_dir, report_file = _submit_array_job(spec=spec, output_root=output_root, args=args)
        run_entry = {
            "name": name,
            "status": "submitted",
            "experiment": str(spec["experiment"]),
            "diameter": str(spec["diameter"]),
            "array_job_id": job_id,
            "best": str(spec["best"]),
            "report": str(report_file),
            "models_dir": str(models_dir),
            "started_at": _now_iso(),
            **submit_logs,
        }
        runs_by_name[name] = run_entry
        run_entry["manifest"] = str(
            _write_spec_manifest(spec=spec, args=args, output_root=output_root, run_entry=run_entry)
        )
        submissions.append((spec, job_id))

    report["runs"] = list(runs_by_name.values())
    _write_report(report_path, report)

    if submissions:
        _wait_for_jobs([job_id for _, job_id in submissions], timeout_seconds=args.timeout_seconds)

    finalize_queue: list[dict[str, Path | str]] = []
    for spec, job_id in submissions:
        name = str(spec["name"])
        run_entry = runs_by_name[name]
        passed, sacct_rows = _array_passed(job_id)
        run_entry["sacct"] = sacct_rows
        if not passed:
            run_entry["status"] = "failed"
            run_entry["completed_at"] = _now_iso()
            run_entry["manifest"] = str(
                _write_spec_manifest(spec=spec, args=args, output_root=output_root, run_entry=run_entry)
            )
            report["status"] = "failed"
            report["runs"] = list(runs_by_name.values())
            _write_report(report_path, report)
            return 1
        finalize_queue.append(spec)

    existing_finalize_names = {str(item["name"]) for item in finalize_queue}
    for spec in selected_specs:
        name = str(spec["name"])
        if name in existing_finalize_names:
            continue
        models_dir, item_report, _manifest_path = _spec_paths(output_root, name)
        if not args.resume or _report_passed(item_report, Path(spec["best"])) or not _has_complete_result_set(models_dir):
            continue
        run_entry = runs_by_name.get(name, {})
        run_entry.update(
            {
                "name": name,
                "status": "ready_to_finalize",
                "experiment": str(spec["experiment"]),
                "diameter": str(spec["diameter"]),
                "best": str(spec["best"]),
                "report": str(item_report),
                "models_dir": str(models_dir),
                "resumed_from_existing_results": True,
            }
        )
        runs_by_name[name] = run_entry
        finalize_queue.append(spec)

    report["runs"] = list(runs_by_name.values())
    _write_report(report_path, report)

    for spec in finalize_queue:
        name = str(spec["name"])
        run_entry = runs_by_name[name]
        finalize = _finalize_spec(spec=spec, output_root=output_root, args=args)
        run_entry["finalize"] = finalize
        run_entry["completed_at"] = _now_iso()
        if finalize["returncode"] != 0 or not _report_passed(Path(finalize["report"]), Path(spec["best"])):
            run_entry["status"] = "failed"
            run_entry["manifest"] = str(
                _write_spec_manifest(spec=spec, args=args, output_root=output_root, run_entry=run_entry)
            )
            report["status"] = "failed"
            report["runs"] = list(runs_by_name.values())
            _write_report(report_path, report)
            return 1
        run_entry["status"] = "passed"
        run_entry["manifest"] = str(
            _write_spec_manifest(spec=spec, args=args, output_root=output_root, run_entry=run_entry)
        )

    report["status"] = "passed"
    report["completed_at"] = _now_iso()
    report["runs"] = list(runs_by_name.values())
    _write_report(report_path, report)
    print(f"DNN training matrix report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
