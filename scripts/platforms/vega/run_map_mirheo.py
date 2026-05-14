#!/usr/bin/env python3
"""Orchestrate MAP Mirheo DPD evaluation for all diameters of one experiment.

Reads the phase3b MAP manifest, converts each dataset entry to a per-diameter
JSON file, then runs evaluate_map_mirheo_optimized_{experiment}.py via MPI
for each diameter.  A summary manifest is written regardless of per-diameter
success/failure so that partial results are always recorded.

Usage:
    python3.8 scripts/platforms/vega/run_map_mirheo.py \\
        --experiment indentation \\
        --model-family reduced-model \\
        --profile production \\
        --output-dir /path/to/run/root \\
        --python-bin python3.8 \\
        --n-displacements 15

    # The phase3b manifest is expected at:
    #   <output-dir>/map_phase3b/phase3b_map_manifest.json
    # Outputs are written under:
    #   <output-dir>/map_mirheo/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from convert_map_manifest import convert_manifest  # noqa: E402

TIMEOUT_SECONDS = 1800  # default: 30 minutes per diameter
MPI_RANKS = 2
MAX_RETRIES = 1
RETRY_DT_SCALE_FACTOR = 0.5
MPI_ENV_EXPORTS = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HDF5_DIR",
    "EBROOTHDF5",
    "MESOUQ_MIRHEO_SRC",
    "MIRHEO_SOURCE_ROOT",
    "MIRHEO_BUILD_DIR",
    "MIRHEO_INSTALL_PREFIX",
)


def _evaluate_script(experiment: str) -> Path:
    name = f"evaluate_map_mirheo_optimized{'_' + experiment if experiment == 'indentation' else ''}.py"
    return REPO_ROOT / "propagation" / "scripts" / name


def _mpirun_export_args() -> list[str]:
    args: list[str] = []
    for name in MPI_ENV_EXPORTS:
        if os.environ.get(name):
            args.extend(["-x", name])
    return args


def _run_diameter(
    dataset_name: str,
    map_json: Path,
    result_json: Path,
    scratch_root: Path,
    experiment: str,
    python_bin: str,
    n_displacements: int,
    mpi_ranks: int,
    extra_args: list[str],
    timeout_seconds: int,
) -> dict:
    """Run Mirheo evaluation for one diameter.  Returns a result dict."""
    eval_script = _evaluate_script(experiment)
    command = [
        "mpirun", "--oversubscribe", * _mpirun_export_args(), "-n", str(mpi_ranks),
        python_bin, str(eval_script),
        "--map-file", str(map_json),
        "--output", str(result_json),
        "--scratch-root", str(scratch_root),
        "--n-displacements", str(n_displacements),
        *extra_args,
    ]
    start = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        raw_out = exc.stdout or b""
        raw_err = exc.stderr or b""
        stdout = raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, bytes) else raw_out
        stderr = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, bytes) else raw_err
    elapsed = time.perf_counter() - start

    return {
        "dataset_name": dataset_name,
        "map_json": str(map_json),
        "result_json": str(result_json) if result_json.exists() else None,
        "scratch_root": str(scratch_root),
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 2),
        "stdout_tail": stdout[-2000:] if stdout else "",
        "stderr_tail": stderr[-2000:] if stderr else "",
    }


def _cleanup_retry_state(scratch_root: Path, result_json: Path) -> list[str]:
    actions: list[str] = []
    if scratch_root.exists():
        shutil.rmtree(scratch_root, ignore_errors=True)
        actions.append(f"removed scratch root: {scratch_root}")
    if result_json.exists():
        result_json.unlink()
        actions.append(f"removed partial result: {result_json}")
    return actions


def _init_directory_policy(map_mirheo_dir: Path) -> dict[str, object]:
    scratch_root_base = map_mirheo_dir / "_scratch"
    return {
        "mode": "auto_prepared_per_dataset_scratch_root",
        "preexisting_init_dirs_required": False,
        "scratch_root_base": str(scratch_root_base),
        "scratch_root_pattern": str(scratch_root_base / "<dataset_name>"),
        "compression_template": "emb/compression/src regenerated through generate_sim/write_parameters",
        "indentation_template": "emb/indentation/src copied into the scratch root",
        "missing_template_behavior": (
            "evaluator fails explicitly; MAP Mirheo smoke does not skip missing init inputs"
        ),
    }


def _build_attempt_args(
    *,
    base_args: list[str],
    attempt_index: int,
    retry_dt_scale_factor: float,
) -> list[str]:
    attempt_args: list[str] = []
    base_retry_attempt = 0
    dt_scale = retry_dt_scale_factor
    idx = 0
    while idx < len(base_args):
        token = base_args[idx]
        if token == "--retry-attempt" and idx + 1 < len(base_args):
            base_retry_attempt = int(base_args[idx + 1])
            idx += 2
            continue
        if token == "--dt-scale-factor" and idx + 1 < len(base_args):
            dt_scale = float(base_args[idx + 1])
            idx += 2
            continue
        attempt_args.append(token)
        idx += 1

    resolved_retry_attempt = base_retry_attempt + attempt_index
    if resolved_retry_attempt > 0:
        attempt_args.extend(
            [
                "--retry-attempt",
                str(resolved_retry_attempt),
                "--dt-scale-factor",
                str(dt_scale),
            ]
        )
    return attempt_args


def run_map_mirheo(
    experiment: str,
    output_dir: Path,
    python_bin: str,
    n_displacements: int,
    mpi_ranks: int = MPI_RANKS,
    extra_args: list[str] | None = None,
    model_family: str = "unknown",
    timeout_seconds: int = TIMEOUT_SECONDS,
    dataset_names: list[str] | None = None,
    max_retries: int = MAX_RETRIES,
    retry_dt_scale_factor: float = RETRY_DT_SCALE_FACTOR,
) -> dict:
    """Run MAP Mirheo for all diameters.  Returns the summary manifest dict."""
    if extra_args is None:
        extra_args = []

    manifest_path = output_dir / "map_phase3b" / "phase3b_map_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Phase3b MAP manifest not found: {manifest_path}\n"
            "Run 'scripts/platforms/vega/extract_map.py --stage phase3b' first."
        )

    map_mirheo_dir = output_dir / "map_mirheo"
    map_json_dir = map_mirheo_dir / "map_json"
    results_dir = map_mirheo_dir / "results"
    map_json_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[map_mirheo] Manifest: {manifest_path}")
    print(f"[map_mirheo] Output:   {map_mirheo_dir}")

    # Convert manifest → per-diameter MAP JSON files
    written = convert_manifest(manifest_path, map_json_dir)
    print(f"[map_mirheo] Wrote {len(written)} per-diameter MAP JSON file(s)")

    # Read dataset names from manifest
    with open(manifest_path) as f:
        raw_manifest = json.load(f)
    datasets = raw_manifest.get("datasets", {})

    requested = set(dataset_names or [])
    available = {
        (map_json.stem[:-4] if map_json.stem.endswith("_map") else map_json.stem): map_json
        for map_json in written
    }
    if requested:
        missing = sorted(requested - set(available))
        if missing:
            raise ValueError(
                "Requested dataset(s) not found in phase3b map manifest: "
                + ", ".join(missing)
            )
        selected = [available[name] for name in sorted(requested)]
    else:
        selected = sorted(written)

    if max_retries < 0:
        raise ValueError("max_retries must be >= 0.")

    diameter_results: list[dict] = []
    for map_json in selected:
        # Derive dataset name from filename: e.g. indentation_3.2um_map.json → indentation_3.2um
        stem = map_json.stem
        dataset_name = stem[:-4] if stem.endswith("_map") else stem
        result_json = results_dir / f"{dataset_name}_result.json"
        scratch_root = map_mirheo_dir / "_scratch" / dataset_name

        print(f"[map_mirheo] Running {dataset_name} ...")
        attempts: list[dict] = []
        cleanup_actions: list[str] = []
        final_result: dict | None = None

        for attempt_index in range(max_retries + 1):
            attempt_args = _build_attempt_args(
                base_args=extra_args,
                attempt_index=attempt_index,
                retry_dt_scale_factor=retry_dt_scale_factor,
            )
            if attempt_index > 0:
                cleanup_actions.extend(_cleanup_retry_state(scratch_root, result_json))

            raw_result = _run_diameter(
                dataset_name=dataset_name,
                map_json=map_json,
                result_json=result_json,
                scratch_root=scratch_root,
                experiment=experiment,
                python_bin=python_bin,
                n_displacements=n_displacements,
                mpi_ranks=mpi_ranks,
                extra_args=attempt_args,
                timeout_seconds=timeout_seconds,
            )
            result = dict(raw_result)
            result["attempt"] = attempt_index + 1
            attempts.append(result)

            if result["timed_out"] and attempt_index < max_retries:
                print(
                    f"[map_mirheo]   {dataset_name}: timed out on attempt {attempt_index + 1}; retrying ..."
                )
                continue

            final_result = dict(result)
            break

        assert final_result is not None  # pragma: no cover
        status = "timed_out" if final_result["timed_out"] else (
            "passed" if final_result["returncode"] == 0 else "failed"
        )
        final_result["status"] = status
        final_result["attempts"] = attempts
        final_result["attempt_count"] = len(attempts)
        final_result["cleanup_actions"] = cleanup_actions
        diameter_results.append(final_result)
        print(f"[map_mirheo]   {dataset_name}: {status} ({final_result['elapsed_seconds']:.1f}s)")

    overall_status = (
        "passed" if diameter_results and all(r["status"] == "passed" for r in diameter_results)
        else "failed"
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "model_family": model_family,
        "manifest_path": str(manifest_path),
        "map_mirheo_dir": str(map_mirheo_dir),
        "init_directory_policy": _init_directory_policy(map_mirheo_dir),
        "n_displacements": n_displacements,
        "mpi_ranks": mpi_ranks,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "retry_dt_scale_factor": retry_dt_scale_factor,
        "selected_datasets": sorted(requested),
        "status": overall_status,
        "diameters": diameter_results,
    }

    summary_path = map_mirheo_dir / "map_mirheo_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[map_mirheo] Summary: {summary_path}")
    print(f"[map_mirheo] Status:  {overall_status}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate MAP Mirheo DPD evaluation for all diameters of one experiment"
    )
    parser.add_argument(
        "--experiment", required=True, choices=["indentation", "compression"],
        help="Which experiment to evaluate"
    )
    parser.add_argument(
        "--model-family", required=True,
        help="Model family (e.g. reduced-model, full-model)"
    )
    parser.add_argument(
        "--profile", required=True,
        help="Profile (e.g. production, validation)"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Root of the workflow run directory (contains map_phase3b/)"
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--n-displacements", type=int, default=15)
    parser.add_argument(
        "--mpi-ranks", type=int, default=MPI_RANKS,
        help=f"Number of MPI ranks per simulation (default: {MPI_RANKS})"
    )
    parser.add_argument(
        "--retry-attempt", type=int, default=0,
        help="Passed through to the evaluate script (0 = no retry scaling)"
    )
    parser.add_argument(
        "--dt-scale-factor", type=float, default=0.5,
        help="Passed through to the evaluate script"
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=TIMEOUT_SECONDS,
        help=f"Per-diameter timeout for evaluate subprocesses (default: {TIMEOUT_SECONDS})"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Retry count for timed-out diameters (default: {MAX_RETRIES}).",
    )
    parser.add_argument(
        "--retry-dt-scale-factor",
        type=float,
        default=RETRY_DT_SCALE_FACTOR,
        help=(
            "dt scaling passed to evaluate scripts during automatic retries "
            f"(default: {RETRY_DT_SCALE_FACTOR})."
        ),
    )
    parser.add_argument(
        "--numsteps", type=int, default=None,
        help="Optional override passed through to evaluate_map_mirheo_optimized*.py"
    )
    parser.add_argument(
        "--numsteps-eq", type=int, default=None,
        help="Optional override passed through to evaluate_map_mirheo_optimized*.py"
    )
    parser.add_argument(
        "--dataset-name", action="append", default=[],
        help="Optional dataset name filter. Repeat to run only selected diameters."
    )
    args = parser.parse_args(argv)

    if args.mpi_ranks != 2:
        print("ERROR: --mpi-ranks must be exactly 2 (required by the Mirheo evaluate scripts)", file=sys.stderr)
        return 1
    if args.n_displacements < 1:
        print("ERROR: --n-displacements must be >= 1", file=sys.stderr)
        return 1
    if args.retry_attempt < 0:
        print("ERROR: --retry-attempt must be >= 0", file=sys.stderr)
        return 1
    if args.timeout_seconds < 1:
        print("ERROR: --timeout-seconds must be >= 1", file=sys.stderr)
        return 1
    if args.max_retries < 0:
        print("ERROR: --max-retries must be >= 0", file=sys.stderr)
        return 1
    if args.retry_dt_scale_factor <= 0:
        print("ERROR: --retry-dt-scale-factor must be > 0", file=sys.stderr)
        return 1

    extra_args: list[str] = []
    if args.retry_attempt > 0:
        extra_args += ["--retry-attempt", str(args.retry_attempt),
                       "--dt-scale-factor", str(args.dt_scale_factor)]
    if args.numsteps is not None:
        extra_args += ["--numsteps", str(args.numsteps)]
    if args.numsteps_eq is not None:
        extra_args += ["--numsteps-eq", str(args.numsteps_eq)]

    try:
        summary = run_map_mirheo(
            experiment=args.experiment,
            output_dir=args.output_dir,
            python_bin=args.python_bin,
            n_displacements=args.n_displacements,
            mpi_ranks=args.mpi_ranks,
            extra_args=extra_args,
            model_family=args.model_family,
            timeout_seconds=args.timeout_seconds,
            dataset_names=args.dataset_name,
            max_retries=args.max_retries,
            retry_dt_scale_factor=args.retry_dt_scale_factor,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
