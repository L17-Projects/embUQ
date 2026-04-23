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
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.hpc_paths import default_runs_root, detect_hpc_site  # noqa: E402
from meso_uq.campaign_manifests import (  # noqa: E402
    MANDATORY_MAIN_FIGURES,
    MANDATORY_SUPPLEMENTARY_FIGURES,
    MANDATORY_TABLES,
    build_paper_release_manifest,
    build_required_asset_entries,
    derive_asset_source_map,
    build_job_manifest,
    build_lane_manifest,
    build_partition_policy_metadata,
    build_phase2_backend_policy,
    infer_gpu_job,
    infer_phase2_backend,
    load_asset_source_map,
    load_git_metadata,
    utc_now_iso,
    write_manifest,
)
from meso_uq.vega_workflows import (  # noqa: E402
    VALID_EXPERIMENTS,
    VALID_MODEL_FAMILIES,
    VALID_PHASE2_BACKENDS,
    VALID_PROFILES,
    VegaWorkflowSelection,
    expand_selection_matrix,
    format_command,
    parse_selection,
    resolve_workflow_config_path,
    selection_key,
    selection_slug,
)


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _selection_output_root(matrix_root: Path, selection: VegaWorkflowSelection) -> Path:
    return matrix_root / "runs" / selection.experiment / selection.model_family / selection.profile


def _selection_logs_root(matrix_root: Path, selection: VegaWorkflowSelection) -> Path:
    return matrix_root / "logs" / selection.experiment / selection.model_family / selection.profile


def _selection_summary_path(matrix_root: Path, selection: VegaWorkflowSelection) -> Path:
    return matrix_root / "summaries" / f"{selection_slug(selection)}.json"


def _selection_lane_manifest_path(matrix_root: Path, selection: VegaWorkflowSelection) -> Path:
    return matrix_root / "manifests" / "lanes" / f"{selection_slug(selection)}.json"


def _selection_job_manifest_path(
    matrix_root: Path,
    selection: VegaWorkflowSelection,
    stage: str,
    ordinal: int,
) -> Path:
    return (
        matrix_root
        / "manifests"
        / "jobs"
        / f"{selection_slug(selection)}__{stage}__{ordinal:02d}.json"
    )


def _step_output_candidates(step_name: str, selection_output_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if step_name == "map_phase1":
        candidates.append(selection_output_root / "map_phase1" / "phase1_map_manifest.json")
    elif step_name == "map_phase3b":
        candidates.append(selection_output_root / "map_phase3b" / "phase3b_map_manifest.json")
    elif step_name == "map_mirheo":
        candidates.append(selection_output_root / "map_mirheo" / "map_mirheo_manifest.json")
    return candidates


def _stage_artifact_roots(step_name: str, selection_output_root: Path) -> list[Path]:
    roots_by_stage: dict[str, list[Path]] = {
        "phase1": [selection_output_root / "results_phase_1"],
        "phase2": [selection_output_root / "results_phase_2"],
        "phase3b": [selection_output_root / "results_phase_3b"],
        "propagation_phase3b": [selection_output_root / "propagation_phase3b"],
        "map_phase1": [selection_output_root / "map_phase1"],
        "map_phase3b": [selection_output_root / "map_phase3b"],
        "map_mirheo": [selection_output_root / "map_mirheo"],
    }
    return roots_by_stage.get(step_name, [])


def _discover_stage_output_files(step_name: str, selection_output_root: Path) -> list[Path]:
    discovered: list[Path] = []
    for root in _stage_artifact_roots(step_name, selection_output_root):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                discovered.append(path)
    return discovered


def _snapshot_output_tree(output_root: Path) -> dict[str, tuple[int, int]]:
    if not output_root.exists():
        return {}
    snapshot: dict[str, tuple[int, int]] = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot[str(path.resolve())] = (int(stat.st_size), int(stat.st_mtime_ns))
    return snapshot


def _diff_output_tree(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> list[Path]:
    changed: list[Path] = []
    for path_text, after_meta in after.items():
        before_meta = before.get(path_text)
        if before_meta != after_meta:
            changed.append(Path(path_text))
    return changed


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for item in paths:
        key = str(item.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _capture_step(name: str, command: list[str], logs_root: Path) -> dict[str, object]:
    logs_root.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_root / f"{name}.stdout.log"
    stderr_path = logs_root / f"{name}.stderr.log"
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    result = subprocess.run(
        command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False
    )
    elapsed = time.perf_counter() - start
    ended_at = datetime.now(timezone.utc)
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "command_text": format_command(command),
        "cwd": str(REPO_ROOT),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "start_utc": started_at.isoformat(),
        "end_utc": ended_at.isoformat(),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _config_overrides(items: Iterable[str]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --config-override value '{item}'. Expected selection=path.")
        selector, path_text = item.split("=", 1)
        resolved[selection_key(parse_selection(selector))] = _resolve_path(path_text)
    return resolved


def _resolve_config_override(
    selection: VegaWorkflowSelection,
    overrides: dict[str, Path],
) -> Path | None:
    return overrides.get(selection_key(selection))


def _build_selection_commands(
    selection: VegaWorkflowSelection,
    selection_output_root: Path,
    python_bin: str,
    phase2_cpu_ranks: int,
    config_override: Path | None,
    inference_device: str,
    propagation_device: str,
    phase2_backend: str | None,
    *,
    skip_phase1_map: bool,
    skip_phase3b_map: bool,
    skip_phase3b_propagation: bool,
    run_map_mirheo: bool,
    map_mirheo_n_displacements: int,
) -> list[tuple[str, list[str]]]:
    resolved_phase2_backend = (
        phase2_backend if phase2_backend is not None else
        ("native-cuda" if selection.profile == "production" else "cpu-mpi")
    )
    inferred_config = resolve_workflow_config_path(REPO_ROOT, selection, config_override)
    config_args = ["--config", str(inferred_config)] if config_override is not None else []

    inference_base = [
        python_bin,
        str(REPO_ROOT / "scripts" / "vega" / "run_inference_stage.py"),
        "--experiment",
        selection.experiment,
        "--model-family",
        selection.model_family,
        "--profile",
        selection.profile,
        "--output-dir",
        str(selection_output_root),
        "--python-bin",
        python_bin,
        *config_args,
    ]
    propagation_base = [
        python_bin,
        str(REPO_ROOT / "scripts" / "vega" / "run_propagation.py"),
        "--experiment",
        selection.experiment,
        "--model-family",
        selection.model_family,
        "--profile",
        selection.profile,
        "--output-dir",
        str(selection_output_root),
        "--python-bin",
        python_bin,
        *config_args,
    ]
    map_base = [
        python_bin,
        str(REPO_ROOT / "scripts" / "vega" / "extract_map.py"),
        "--experiment",
        selection.experiment,
        "--model-family",
        selection.model_family,
        "--profile",
        selection.profile,
        "--output-dir",
        str(selection_output_root),
        *config_args,
    ]

    commands: list[tuple[str, list[str]]] = [
        ("phase1", [*inference_base, "--stage", "phase1", "--device", inference_device]),
    ]
    if not skip_phase1_map:
        commands.append(("map_phase1", [*map_base, "--stage", "phase1"]))
    commands.append(
        (
            "phase2",
            [
                *inference_base,
                "--stage",
                "phase2",
                "--cpu-ranks",
                str(phase2_cpu_ranks),
                "--phase2-backend",
                resolved_phase2_backend,
            ],
        )
    )
    commands.append(
        ("phase3b", [*inference_base, "--stage", "phase3b", "--device", inference_device])
    )
    if not skip_phase3b_propagation:
        commands.append(
            (
                "propagation_phase3b",
                [*propagation_base, "--stage", "phase3b", "--device", propagation_device],
            )
        )
    if not skip_phase3b_map:
        commands.append(("map_phase3b", [*map_base, "--stage", "phase3b"]))
    if run_map_mirheo:
        commands.append((
            "map_mirheo",
            [
                python_bin,
                str(REPO_ROOT / "scripts" / "vega" / "run_map_mirheo.py"),
                "--experiment", selection.experiment,
                "--model-family", selection.model_family,
                "--profile", selection.profile,
                "--output-dir", str(selection_output_root),
                "--python-bin", python_bin,
                "--n-displacements", str(map_mirheo_n_displacements),
            ],
        ))
    return commands


def _selection_artifacts(selection_output_root: Path) -> dict[str, str]:
    return {
        "phase1_map_manifest": str(
            selection_output_root / "map_phase1" / "phase1_map_manifest.json"
        ),
        "phase3b_map_manifest": str(
            selection_output_root / "map_phase3b" / "phase3b_map_manifest.json"
        ),
        "map_mirheo_manifest": str(
            selection_output_root / "map_mirheo" / "map_mirheo_manifest.json"
        ),
        "phase3b_propagation_root": str(selection_output_root / "propagation_phase3b"),
    }


def _write_selection_summary(summary_path: Path, payload: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _deduplicate(selections: Iterable[VegaWorkflowSelection]) -> list[VegaWorkflowSelection]:
    ordered: list[VegaWorkflowSelection] = []
    seen: set[str] = set()
    for selection in selections:
        key = selection_key(selection)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(selection)
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fresh-clone Vega workflow matrix with explicit model-family/profile axes."
        )
    )
    parser.add_argument(
        "--selection",
        action="append",
        default=[],
        help="Explicit selection in experiment:model-family:profile form.",
    )
    parser.add_argument("--experiments", nargs="+", choices=VALID_EXPERIMENTS, default=None)
    parser.add_argument("--model-families", nargs="+", choices=VALID_MODEL_FAMILIES, default=None)
    parser.add_argument("--profiles", nargs="+", choices=VALID_PROFILES, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--phase2-cpu-ranks", type=int, default=1)
    parser.add_argument(
        "--phase2-backend",
        choices=VALID_PHASE2_BACKENDS,
        default=None,
        help="Optional explicit Phase 2 backend override. By default: production=native-cuda, validation=cpu-mpi.",
    )
    parser.add_argument("--inference-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--propagation-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    parser.add_argument("--run-phase1-map", action="store_true", default=False,
                        help="Run the phase1 MAP extraction step (skipped by default).")
    parser.add_argument("--skip-phase3b-map", action="store_true", default=False)
    parser.add_argument("--skip-phase3b-propagation", action="store_true", default=False)
    parser.add_argument(
        "--run-map-mirheo", action="store_true", default=False,
        help="Run MAP Mirheo DPD evaluation after map_phase3b (requires Mirheo on this host).",
    )
    parser.add_argument(
        "--map-mirheo-n-displacements", type=int, default=15,
        help="Number of displacement points for MAP Mirheo evaluation (default: 15).",
    )
    parser.add_argument(
        "--figures-main-root",
        type=str,
        default=None,
        help="Directory containing main paper figures for release manifest hashing.",
    )
    parser.add_argument(
        "--figures-supplementary-root",
        type=str,
        default=None,
        help="Directory containing supplementary paper figures for release manifest hashing.",
    )
    parser.add_argument(
        "--tables-root",
        type=str,
        default=None,
        help="Directory containing paper tables for release manifest hashing.",
    )
    parser.add_argument(
        "--asset-source-map",
        type=str,
        default=None,
        help="Optional JSON mapping from required asset relative path to source lane/artifacts.",
    )
    parser.add_argument(
        "--allow-release-fail",
        action="store_true",
        default=False,
        help="Do not force non-zero exit code when release_status=FAIL.",
    )
    parser.add_argument(
        "--skip-release-manifest",
        action="store_true",
        default=False,
        help=(
            "Treat this run as a workflow-only canary and skip paper-release asset gating. "
            "The report will set release_status=SKIPPED."
        ),
    )
    args = parser.parse_args(argv)

    if args.phase2_cpu_ranks < 1:
        raise ValueError("--phase2-cpu-ranks must be a positive integer.")

    resolved_site = args.site if args.site is not None else detect_hpc_site()
    matrix_root = (
        _resolve_path(args.output_root)
        if args.output_root is not None
        else default_runs_root(REPO_ROOT, "workflow_matrix", site=resolved_site, run_tag=args.run_tag)
    )
    matrix_root.mkdir(parents=True, exist_ok=True)
    overrides = _config_overrides(args.config_override)

    explicit = [parse_selection(value) for value in args.selection]
    experiments = (
        args.experiments
        if args.experiments is not None
        else (list(VALID_EXPERIMENTS) if not explicit else [])
    )
    model_families = (
        args.model_families
        if args.model_families is not None
        else (list(VALID_MODEL_FAMILIES) if not explicit else [])
    )
    profiles = (
        args.profiles if args.profiles is not None else (["validation"] if not explicit else [])
    )
    expanded = expand_selection_matrix(experiments, model_families, profiles)
    selections = _deduplicate([*explicit, *expanded])
    if not selections:
        raise ValueError(
            "No workflow selections were resolved. Provide --selection or the matrix axes."
        )

    report: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "python_bin": args.python_bin,
        "phase2_cpu_ranks": args.phase2_cpu_ranks,
        "phase2_backend": args.phase2_backend,
        "matrix_root": str(matrix_root),
        "site": resolved_site,
        "run_tag": matrix_root.name,
        "status": "running",
        "release_scope": "workflow-only" if args.skip_release_manifest else "paper-release",
        "manifests": {
            "jobs_root": str(matrix_root / "manifests" / "jobs"),
            "lanes_root": str(matrix_root / "manifests" / "lanes"),
        },
        "selections": [],
    }
    git_metadata = load_git_metadata(REPO_ROOT)
    lane_manifest_paths: list[Path] = []

    exit_code = 0
    for selection in selections:
        selection_output_root = _selection_output_root(matrix_root, selection)
        selection_logs_root = _selection_logs_root(matrix_root, selection)
        config_override = _resolve_config_override(selection, overrides)
        config_path = resolve_workflow_config_path(REPO_ROOT, selection, config_override)
        steps: list[dict[str, object]] = []
        job_manifest_paths: list[str] = []
        job_manifests: list[dict[str, object]] = []
        selection_commands = _build_selection_commands(
            selection,
            selection_output_root=selection_output_root,
            python_bin=args.python_bin,
            phase2_cpu_ranks=args.phase2_cpu_ranks,
            config_override=config_override,
            inference_device=args.inference_device,
            propagation_device=args.propagation_device,
            phase2_backend=args.phase2_backend,
            skip_phase1_map=not args.run_phase1_map,
            skip_phase3b_map=args.skip_phase3b_map,
            skip_phase3b_propagation=args.skip_phase3b_propagation,
            run_map_mirheo=args.run_map_mirheo,
            map_mirheo_n_displacements=args.map_mirheo_n_displacements,
        )
        for step_index, (step_name, command) in enumerate(selection_commands, start=1):
            before_snapshot = _snapshot_output_tree(selection_output_root)
            step = _capture_step(step_name, command, selection_logs_root)
            steps.append(step)
            after_snapshot = _snapshot_output_tree(selection_output_root)

            is_gpu_job = infer_gpu_job(step_name=step_name, command=command)
            phase2_backend = infer_phase2_backend(command)
            partition_policy = build_partition_policy_metadata(
                is_gpu_job=is_gpu_job,
                partition=os.environ.get("SLURM_JOB_PARTITION"),
                time_limit=os.environ.get("SLURM_TIMELIMIT"),
            )
            backend_metadata = {
                "phase2_backend": phase2_backend,
                "phase2_backend_policy": build_phase2_backend_policy(
                    stage=step_name,
                    profile=selection.profile,
                    phase2_backend=phase2_backend,
                ),
                "partition_policy": partition_policy,
            }
            job_manifest_path = _selection_job_manifest_path(
                matrix_root, selection, step_name, step_index
            )
            job_manifest = build_job_manifest(
                job_id=job_manifest_path.stem,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                stage=step_name,
                lane=selection_key(selection),
                status="passed" if step["returncode"] == 0 else "failed",
                command=command,
                cwd=REPO_ROOT,
                start_utc=str(step["start_utc"]),
                end_utc=str(step["end_utc"]),
                duration_seconds=float(step["elapsed_seconds"]),
                partition=os.environ.get("SLURM_JOB_PARTITION"),
                time_limit=os.environ.get("SLURM_TIMELIMIT"),
                node_list=os.environ.get("SLURM_JOB_NODELIST"),
                gpu_type=os.environ.get("MESOUQ_GPU_TYPE"),
                gpu_count=_optional_int(os.environ.get("SLURM_GPUS_ON_NODE")),
                cpu_count=_optional_int(os.environ.get("SLURM_CPUS_ON_NODE")),
                mem_mb=_optional_int(os.environ.get("SLURM_MEM_PER_NODE")),
                config_path=config_path,
                logs={"stdout": str(step["stdout_log"]), "stderr": str(step["stderr_log"])},
                backend_metadata=backend_metadata,
                git_metadata=git_metadata,
                input_files=_unique_paths(
                    [
                        config_path,
                        Path(command[1]),
                    ]
                ),
                output_files=_unique_paths(
                    [
                        Path(str(step["stdout_log"])),
                        Path(str(step["stderr_log"])),
                        *_step_output_candidates(step_name, selection_output_root),
                        *_discover_stage_output_files(step_name, selection_output_root),
                        *_diff_output_tree(before_snapshot, after_snapshot),
                    ]
                ),
            )
            write_manifest(job_manifest_path, job_manifest)
            job_manifest_paths.append(str(job_manifest_path))
            job_manifests.append(job_manifest)

            if step["returncode"] != 0:
                exit_code = 1
                if not args.continue_on_error:
                    break

        artifacts = _selection_artifacts(selection_output_root)
        lane_manifest = build_lane_manifest(
            lane=selection_key(selection),
            selection={
                "experiment": selection.experiment,
                "model_family": selection.model_family,
                "profile": selection.profile,
            },
            steps=steps,
            artifacts=artifacts,
            job_manifest_paths=job_manifest_paths,
            job_manifests=job_manifests,
        )
        lane_manifest_path = _selection_lane_manifest_path(matrix_root, selection)
        write_manifest(lane_manifest_path, lane_manifest)
        lane_manifest_paths.append(lane_manifest_path)

        selection_summary = {
            "selection": selection_key(selection),
            "experiment": selection.experiment,
            "model_family": selection.model_family,
            "profile": selection.profile,
            "config": str(config_path),
            "output_root": str(selection_output_root),
            "artifacts": artifacts,
            "steps": steps,
            "job_manifests": job_manifest_paths,
            "lane_manifest": str(lane_manifest_path),
            "status": (
                "passed" if steps and all(step["returncode"] == 0 for step in steps) else "failed"
            ),
        }
        summary_path = _selection_summary_path(matrix_root, selection)
        _write_selection_summary(summary_path, selection_summary)
        selection_summary["summary_path"] = str(summary_path)
        report["selections"].append(selection_summary)

        if exit_code != 0 and not args.continue_on_error:
            break

    report["status"] = "passed" if exit_code == 0 else "failed"
    if args.skip_release_manifest:
        report["release_status"] = "SKIPPED"
        report["hard_failures"] = []
        report["manifests"]["paper_release_manifest"] = None
    else:
        figures_main_root = (
            _resolve_path(args.figures_main_root)
            if args.figures_main_root is not None
            else (matrix_root / "figures" / "main")
        )
        figures_supplementary_root = (
            _resolve_path(args.figures_supplementary_root)
            if args.figures_supplementary_root is not None
            else (matrix_root / "figures" / "supplementary")
        )
        tables_root = (
            _resolve_path(args.tables_root)
            if args.tables_root is not None
            else (matrix_root / "tables")
        )
        source_map_overrides = load_asset_source_map(
            _resolve_path(args.asset_source_map) if args.asset_source_map is not None else None
        )
        lane_manifests_payload = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in lane_manifest_paths
            if path.exists()
        ]
        required_relative_paths = [
            *[f"figures/main/{name}" for name in MANDATORY_MAIN_FIGURES],
            *[f"figures/supplementary/{name}" for name in MANDATORY_SUPPLEMENTARY_FIGURES],
            *[f"tables/{name}" for name in MANDATORY_TABLES],
        ]
        source_map_auto = derive_asset_source_map(
            required_relative_paths=required_relative_paths,
            lane_manifests=lane_manifests_payload,
        )
        source_map = dict(source_map_auto)
        source_map.update(source_map_overrides)
        figures_main_entries = build_required_asset_entries(
            category="figures/main",
            root=figures_main_root,
            required_files=MANDATORY_MAIN_FIGURES,
            source_map=source_map,
        )
        figures_supplementary_entries = build_required_asset_entries(
            category="figures/supplementary",
            root=figures_supplementary_root,
            required_files=MANDATORY_SUPPLEMENTARY_FIGURES,
            source_map=source_map,
        )
        table_entries = build_required_asset_entries(
            category="tables",
            root=tables_root,
            required_files=MANDATORY_TABLES,
            source_map=source_map,
        )
        paper_release_manifest = build_paper_release_manifest(
            run_campaign_id=matrix_root.name,
            generated_at_utc=utc_now_iso(),
            lane_manifest_paths=lane_manifest_paths,
            figures_main_entries=figures_main_entries,
            figures_supplementary_entries=figures_supplementary_entries,
            table_entries=table_entries,
        )
        paper_release_manifest_path = matrix_root / "manifests" / "paper_release_manifest.json"
        write_manifest(paper_release_manifest_path, paper_release_manifest)
        report["release_status"] = paper_release_manifest["release_status"]
        report["hard_failures"] = paper_release_manifest["hard_failures"]
        report["manifests"]["paper_release_manifest"] = str(paper_release_manifest_path)
        report["asset_roots"] = {
            "figures_main": str(figures_main_root),
            "figures_supplementary": str(figures_supplementary_root),
            "tables": str(tables_root),
        }
    report_path = matrix_root / "workflow_matrix_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Workflow matrix report: {report_path}")
    print(f"Workflow matrix status: {report['status']}")
    if report["release_status"] not in {"PASS", "SKIPPED"} and not args.allow_release_fail:
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
