#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.vega_workflows import (  # noqa: E402
    VALID_EXPERIMENTS,
    VALID_MODEL_FAMILIES,
    VALID_PROFILES,
    VegaWorkflowSelection,
    expand_selection_matrix,
    format_command,
    parse_selection,
    resolve_workflow_config_path,
    selection_key,
    selection_slug,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_vega" / "workflow_matrix"


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


def _capture_step(name: str, command: list[str], logs_root: Path) -> dict[str, object]:
    logs_root.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_root / f"{name}.stdout.log"
    stderr_path = logs_root / f"{name}.stderr.log"
    start = time.perf_counter()
    result = subprocess.run(
        command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False
    )
    elapsed = time.perf_counter() - start
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "command_text": format_command(command),
        "cwd": str(REPO_ROOT),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
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
    *,
    skip_phase1_map: bool,
    skip_phase3b_map: bool,
    skip_phase3b_propagation: bool,
    run_map_mirheo: bool,
    map_mirheo_n_displacements: int,
) -> list[tuple[str, list[str]]]:
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
        ("phase2", [*inference_base, "--stage", "phase2", "--cpu-ranks", str(phase2_cpu_ranks)])
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
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--phase2-cpu-ranks", type=int, default=1)
    parser.add_argument("--inference-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--propagation-device", choices=["cpu", "gpu"], default="cpu")
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
    args = parser.parse_args(argv)

    if args.phase2_cpu_ranks < 1:
        raise ValueError("--phase2-cpu-ranks must be a positive integer.")

    matrix_root = _resolve_path(args.output_root)
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
        "matrix_root": str(matrix_root),
        "status": "running",
        "selections": [],
    }

    exit_code = 0
    for selection in selections:
        selection_output_root = _selection_output_root(matrix_root, selection)
        selection_logs_root = _selection_logs_root(matrix_root, selection)
        config_override = _resolve_config_override(selection, overrides)
        config_path = resolve_workflow_config_path(REPO_ROOT, selection, config_override)
        steps: list[dict[str, object]] = []

        for step_name, command in _build_selection_commands(
            selection,
            selection_output_root=selection_output_root,
            python_bin=args.python_bin,
            phase2_cpu_ranks=args.phase2_cpu_ranks,
            config_override=config_override,
            inference_device=args.inference_device,
            propagation_device=args.propagation_device,
            skip_phase1_map=not args.run_phase1_map,
            skip_phase3b_map=args.skip_phase3b_map,
            skip_phase3b_propagation=args.skip_phase3b_propagation,
            run_map_mirheo=args.run_map_mirheo,
            map_mirheo_n_displacements=args.map_mirheo_n_displacements,
        ):
            step = _capture_step(step_name, command, selection_logs_root)
            steps.append(step)
            if step["returncode"] != 0:
                exit_code = 1
                if not args.continue_on_error:
                    break

        selection_summary = {
            "selection": selection_key(selection),
            "experiment": selection.experiment,
            "model_family": selection.model_family,
            "profile": selection.profile,
            "config": str(config_path),
            "output_root": str(selection_output_root),
            "artifacts": _selection_artifacts(selection_output_root),
            "steps": steps,
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
    report_path = matrix_root / "workflow_matrix_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Workflow matrix report: {report_path}")
    print(f"Workflow matrix status: {report['status']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
