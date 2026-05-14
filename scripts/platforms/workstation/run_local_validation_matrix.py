#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import site
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.production_sanity import render_production_sanity_plots  # noqa: E402
from meso_uq.vega_workflows import (  # noqa: E402
    VALID_EXPERIMENTS,
    VALID_MODEL_FAMILIES,
    VegaWorkflowSelection,
    expand_selection_matrix,
    format_command,
    parse_selection,
    resolve_workflow_config_path,
    selection_key,
    selection_slug,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_o369_runs"
DEFAULT_SELECTIONS = [
    "compression:full-model:validation",
    "compression:reduced-model:validation",
    "indentation:full-model:validation",
    "indentation:reduced-model:validation",
]
DEFAULT_SMOKE_OVERRIDES = {
    "pop_size": 64,
    "max_gen": 1,
    "hbi_pop_size": 64,
    "phase3b_pop_size": 64,
    "phase3b_max_gen": 1,
    "map_n_displacements": 1,
}


def _resolve_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    existing = env.get("PYTHONPATH", "")
    prefix = str(path)
    if not existing:
        env["PYTHONPATH"] = prefix
        return
    entries = [entry for entry in existing.split(":") if entry]
    if prefix in entries:
        return
    env["PYTHONPATH"] = f"{prefix}:{existing}"


def _discover_repo_local_korali_site() -> Path | None:
    candidates = sorted(
        (REPO_ROOT / "_vega" / "korali" / "install" / "lib").glob("python*/site-packages")
    )
    if not candidates:
        return None
    return candidates[-1]


def _probe_korali_engine(python_bin: str, env: dict[str, str]) -> tuple[bool, str]:
    result = subprocess.run(
        [python_bin, "-c", "import korali; korali.Engine()"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode == 0:
        return True, ""
    return False, (result.stderr or result.stdout or "").strip()


def _build_runtime_env(python_bin: str) -> tuple[dict[str, str], list[str]]:
    env = dict(os.environ)
    notes: list[str] = []

    korali_site = _discover_repo_local_korali_site()
    if korali_site is not None:
        _prepend_pythonpath(env, korali_site)
        notes.append(f"Using repo-local Korali runtime path: {korali_site}")
    else:
        notes.append(
            "Repo-local Korali runtime path not found under "
            "_vega/korali/install/lib/python*/site-packages"
        )

    ok, error_text = _probe_korali_engine(python_bin, env)
    if ok:
        return env, notes

    user_site = Path(site.getusersitepackages())
    if "Could not load mpi4py API." in error_text and user_site.exists():
        _prepend_pythonpath(env, user_site)
        ok_retry, retry_error = _probe_korali_engine(python_bin, env)
        if ok_retry:
            notes.append(
                "Prepended user-site packages to satisfy mpi4py API linkage for repo-local Korali."
            )
            return env, notes
        notes.append(f"Retry after user-site prepend still failed: {retry_error}")
    else:
        notes.append(f"Korali probe failed before workflow run: {error_text}")

    return env, notes


def _resolve_selections(values: list[str], all_lanes: bool) -> list[VegaWorkflowSelection]:
    if values:
        selections = [parse_selection(value) for value in values]
    elif all_lanes:
        selections = expand_selection_matrix(
            VALID_EXPERIMENTS, VALID_MODEL_FAMILIES, ("validation",)
        )
    else:
        selections = [parse_selection(value) for value in DEFAULT_SELECTIONS]

    for selection in selections:
        if selection.profile != "validation":
            raise ValueError(
                "Workstation local matrix runner supports validation profile only. "
                f"Got: {selection_key(selection)}"
            )
    return selections


def _derive_validation_smoke_config(
    selection: VegaWorkflowSelection,
    output_root: Path,
    overrides: dict[str, object],
) -> Path:
    base_config = resolve_workflow_config_path(REPO_ROOT, selection)
    with base_config.open("rb") as handle:
        payload = yaml.load(handle, Loader=yaml.CLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {base_config}, got {type(payload).__name__}")

    derived = dict(payload)
    for key, value in overrides.items():
        if key in derived:
            derived[key] = value
    derived["description"] = (
        f"o369 local validation smoke override from {base_config.name} "
        f"for {selection_key(selection)}"
    )

    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{selection_slug(selection)}.yaml"
    config_path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")
    return config_path


def _collect_overlay_paths(plots: dict[str, object]) -> dict[str, dict[str, list[str]]]:
    collected: dict[str, dict[str, list[str]]] = {}
    for selection, selection_payload in plots.items():
        datasets = selection_payload.get("datasets", {})
        propagation_plots: list[str] = []
        map_plots: list[str] = []
        for dataset_payload in datasets.values():
            propagation_payload = dataset_payload.get("propagation_phase3b", {})
            map_payload = dataset_payload.get("map_phase3b", {})
            propagation_plot = propagation_payload.get("plot")
            map_plot = map_payload.get("plot")
            if propagation_plot:
                propagation_plots.append(str(propagation_plot))
            if map_plot:
                map_plots.append(str(map_plot))
        collected[selection] = {
            "propagation_vs_reference_plots": propagation_plots,
            "map_vs_reference_plots": map_plots,
        }
    return collected


def _validate_overlay_outputs(overlay_paths: dict[str, dict[str, list[str]]]) -> list[str]:
    missing: list[str] = []
    for selection, payload in overlay_paths.items():
        propagation = payload["propagation_vs_reference_plots"]
        map_plots = payload["map_vs_reference_plots"]
        if not propagation:
            missing.append(f"{selection}: missing propagation_vs_reference overlays")
        if not map_plots:
            missing.append(f"{selection}: missing map_vs_reference overlays")
        for plot in propagation + map_plots:
            if not Path(plot).exists():
                missing.append(f"{selection}: missing file {plot}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run o369 local validation workflows (full/reduced x EMB compression/indentation) "
            "without SLURM, then generate propagation and MAP overlays."
        )
    )
    parser.add_argument("--selection", action="append", default=[])
    parser.add_argument("--all-lanes", action="store_true", default=False)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--phase2-cpu-ranks", type=int, default=2)
    parser.add_argument(
        "--inference-device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device for phase1 and phase3b. Phase2 remains CPU MPI.",
    )
    parser.add_argument(
        "--propagation-device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device for propagation phase3b.",
    )
    parser.add_argument("--smoke-pop-size", type=int, default=64)
    parser.add_argument("--smoke-max-gen", type=int, default=1)
    parser.add_argument("--smoke-map-n-displacements", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    args = parser.parse_args(argv)

    if args.phase2_cpu_ranks < 1:
        raise ValueError("--phase2-cpu-ranks must be >= 1")

    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    selections = _resolve_selections(args.selection, args.all_lanes)
    smoke_overrides = dict(DEFAULT_SMOKE_OVERRIDES)
    smoke_overrides["pop_size"] = args.smoke_pop_size
    smoke_overrides["hbi_pop_size"] = args.smoke_pop_size
    smoke_overrides["phase3b_pop_size"] = args.smoke_pop_size
    smoke_overrides["max_gen"] = args.smoke_max_gen
    smoke_overrides["phase3b_max_gen"] = args.smoke_max_gen
    smoke_overrides["map_n_displacements"] = args.smoke_map_n_displacements

    config_overrides: dict[str, str] = {}
    for selection in selections:
        derived_config = _derive_validation_smoke_config(selection, output_root, smoke_overrides)
        config_overrides[selection_key(selection)] = str(derived_config)

    device_contract = {
        "inference_requested": args.inference_device,
        "inference_effective": args.inference_device,
        "propagation_requested": args.propagation_device,
        "propagation_effective": args.propagation_device,
    }
    matrix_root = output_root / "matrix"
    runtime_env, runtime_notes = _build_runtime_env(args.python_bin)
    command = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "vega" / "run_workflow_matrix.py"),
        "--output-root",
        str(matrix_root),
        "--python-bin",
        args.python_bin,
        "--phase2-cpu-ranks",
        str(args.phase2_cpu_ranks),
        "--inference-device",
        args.inference_device,
        "--propagation-device",
        args.propagation_device,
        "--skip-phase1-map",
    ]
    if args.continue_on_error:
        command.append("--continue-on-error")
    for selection in selections:
        key = selection_key(selection)
        command.extend(["--selection", key])
        command.extend(["--config-override", f"{key}={config_overrides[key]}"])

    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env=runtime_env,
    )

    stdout_log = logs_root / "local_matrix.stdout.log"
    stderr_log = logs_root / "local_matrix.stderr.log"
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")

    plots: dict[str, object] = {}
    plot_error: str | None = None
    overlay_paths: dict[str, dict[str, list[str]]] = {}
    missing_outputs: list[str] = []
    status = "failed"

    if result.returncode == 0:
        try:
            plots = render_production_sanity_plots(
                REPO_ROOT,
                selections=selections,
                sanity_configs=config_overrides,
                matrix_root=matrix_root,
                output_root=output_root,
            )
            overlay_paths = _collect_overlay_paths(plots)
            missing_outputs = _validate_overlay_outputs(overlay_paths)
            status = "passed" if not missing_outputs else "failed"
        except Exception as exc:  # pragma: no cover - integration guarded
            plot_error = str(exc)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "status": status,
        "phase2_backend_contract": "dual_backend",
        "phase2_backend_default_for_profile": "cpu-mpi",
        "phase2_backend_effective": "cpu-mpi",
        "python_bin": args.python_bin,
        "phase2_cpu_ranks": args.phase2_cpu_ranks,
        "device_contract": device_contract,
        "compatibility_warnings": [],
        "runtime_notes": runtime_notes,
        "smoke_overrides": smoke_overrides,
        "selections": [selection_key(selection) for selection in selections],
        "config_overrides": dict(config_overrides),
        "matrix": {
            "command": command,
            "command_text": format_command(command),
            "returncode": result.returncode,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "report_path": str(matrix_root / "workflow_matrix_report.json"),
        },
        "overlays": overlay_paths,
        "missing_outputs": missing_outputs,
        "plots": plots,
        "plot_error": plot_error,
    }
    report_path = output_root / "local_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Local matrix command: {format_command(command)}")
    for note in runtime_notes:
        print(f"Runtime note: {note}")
    print(f"Local matrix report:  {report_path}")
    print(f"Local matrix status:  {status}")
    if result.returncode != 0:
        print(f"Matrix stage return code: {result.returncode}")
        return result.returncode
    if status != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
