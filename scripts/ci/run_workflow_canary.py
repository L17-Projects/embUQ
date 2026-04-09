#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.experiments import load_experiments
from meso_uq.vega_workflows import selection_slug
from meso_uq.vega_workflows import VegaWorkflowSelection

DEFAULT_SELECTION = "compression:reduced-model:validation"
DEFAULT_CONFIG = REPO_ROOT / "reduced" / "configs" / "ci" / "ci_canary_config_compression.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_ci" / "workflow_canary"


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _load_datasets(config_path: Path, experiment_name: str) -> list[tuple[float, str]]:
    with config_path.open("rb") as handle:
        config = yaml.safe_load(handle)
    experiments = [exp for exp in load_experiments(config, REPO_ROOT) if exp.enabled and exp.name == experiment_name]
    datasets: list[tuple[float, str]] = []
    for exp in experiments:
        for diameter_um in exp.diameters:
            datasets.append((float(diameter_um), exp.dataset_name(diameter_um)))
    if not datasets:
        raise ValueError(f"No enabled datasets found for experiment '{experiment_name}' in {config_path}")
    return datasets


def _required_artifacts(workflow_dir: Path, datasets: list[tuple[float, str]]) -> dict[str, str]:
    first_diameter, first_dataset = datasets[0]
    diameter_label = f"{first_diameter:g}um"
    return {
        "workflow_summary": str(workflow_dir / "summary.json"),
        "phase1_latest": str(workflow_dir / "results" / "results_phase_1" / first_dataset / "latest"),
        "phase2_latest": str(workflow_dir / "results" / "results_phase_2" / "latest"),
        "phase3b_latest": str(workflow_dir / "results" / "results_phase_3b" / first_dataset / "latest"),
        "phase3b_propagation": str(workflow_dir / "results" / "propagation_phase3b" / first_dataset / "summary.csv"),
        "phase3b_map_manifest": str(workflow_dir / "map_phase3b" / "all_diameters_map.json"),
        "overlay_plot": str(workflow_dir / "overlay_uq_ref" / f"uq_overlay_{diameter_label}.png"),
        "posterior_plot": str(workflow_dir / "posteriors_phase3b" / f"posterior_marginals_{diameter_label}.png"),
    }


def _assert_artifacts_exist(artifacts: dict[str, str]) -> None:
    missing = [path for path in artifacts.values() if not Path(path).exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Workflow canary completed but required artifacts are missing:\n{formatted}")


def _run_logged_command(command: list[str], cwd: Path, output_root: Path) -> dict[str, str]:
    stdout_log = output_root / "workflow_canary.stdout.log"
    stderr_log = output_root / "workflow_canary.stderr.log"
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    stdout_log.write_text(result.stdout or "", encoding="utf-8")
    stderr_log.write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return {
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GitHub CI real workflow canary and assert real artifacts.")
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--cpu-ranks", type=int, default=2)
    args = parser.parse_args(argv)

    selection_parts = args.selection.split(":")
    if len(selection_parts) != 3:
        raise ValueError(f"Expected selection in experiment:model-family:profile form, got '{args.selection}'")
    selection = VegaWorkflowSelection(*selection_parts)

    config_path = _resolve_repo_path(args.config)
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    command = [
        args.python_bin,
        str(REPO_ROOT / "scripts" / "vega" / "run_validation_suite.py"),
        "--workflows",
        args.selection,
        "--output-root",
        str(output_root),
        "--python-bin",
        args.python_bin,
        "--cpu-ranks",
        str(args.cpu_ranks),
        "--config-override",
        f"{args.selection}={config_path}",
    ]
    log_artifacts = _run_logged_command(command, REPO_ROOT, output_root)

    workflow_dir = output_root / selection_slug(selection)
    suite_summary_path = output_root / "workflow_suite_summary.json"
    datasets = _load_datasets(config_path, selection.experiment)
    artifacts = _required_artifacts(workflow_dir, datasets)
    artifacts["suite_summary"] = str(suite_summary_path)
    artifacts.update(log_artifacts)
    _assert_artifacts_exist(artifacts)

    summary = json.loads((workflow_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("selection") != args.selection:
        raise ValueError(f"Workflow summary selection mismatch: expected {args.selection}, got {summary.get('selection')}")

    report = {
        "status": "passed",
        "selection": args.selection,
        "config": str(config_path),
        "output_root": str(output_root),
        "cpu_ranks": args.cpu_ranks,
        "command": command,
        "datasets": [
            {"diameter_um": diameter_um, "dataset": dataset_name}
            for diameter_um, dataset_name in datasets
        ],
        "artifacts": artifacts,
    }
    report_path = output_root / "workflow_canary_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Workflow canary report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
