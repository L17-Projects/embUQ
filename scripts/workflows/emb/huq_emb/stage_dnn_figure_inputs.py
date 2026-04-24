#!/usr/bin/env python3
"""Stage DNN grouped-holdout and Sobol inputs for paper figures under paper_data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_DATASETS = (
    {"name": "compression_2.1um", "modality": "compression", "diameter_dir": "2.1um", "sobol_name": "sobol_vs_disp_2.1um.csv"},
    {"name": "compression_2.9um", "modality": "compression", "diameter_dir": "2.9um", "sobol_name": "sobol_vs_disp_2.9um.csv"},
    {"name": "compression_3.0um", "modality": "compression", "diameter_dir": "3.0um", "sobol_name": "sobol_vs_disp_3.0um.csv"},
    {"name": "indentation_3.2um", "modality": "indentation", "diameter_dir": "3.2um", "sobol_name": "sobol_vs_force_3.2um.csv"},
    {"name": "indentation_3.4um", "modality": "indentation", "diameter_dir": "3.4um", "sobol_name": "sobol_vs_force_3.4um.csv"},
    {"name": "indentation_5.8um", "modality": "indentation", "diameter_dir": "5.8um", "sobol_name": "sobol_vs_force_5.8um.csv"},
)
DEFAULT_PARAMETERS = ("Yt", "kb", "b1", "b2", "a3", "a4")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _runner_path(site: str, stem: str) -> Path:
    return REPO_ROOT / "scripts" / "platforms" / site / stem


def _postprocess_script(name: str) -> Path:
    return REPO_ROOT / "scripts" / "shared" / "postprocess" / name


def _dataset_specs(*, selected: list[str] | None = None) -> list[dict[str, str]]:
    if not selected:
        return [dict(item) for item in DEFAULT_DATASETS]
    allowed = set(selected)
    filtered = [dict(item) for item in DEFAULT_DATASETS if item["name"] in allowed]
    if not filtered:
        raise ValueError(f"No datasets matched --only values: {sorted(allowed)}")
    return filtered


def _staging_roots(
    *,
    paper_data_root: Path,
    campaign_id: str,
    staging_dirname: str,
) -> dict[str, Path]:
    staging_root = paper_data_root / "runs" / campaign_id / "postprocess_graph" / staging_dirname
    return {
        "paper_data_root": paper_data_root,
        "campaign_root": paper_data_root / "runs" / campaign_id,
        "staging_root": staging_root,
        "group_holdout_root": staging_root / "group_holdout",
        "sobol_root": staging_root / "sobol",
        "holdout_figures_root": staging_root / "figures" / "holdout_l2",
        "sensitivity_figures_root": staging_root / "figures" / "sensitivity",
        "logs_root": staging_root / "logs",
        "manifest_path": staging_root / "dnn_figure_input_staging_report.json",
    }


def _expected_holdout_outputs(root: Path, specs: list[dict[str, str]]) -> list[Path]:
    paths = [
        root / spec["modality"] / spec["diameter_dir"] / "dnn" / "summary.json"
        for spec in specs
    ]
    paths.extend(
        [
            root / "surrogate_group_holdout_report.csv",
            root / "surrogate_group_holdout_report.json",
        ]
    )
    return paths


def _expected_sobol_outputs(root: Path, specs: list[dict[str, str]]) -> list[Path]:
    paths = [
        root / spec["modality"] / "dnn" / spec["sobol_name"]
        for spec in specs
    ]
    paths.extend(
        [
            root / "sobol_matrix_report.csv",
            root / "sobol_matrix_report.json",
        ]
    )
    return paths


def _expected_summary_outputs(roots: dict[str, Path]) -> list[Path]:
    return [
        *_expected_holdout_summary_outputs(roots),
        *_expected_sensitivity_summary_outputs(roots),
    ]


def _expected_holdout_summary_outputs(roots: dict[str, Path]) -> list[Path]:
    return [
        roots["holdout_figures_root"] / "surrogate_holdout_l2_summary.csv",
        roots["holdout_figures_root"] / "surrogate_holdout_l2_comparison.png",
        roots["holdout_figures_root"] / "surrogate_holdout_l2_comparison.pdf",
    ]


def _expected_sensitivity_summary_outputs(roots: dict[str, Path]) -> list[Path]:
    return [
        roots["sensitivity_figures_root"] / "surrogate_sensitivity_summary.csv",
        roots["sensitivity_figures_root"] / "surrogate_sensitivity_comparison.png",
        roots["sensitivity_figures_root"] / "surrogate_sensitivity_comparison.pdf",
    ]


def _outputs_ready(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _run_step(*, name: str, command: list[str], logs_root: Path) -> dict[str, Any]:
    logs_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    start_utc = _utc_now_iso()
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    stdout_log = logs_root / f"{name}.stdout.log"
    stderr_log = logs_root / f"{name}.stderr.log"
    stdout_log.write_text(proc.stdout or "", encoding="utf-8")
    stderr_log.write_text(proc.stderr or "", encoding="utf-8")
    return {
        "name": name,
        "status": "passed" if proc.returncode == 0 else "failed",
        "command": command,
        "returncode": int(proc.returncode),
        "start_utc": start_utc,
        "end_utc": _utc_now_iso(),
        "elapsed_seconds": elapsed,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def _skip_step(*, name: str, reason: str, outputs: list[Path]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "skipped_existing",
        "reason": reason,
        "expected_outputs": [str(path) for path in outputs],
    }


def _mark_missing_outputs(step: dict[str, Any], outputs: list[Path]) -> dict[str, Any]:
    missing = [str(path) for path in outputs if not path.exists()]
    if not missing:
        return step
    step = dict(step)
    step["status"] = "failed"
    step["returncode"] = 1
    step["missing_outputs"] = missing
    return step


def _build_group_holdout_command(
    *,
    python_bin: str,
    site: str,
    output_root: Path,
    only: list[str],
    seed: int,
    val_fraction: float,
    predictive_mc_samples: int,
    predictive_mc_chunk_size: int,
    device: str,
    disp_source: str,
    rupture_ratio: float,
) -> list[str]:
    command = [
        python_bin,
        str(_runner_path(site, "run_surrogate_group_holdout.py")),
        "--python-bin",
        python_bin,
        "--output-root",
        str(output_root),
        "--site",
        site,
        "--surrogate-family",
        "dnn",
        "--seed",
        str(seed),
        "--val-fraction",
        str(val_fraction),
        "--predictive-mc-samples",
        str(predictive_mc_samples),
        "--predictive-mc-chunk-size",
        str(predictive_mc_chunk_size),
        "--device",
        device,
        "--disp-source",
        disp_source,
        "--rupture-ratio",
        str(rupture_ratio),
    ]
    for item in only:
        command.extend(["--only", item])
    return command


def _build_sobol_command(
    *,
    python_bin: str,
    site: str,
    output_root: Path,
    only: list[str],
    n_samples: int,
    n_axis_points: int,
    predictive_mc_samples: int,
    predictive_mc_chunk_size: int,
    device: str,
) -> list[str]:
    command = [
        python_bin,
        str(_runner_path(site, "run_sobol_matrix.py")),
        "--python-bin",
        python_bin,
        "--output-root",
        str(output_root),
        "--site",
        site,
        "--surrogate-family",
        "dnn",
        "--n-samples",
        str(n_samples),
        "--n-axis-points",
        str(n_axis_points),
        "--predictive-mc-samples",
        str(predictive_mc_samples),
        "--predictive-mc-chunk-size",
        str(predictive_mc_chunk_size),
        "--device",
        device,
    ]
    for item in only:
        command.extend(["--only", item])
    return command


def _build_summary_commands(
    *,
    python_bin: str,
    roots: dict[str, Path],
    index_type: str,
    parameters: list[str],
) -> list[tuple[str, list[str]]]:
    return [
        (
            "holdout_summary",
            [
                python_bin,
                str(_postprocess_script("generate_surrogate_holdout_l2_figure.py")),
                "--input-root",
                str(roots["group_holdout_root"]),
                "--output-dir",
                str(roots["holdout_figures_root"]),
                "--metric",
                "median",
            ],
        ),
        (
            "sensitivity_summary",
            [
                python_bin,
                str(_postprocess_script("generate_surrogate_sensitivity_figure.py")),
                "--input-root",
                str(roots["sobol_root"]),
                "--output-dir",
                str(roots["sensitivity_figures_root"]),
                "--index-type",
                index_type,
                "--parameters",
                *parameters,
            ],
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage DNN grouped-holdout and Sobol figure inputs under paper_data/runs/<campaign>/postprocess_graph."
    )
    parser.add_argument("--paper-data-root", required=True)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--site", choices=["vega", "karolina"], default="vega")
    parser.add_argument("--staging-dirname", default="dnn_figure_input_staging")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-holdout-seed", type=int, default=20260317)
    parser.add_argument("--group-holdout-val-fraction", type=float, default=0.10)
    parser.add_argument("--predictive-mc-samples", type=int, default=64)
    parser.add_argument("--predictive-mc-chunk-size", type=int, default=8)
    parser.add_argument("--disp-source", choices=["auto", "diameter", "displacement"], default="auto")
    parser.add_argument("--rupture-ratio", type=float, default=2.0)
    parser.add_argument("--sobol-n-samples", type=int, default=1024)
    parser.add_argument("--sobol-n-axis-points", type=int, default=20)
    parser.add_argument("--index-type", choices=["S1", "ST"], default="ST")
    parser.add_argument("--parameters", nargs="+", default=list(DEFAULT_PARAMETERS))
    parser.add_argument("--skip-group-holdout", action="store_true", default=False)
    parser.add_argument("--skip-sobol", action="store_true", default=False)
    parser.add_argument("--skip-summary-figures", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args(argv)

    campaign_id = args.campaign_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paper_data_root = _resolve_path(args.paper_data_root)
    specs = _dataset_specs(selected=args.only)
    roots = _staging_roots(
        paper_data_root=paper_data_root,
        campaign_id=campaign_id,
        staging_dirname=args.staging_dirname,
    )
    for path in roots.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    holdout_outputs = _expected_holdout_outputs(roots["group_holdout_root"], specs)
    sobol_outputs = _expected_sobol_outputs(roots["sobol_root"], specs)
    summary_outputs = _expected_summary_outputs(roots)
    holdout_summary_outputs = _expected_holdout_summary_outputs(roots)
    sensitivity_summary_outputs = _expected_sensitivity_summary_outputs(roots)

    steps: list[dict[str, Any]] = []
    failed = False

    if args.skip_group_holdout:
        steps.append(_skip_step(name="group_holdout", reason="skip requested", outputs=holdout_outputs))
    elif not args.force and _outputs_ready(holdout_outputs):
        steps.append(_skip_step(name="group_holdout", reason="outputs already exist", outputs=holdout_outputs))
    else:
        step = _run_step(
            name="group_holdout",
            command=_build_group_holdout_command(
                python_bin=args.python_bin,
                site=args.site,
                output_root=roots["group_holdout_root"],
                only=[spec["name"] for spec in specs],
                seed=args.group_holdout_seed,
                val_fraction=args.group_holdout_val_fraction,
                predictive_mc_samples=args.predictive_mc_samples,
                predictive_mc_chunk_size=args.predictive_mc_chunk_size,
                device=args.device,
                disp_source=args.disp_source,
                rupture_ratio=args.rupture_ratio,
            ),
            logs_root=roots["logs_root"],
        )
        step = _mark_missing_outputs(step, holdout_outputs)
        steps.append(step)
        failed = step["returncode"] != 0

    if not failed:
        if args.skip_sobol:
            steps.append(_skip_step(name="sobol", reason="skip requested", outputs=sobol_outputs))
        elif not args.force and _outputs_ready(sobol_outputs):
            steps.append(_skip_step(name="sobol", reason="outputs already exist", outputs=sobol_outputs))
        else:
            step = _run_step(
                name="sobol",
                command=_build_sobol_command(
                    python_bin=args.python_bin,
                    site=args.site,
                    output_root=roots["sobol_root"],
                    only=[spec["name"] for spec in specs],
                    n_samples=args.sobol_n_samples,
                    n_axis_points=args.sobol_n_axis_points,
                    predictive_mc_samples=args.predictive_mc_samples,
                    predictive_mc_chunk_size=args.predictive_mc_chunk_size,
                    device=args.device,
                ),
                logs_root=roots["logs_root"],
            )
            step = _mark_missing_outputs(step, sobol_outputs)
            steps.append(step)
            failed = step["returncode"] != 0

    if not failed:
        summary_commands = _build_summary_commands(
            python_bin=args.python_bin,
            roots=roots,
            index_type=args.index_type,
            parameters=list(args.parameters),
        )
        if args.skip_summary_figures:
            for name, _ in summary_commands:
                steps.append(_skip_step(name=name, reason="skip requested", outputs=summary_outputs))
        elif not args.force and _outputs_ready(summary_outputs):
            for name, _ in summary_commands:
                steps.append(_skip_step(name=name, reason="outputs already exist", outputs=summary_outputs))
        else:
            for name, command in summary_commands:
                step = _run_step(name=name, command=command, logs_root=roots["logs_root"])
                if name == "holdout_summary":
                    step = _mark_missing_outputs(step, holdout_summary_outputs)
                else:
                    step = _mark_missing_outputs(step, sensitivity_summary_outputs)
                steps.append(step)
                if step["returncode"] != 0:
                    failed = True
                    break

    report = {
        "created_at_utc": _utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "paper_data_root": str(paper_data_root),
        "campaign_id": campaign_id,
        "site": args.site,
        "staging_root": str(roots["staging_root"]),
        "group_holdout_root": str(roots["group_holdout_root"]),
        "sobol_root": str(roots["sobol_root"]),
        "holdout_figures_root": str(roots["holdout_figures_root"]),
        "sensitivity_figures_root": str(roots["sensitivity_figures_root"]),
        "selected_datasets": [spec["name"] for spec in specs],
        "expected_outputs": {
            "group_holdout": [str(path) for path in holdout_outputs],
            "sobol": [str(path) for path in sobol_outputs],
            "summaries": [str(path) for path in summary_outputs],
        },
        "steps": steps,
        "status": "failed" if failed else "passed",
    }
    roots["manifest_path"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Staging root: {roots['staging_root']}")
    print(f"Report: {roots['manifest_path']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
