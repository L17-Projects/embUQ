#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.hpc_paths import detect_hpc_site  # noqa: E402
from meso_uq.vega_workflows import (  # noqa: E402
    VALID_EXPERIMENTS,
    VALID_INFERENCE_STAGES,
    VALID_MODEL_FAMILIES,
    VALID_PHASE2_BACKENDS,
    VALID_PROFILES,
    VegaWorkflowSelection,
    build_inference_command,
    format_command,
    resolve_workflow_config_path,
    resolve_workflow_output_root,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one public inference stage on Vega with explicit workflow selection."
    )
    parser.add_argument("--experiment", choices=VALID_EXPERIMENTS, required=True)
    parser.add_argument("--model-family", choices=VALID_MODEL_FAMILIES, required=True)
    parser.add_argument("--profile", choices=VALID_PROFILES, required=True)
    parser.add_argument("--stage", choices=VALID_INFERENCE_STAGES, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--cpu-ranks", type=int, default=1)
    parser.add_argument(
        "--phase2-backend",
        choices=VALID_PHASE2_BACKENDS,
        default=None,
        help="Phase 2 backend: production defaults to native-cuda; validation defaults to cpu-mpi.",
    )
    parser.add_argument("--profiling", action="store_true", default=False)
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="gpu (default): Sequential GPU-batch (single rank); cpu: Distributed MPI",
    )
    args = parser.parse_args(argv)

    selection = VegaWorkflowSelection(args.experiment, args.model_family, args.profile)
    config_path = resolve_workflow_config_path(REPO_ROOT, selection, args.config)
    resolved_site = args.site if args.site is not None else detect_hpc_site()
    output_root = resolve_workflow_output_root(
        REPO_ROOT,
        selection,
        args.output_dir,
        run_tag=args.run_tag,
        site=resolved_site,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    command = build_inference_command(
        REPO_ROOT,
        selection,
        stage=args.stage,
        python_bin=args.python_bin,
        config_path=config_path,
        output_root=output_root,
        cpu_ranks=args.cpu_ranks,
        profiling=args.profiling,
        restart=args.restart,
        dry_run=args.dry_run,
        device=args.device,
        phase2_backend=args.phase2_backend,
        dataset_name=args.dataset_name,
        diameter=args.diameter,
    )

    print(f"Experiment:    {selection.experiment}")
    print(f"Model family:  {selection.model_family}")
    print(f"Profile:       {selection.profile}")
    print(f"Stage:         {args.stage}")
    print(f"Config:        {config_path}")
    print(f"Output root:   {output_root}")
    print(f"Command:       {format_command(command)}")

    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
