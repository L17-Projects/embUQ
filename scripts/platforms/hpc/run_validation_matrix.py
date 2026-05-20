#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.platforms.site_selector import resolve_hpc_site  # noqa: E402
from meso_uq.hpc_paths import (  # noqa: E402
    default_runs_root,
    detect_hpc_site,
    ensure_canonical_output_root,
)

VALID_EXPERIMENTS = ("compression", "indentation")
VALID_MODEL_FAMILIES = ("full-model", "reduced-model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the public HPC validation matrix across the shipped validation workflows."
    )
    parser.add_argument("--selection", action="append", default=[], help="Explicit selection in experiment:model-family:validation form.")
    parser.add_argument("--experiments", nargs="+", choices=VALID_EXPERIMENTS, default=None)
    parser.add_argument("--model-families", nargs="+", choices=VALID_MODEL_FAMILIES, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--phase2-cpu-ranks", type=int, default=1)
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    parser.add_argument("--run-phase1-map", action="store_true", default=False,
                        help="Run the phase1 MAP extraction step (skipped by default).")
    parser.add_argument("--skip-phase3b-map", action="store_true", default=False)
    parser.add_argument("--skip-phase3b-propagation", action="store_true", default=False)
    parser.add_argument(
        "--run-map-mirheo", action="store_true", default=False,
        help="Run MAP Mirheo DPD evaluation after map_phase3b.",
    )
    parser.add_argument(
        "--map-mirheo-n-displacements", type=int, default=15,
        help="Number of displacement points for MAP Mirheo evaluation.",
    )
    parser.add_argument(
        "--skip-release-manifest", action="store_true", default=False,
        help="Skip paper-release asset gating and emit workflow-only validation evidence.",
    )
    args = parser.parse_args(argv)

    resolved_site = resolve_hpc_site(cli_site=args.site, env=os.environ, allow_hostname=True, default="vega")
    if args.output_root is not None:
        output_root = ensure_canonical_output_root(args.output_root, REPO_ROOT, field_name="--output-root")
    else:
        output_root = default_runs_root(
            REPO_ROOT,
            "validation_matrix",
            site=resolved_site,
            run_tag=args.run_tag,
        )

    command = [args.python_bin, str(SCRIPT_DIR / "run_workflow_matrix.py")]
    for selection in args.selection:
        command.extend(["--selection", selection])
    if args.experiments is not None:
        command.extend(["--experiments", *args.experiments])
    if args.model_families is not None:
        command.extend(["--model-families", *args.model_families])
    command.extend(
        [
            "--profiles",
            "validation",
            "--output-root",
            str(output_root),
            "--phase2-cpu-ranks",
            str(args.phase2_cpu_ranks),
            "--python-bin",
            args.python_bin,
            "--site",
            resolved_site,
        ]
    )
    for item in args.config_override:
        command.extend(["--config-override", item])
    if args.run_tag is not None:
        command.extend(["--run-tag", args.run_tag])
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.run_phase1_map:
        command.append("--run-phase1-map")
    if args.skip_phase3b_map:
        command.append("--skip-phase3b-map")
    if args.skip_phase3b_propagation:
        command.append("--skip-phase3b-propagation")
    if args.run_map_mirheo:
        command.append("--run-map-mirheo")
        command.extend(["--map-mirheo-n-displacements", str(args.map_mirheo_n_displacements)])
    if args.skip_release_manifest:
        command.append("--skip-release-manifest")

    print("Validation profile: validation")
    print(f"Output root:         {output_root}")
    print(f"Command:             {' '.join(command)}")

    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
