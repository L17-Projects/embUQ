#!/usr/bin/env python3
"""Evaluate MAP parameters with Mirheo DPD simulation (Optimized Grid) — compression.

Runs a single Mirheo simulation for the MAP parameters using an optimized
displacement grid that extends ±X% beyond the experimental range (after d0
shift is applied).

Supports automatic retry with reduced timestep for stability:
- retry_attempt=0: Use default dt from parameters-default*.yaml
- retry_attempt=1: dt *= 0.5 (halved)
- retry_attempt=2: dt *= 0.25 (quartered)
- etc.

Usage:
    mpirun --oversubscribe -n 2 python3.8 \\
        propagation/scripts/evaluate_map_mirheo_optimized.py \\
        --map-file compression_2.1um_map.json \\
        --output result_2.1um.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "compression"))
sys.path.insert(0, str(PROJECT_ROOT / "compression" / "evalkit"))

from compression.evalkit.posterior_compression import compute_compression
from compression.evalkit.tools import datedPrint, getReferencePoints


def load_map_parameters(map_file: str) -> dict:
    """Load MAP parameters from JSON file."""
    with open(map_file, "r") as f:
        return json.load(f)


def calculate_optimized_displacement_grid(
    diameter_um: float, d0_offset: float, n_points: int = 15, extend_pct: float = 0.10
) -> np.ndarray:
    """Return PRE-SHIFT displacement grid spanning ±extend_pct beyond the experimental range.

    The d0 shift is applied in post-processing/plotting — do NOT add it here.
    """
    exp_displ = np.array(getReferencePoints(diameter_um))
    exp_min = exp_displ.min()
    exp_max = exp_displ.max()
    exp_range = exp_max - exp_min

    target_min = exp_min - extend_pct * exp_range
    target_max = exp_max + extend_pct * exp_range

    return np.linspace(target_min - d0_offset, target_max - d0_offset, n_points)


def setup_map_specific_init_dir(
    diameter_um: float, retry_attempt: int, dt_scale_factor: float, rank: int
) -> str:
    """Create MAP-specific init directory and apply dt/numsteps scaling for retries.

    Returns the path to the MAP-specific directory.
    """
    comm = MPI.COMM_WORLD
    original_init_dir = str(PROJECT_ROOT / f"_init_compression_{diameter_um}um")
    map_init_dir = str(PROJECT_ROOT / f"_init_compression_{diameter_um}um_map")

    if rank == 0:
        if os.path.exists(map_init_dir):
            shutil.rmtree(map_init_dir)

        datedPrint("[MAP] Creating MAP-specific init directory")
        datedPrint(f"  Source: {original_init_dir}")
        datedPrint(f"  Target: {map_init_dir}")
        shutil.copytree(original_init_dir, map_init_dir)

        if retry_attempt > 0:
            dt_multiplier = dt_scale_factor**retry_attempt
            numsteps_multiplier = 1.0 / dt_multiplier
            datedPrint(
                f"[MAP] [Retry {retry_attempt}] Scaling dt by {dt_multiplier:.4f}, "
                f"numsteps by {numsteps_multiplier:.4f}"
            )

            param_dir = os.path.join(map_init_dir, "parameter")
            param_files = [
                f
                for f in os.listdir(param_dir)
                if f.startswith("parameters-default") and f.endswith(".yaml")
            ]

            for param_file in param_files:
                filepath = os.path.join(param_dir, param_file)
                with open(filepath, "r") as f:
                    params = yaml.load(f, Loader=yaml.CLoader)

                if "dt" in params:
                    old = params["dt"]
                    params["dt"] = float(old) * dt_multiplier
                    datedPrint(f"  {param_file}: dt {old:.6f} → {params['dt']:.6f}")

                if "dt_eq" in params:
                    old = params["dt_eq"]
                    params["dt_eq"] = float(old) * dt_multiplier
                    datedPrint(f"  {param_file}: dt_eq {old:.6f} → {params['dt_eq']:.6f}")

                if "numsteps" in params:
                    old = params["numsteps"]
                    params["numsteps"] = int(old * numsteps_multiplier)
                    datedPrint(f"  {param_file}: numsteps {old} → {params['numsteps']}")

                if "numsteps_dt" in params:
                    old = params["numsteps_dt"]
                    params["numsteps_dt"] = int(old * numsteps_multiplier)
                    datedPrint(f"  {param_file}: numsteps_dt {old} → {params['numsteps_dt']}")

                with open(filepath, "w") as f:
                    yaml.dump(
                        params, f, Dumper=yaml.CDumper, default_flow_style=False, sort_keys=False
                    )

            datedPrint(f"[MAP] [Retry {retry_attempt}] dt scaling complete")
        else:
            datedPrint("[MAP] Init directory ready (default dt)")

    comm.Barrier()
    return map_init_dir


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print(f"ERROR: requires exactly 2 MPI ranks (got {size})")
            print("Usage: mpirun -n 2 python3 evaluate_map_mirheo_optimized.py ...")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Evaluate MAP parameters with Mirheo (optimized grid) — compression"
    )
    parser.add_argument("--map-file", required=True, help="Path to MAP parameters JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--n-displacements", type=int, default=15)
    parser.add_argument("--extend-range", type=float, default=0.10)
    parser.add_argument(
        "--retry-attempt", type=int, default=0,
        help="Retry attempt number (0=first attempt, >= 1 triggers dt scaling)"
    )
    parser.add_argument("--dt-scale-factor", type=float, default=0.5)
    parser.add_argument("--numsteps", type=int, default=None)
    parser.add_argument("--numsteps-eq", type=int, default=None)
    args = parser.parse_args()

    if args.retry_attempt < 0:
        if rank == 0:
            print("ERROR: --retry-attempt must be >= 0")
        sys.exit(1)

    map_data = load_map_parameters(args.map_file)
    diameter_um = map_data["diameter_um"]
    map_params = list(map_data["parameters"])
    param_names = map_data["parameter_names"]

    if "d0" not in param_names:
        if rank == 0:
            print(f"ERROR: 'd0' not found in parameter_names: {param_names}")
        sys.exit(1)
    d0_idx = param_names.index("d0")
    d0_offset = map_params[d0_idx]

    if rank == 0:
        datedPrint("=" * 70)
        datedPrint(f"MAP MIRHEO EVALUATION (OPTIMIZED) — COMPRESSION {diameter_um} µm")
        datedPrint("=" * 70)
        datedPrint(f"MAP file: {args.map_file}")
        datedPrint(f"Sample ID: {map_data['sample_id']}")
        datedPrint(f"Log Posterior: {map_data['logPosterior']:.6f}")
        datedPrint(f"MPI ranks: {size}")
        if args.retry_attempt > 0:
            dt_multiplier = args.dt_scale_factor**args.retry_attempt
            datedPrint(f"RETRY ATTEMPT {args.retry_attempt} (dt x {dt_multiplier:.4f})")
        for name, value in zip(param_names, map_params):
            datedPrint(f"  {name:6s} = {value:.6g}")
        datedPrint(
            f"Grid: {args.n_displacements} pts, ±{args.extend_range*100:.0f}% extended, "
            f"d0={d0_offset:.6f}"
        )

    displacement_points = calculate_optimized_displacement_grid(
        diameter_um, d0_offset, args.n_displacements, args.extend_range
    )

    # Expand 3-param reduced model [Yt, kb, d0] → 7-param [Yt, kb, 0, 0, 0, 0, d0]
    if len(map_params) == 3 and param_names == ["Yt", "kb", "d0"]:
        if rank == 0:
            datedPrint("Detected reduced model — expanding to 7-param for Mirheo")
        map_params_for_sim = [map_params[0], map_params[1], 0.0, 0.0, 0.0, 0.0, map_params[2]]
    else:
        map_params_for_sim = map_params

    sample = {"Parameters": map_params_for_sim, "Sample Id": map_data["sample_id"]}

    map_init_dir = setup_map_specific_init_dir(
        diameter_um, args.retry_attempt, args.dt_scale_factor, rank
    )

    if rank == 0 and (args.numsteps is not None or args.numsteps_eq is not None):
        param_dir = os.path.join(map_init_dir, "parameter")
        for fname in os.listdir(param_dir):
            if fname.startswith("parameters-default") and fname.endswith(".yaml"):
                fpath = os.path.join(param_dir, fname)
                with open(fpath, "r") as f:
                    params = yaml.load(f, Loader=yaml.CLoader)
                if args.numsteps is not None and "numsteps" in params:
                    old = params["numsteps"]
                    params["numsteps"] = args.numsteps
                    datedPrint(f"[MAP] {fname}: numsteps override {old} → {args.numsteps}")
                if args.numsteps_eq is not None and "numsteps_eq" in params:
                    old = params["numsteps_eq"]
                    params["numsteps_eq"] = args.numsteps_eq
                    datedPrint(f"[MAP] {fname}: numsteps_eq override {old} → {args.numsteps_eq}")
                with open(fpath, "w") as f:
                    yaml.dump(params, f, Dumper=yaml.CDumper, default_flow_style=False,
                              sort_keys=False)

    comm.Barrier()

    compute_compression(sample, displacement_points.tolist(), diameter_um,
                        init_compression_path=map_init_dir)

    if rank == 0:
        forces = np.array(sample["Reference Evaluations"])
        datedPrint(f"Force range: [{forces.min():.2f}, {forces.max():.2f}] (DPD units)")

        result = {
            "diameter_um": diameter_um,
            "map_parameters": map_params,
            "parameter_names": param_names,
            "displacement_points": displacement_points.tolist(),
            "forces": forces.tolist(),
            "sample_id": map_data["sample_id"],
            "generation": map_data["generation"],
            "logPosterior": map_data["logPosterior"],
            "evaluation_mode": "mirheo_map_optimized",
            "grid_config": {
                "n_points": args.n_displacements,
                "extend_range_pct": args.extend_range,
                "d0_offset": d0_offset,
                "note": "Displacements are PRE-SHIFT (add d0 for plotting)",
            },
            "retry_attempt": args.retry_attempt,
            "dt_multiplier": args.dt_scale_factor**args.retry_attempt if args.retry_attempt > 0 else 1.0,
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        datedPrint(f"Saved to: {args.output}")

        if os.path.exists(map_init_dir):
            shutil.rmtree(map_init_dir)
            datedPrint("[MAP] Cleaned up MAP-specific init directory")

        datedPrint("=" * 70)


if __name__ == "__main__":
    main()
