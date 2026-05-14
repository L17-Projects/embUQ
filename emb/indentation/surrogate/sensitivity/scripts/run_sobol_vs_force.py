#!/usr/bin/env python3

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    # Allow direct script execution from a source checkout.
    sys.path.insert(0, str(REPO_ROOT))

from meso_uq.sensitivity import build_problem, run_sobol_over_axis
from meso_uq.surrogate.bnn import VariationalBNNPredictor
from emb.indentation.surrogate.sensitivity.scripts.prior import ind_variables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--surrogate-family", choices=["dnn", "bnn"], default="dnn")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--n-forces", type=int, default=20)
    ap.add_argument("--predictive-mc-samples", type=int, default=64)
    ap.add_argument("--predictive-mc-chunk-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    problem = build_problem(ind_variables)
    fixed_axis_values = np.linspace(0.0, 24000.0, args.n_forces).tolist()
    eval_cols = ["Yt", "kb", "b1", "b2", "a3", "a4", "F"]
    if args.surrogate_family == "dnn":
        with open(args.model, "rb") as f:
            data = pickle.load(f)
        df = run_sobol_over_axis(
            model=data["model"],
            xshift=data["xshift"],
            xscale=data["xscale"],
            yshift=data["yshift"],
            yscale=data["yscale"],
            problem=problem,
            fixed_axis_name="F",
            fixed_axis_values=fixed_axis_values,
            evaluate_columns=eval_cols,
            n_samples=args.n_samples,
            calc_second_order=False,
        )
    else:
        predictor = VariationalBNNPredictor(args.model, device=args.device)
        df = run_sobol_over_axis(
            predictor=predictor,
            problem=problem,
            fixed_axis_name="F",
            fixed_axis_values=fixed_axis_values,
            evaluate_columns=eval_cols,
            n_samples=args.n_samples,
            calc_second_order=False,
            predictive_mc_samples=args.predictive_mc_samples,
            predictive_mc_chunk_size=args.predictive_mc_chunk_size,
        )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
