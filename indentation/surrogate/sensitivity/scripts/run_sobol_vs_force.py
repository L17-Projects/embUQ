#!/usr/bin/env python3

import argparse
import os
import pickle
import numpy as np

from meso_uq.sensitivity import build_problem, run_sobol_over_axis
from indentation.surrogate.sensitivity.scripts.prior import ind_variables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--n-forces", type=int, default=20)
    args = ap.parse_args()
    with open(args.model, "rb") as f:
        data = pickle.load(f)
    df = run_sobol_over_axis(
        model=data["model"],
        xshift=data["xshift"],
        xscale=data["xscale"],
        yshift=data["yshift"],
        yscale=data["yscale"],
        problem=build_problem(ind_variables),
        fixed_axis_name="F",
        fixed_axis_values=np.linspace(0.0, 24000.0, args.n_forces).tolist(),
        evaluate_columns=["Yt", "kb", "b1", "b2", "a3", "a4", "F"],
        n_samples=args.n_samples,
        calc_second_order=False,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
