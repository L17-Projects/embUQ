#!/usr/bin/env python3

import argparse
import os

import pandas as pd
import yaml
from scipy.stats import qmc


DEFAULT_BOUNDS = {
    "Yt": [1.0e7, 1.0e8],
    "kb": [100.0, 1000.0],
    "b1": [0.0, 3.0],
    "b2": [0.0, 10.0],
    "a3": [-2.5, 3.0],
    "a4": [0.0, 4.0],
}


def load_config(path):
    if path is None:
        return {"sampling": {"n_samples": 512}, "parameter_bounds": DEFAULT_BOUNDS}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Generate Latin Hypercube parameter designs for surrogate-data workflows")
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-samples", type=int, default=None)
    args = ap.parse_args()
    config = load_config(args.config)
    bounds_dict = config.get("parameter_bounds", DEFAULT_BOUNDS)
    n_samples = args.n_samples or config.get("sampling", {}).get("n_samples", 512)
    param_names = ["Yt", "kb", "b1", "b2", "a3", "a4"]
    bounds = [bounds_dict[name] for name in param_names]
    sampler = qmc.LatinHypercube(d=len(param_names))
    unit_samples = sampler.random(n_samples)
    scaled = qmc.scale(unit_samples, [b[0] for b in bounds], [b[1] for b in bounds])
    df = pd.DataFrame(scaled, columns=param_names)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} LHS samples -> {args.output}")


if __name__ == "__main__":
    main()
