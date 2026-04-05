#!/usr/bin/env python3

import argparse

from meso_uq.postprocess import plot_posterior_marginals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    plot_posterior_marginals(args.samples_csv, args.output)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
