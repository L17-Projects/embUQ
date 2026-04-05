#!/usr/bin/env python3

import argparse

from meso_uq.postprocess import plot_d0_correlations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--d0-col", default="d0")
    args = ap.parse_args()
    plot_d0_correlations(args.samples_csv, args.output, d0_col=args.d0_col)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
