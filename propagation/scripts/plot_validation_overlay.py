#!/usr/bin/env python3

import argparse

from meso_uq.postprocess import plot_validation_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_csv")
    ap.add_argument("prediction_csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--x-col", default=None)
    ap.add_argument("--y-ref-col", default=None)
    ap.add_argument("--y-pred-col", default=None)
    args = ap.parse_args()
    plot_validation_overlay(args.reference_csv, args.prediction_csv, args.output, x_col=args.x_col, y_ref_col=args.y_ref_col, y_pred_col=args.y_pred_col)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
