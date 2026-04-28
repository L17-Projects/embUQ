#!/usr/bin/env python3

import argparse

from meso_uq.surrogate.cli import read_indentation_table, train_tabular_surrogate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="trained/microbubble_disp_BEST.pkl")
    ap.add_argument("--report-path", default=None)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epoch", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--disp-source", type=str, default="auto", choices=["auto", "diameter", "displacement"])
    ap.add_argument("--rupture-ratio", type=float, default=2.0)
    args = ap.parse_args()
    rupture_ratio = None if args.rupture_ratio <= 0 else float(args.rupture_ratio)
    df = read_indentation_table(args.data, disp_source=args.disp_source, rupture_ratio_threshold=rupture_ratio)
    result = train_tabular_surrogate(
        df,
        input_cols=["Yt", "kb", "b1", "b2", "a3", "a4", "F"],
        target_col="disp",
        out_path=args.out,
        width=args.width,
        depth=args.depth,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epoch=args.max_epoch,
        seed=args.seed,
        report_path=args.report_path,
    )
    print(f"Saved -> {result['out']}. Final train={result['train_loss']:.3e}, valid={result['val_loss']:.3e}")


if __name__ == "__main__":
    main()
