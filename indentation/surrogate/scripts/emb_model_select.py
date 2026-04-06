#!/usr/bin/env python3

import argparse

from meso_uq.surrogate.cli import read_wide_curve_table
from meso_uq.surrogate.model_selection import grid_search_tabular_surrogate


def _parse_int_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Indentation surrogate model selection over width/depth grid")
    ap.add_argument("data", help="Path to whitespace training table")
    ap.add_argument("--output-dir", default="trained/model_selection")
    ap.add_argument("--widths", default="32,64,128")
    ap.add_argument("--depths", default="2,3,4")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epoch", type=int, default=100)
    args = ap.parse_args()

    df = read_wide_curve_table(args.data, curve_axis_name="F", value_name="disp")
    result = grid_search_tabular_surrogate(
        df,
        input_cols=["Yt", "kb", "b1", "b2", "a3", "a4", "F"],
        target_col="disp",
        output_dir=args.output_dir,
        widths=_parse_int_list(args.widths),
        depths=_parse_int_list(args.depths),
        batch_size=args.batch_size,
        lr=args.lr,
        max_epoch=args.max_epoch,
    )
    best = result["best"]
    print(
        f"Best model -> width={best['width']}, depth={best['depth']}, "
        f"valid={best['val_loss']:.3e}, train={best['train_loss']:.3e}"
    )
    print(f"Leaderboard -> {result['leaderboard_path']}")
    print(f"Best model metadata -> {result['best_path']}")


if __name__ == "__main__":
    main()
