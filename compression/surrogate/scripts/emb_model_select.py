#!/usr/bin/env python3

import argparse

from meso_uq.surrogate.cli import read_wide_curve_table
from meso_uq.surrogate.model_selection import grid_search_tabular_surrogate


DEFAULT_ARCHITECTURES = (
    "32x2,32x3,32x4,64x2,64x3,64x4,64x5,128x2,128x3,128x4,256x2,256x3"
)


def _parse_int_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_architecture_list(text: str) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    for token in text.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(
                f"Invalid architecture token '{token}'. Expected WIDTHxDEPTH, e.g. 64x3."
            )
        width_text, depth_text = item.split("x", 1)
        values.append((int(width_text), int(depth_text)))
    if not values:
        raise ValueError("architecture list must be non-empty")
    return values


def main():
    ap = argparse.ArgumentParser(description="Compression surrogate model selection over width/depth grid")
    ap.add_argument("data", help="Path to whitespace training table")
    ap.add_argument("--output-dir", default="trained/model_selection")
    ap.add_argument(
        "--architectures",
        default=DEFAULT_ARCHITECTURES,
        help=(
            "Comma-separated WIDTHxDEPTH list. "
            f"Default is the 12-architecture paper sweep: {DEFAULT_ARCHITECTURES}"
        ),
    )
    ap.add_argument("--widths", default="32,64,128", help="Legacy fallback when --architectures is empty.")
    ap.add_argument("--depths", default="2,3,4", help="Legacy fallback when --architectures is empty.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epoch", type=int, default=100)
    args = ap.parse_args()

    df = read_wide_curve_table(args.data, curve_axis_name="disp", value_name="F")
    result = grid_search_tabular_surrogate(
        df,
        input_cols=["Yt", "kb", "b1", "b2", "a3", "a4", "disp"],
        target_col="F",
        output_dir=args.output_dir,
        widths=_parse_int_list(args.widths),
        depths=_parse_int_list(args.depths),
        architectures=_parse_architecture_list(args.architectures) if args.architectures.strip() else None,
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
