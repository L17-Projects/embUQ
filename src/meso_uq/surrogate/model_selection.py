from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .cli import train_tabular_surrogate


def _as_int_list(values: Sequence[int] | Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def grid_search_tabular_surrogate(
    df,
    *,
    input_cols,
    target_col,
    output_dir,
    widths: Sequence[int],
    depths: Sequence[int],
    batch_size: int = 128,
    lr: float = 5e-4,
    max_epoch: int = 100,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    widths = _as_int_list(widths)
    depths = _as_int_list(depths)
    if not widths or not depths:
        raise ValueError("widths and depths must be non-empty")

    rows = []
    best = None

    for width, depth in product(widths, depths):
        model_path = output_dir / f"model_w{width}_d{depth}.pkl"
        result = train_tabular_surrogate(
            df,
            input_cols=input_cols,
            target_col=target_col,
            out_path=str(model_path),
            width=width,
            depth=depth,
            batch_size=batch_size,
            lr=lr,
            max_epoch=max_epoch,
        )
        row = {
            "width": width,
            "depth": depth,
            "train_loss": float(result["train_loss"]),
            "val_loss": float(result["val_loss"]),
            "model_path": str(model_path),
        }
        rows.append(row)
        if best is None or row["val_loss"] < best["val_loss"]:
            best = row

    leaderboard = pd.DataFrame(rows).sort_values(["val_loss", "train_loss", "width", "depth"]).reset_index(drop=True)
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    best_path = output_dir / "best_model.json"
    with open(best_path, "w") as handle:
        json.dump(best, handle, indent=2)

    return {
        "best": best,
        "leaderboard": leaderboard,
        "leaderboard_path": str(leaderboard_path),
        "best_path": str(best_path),
    }
