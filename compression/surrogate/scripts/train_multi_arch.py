#!/usr/bin/env python3
"""Multi-architecture training script for the compression force surrogate.

This is the paper-facing counterpart to the indentation multi-architecture
trainer: run the 12-architecture sweep, write diagnostics, and promote the
best model into the canonical `*_BEST.pkl` artifact consumed by inference.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from meso_uq.surrogate.cli import read_wide_curve_table, train_tabular_surrogate

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SURROGATE_ROOT = PROJECT_ROOT / "compression" / "surrogate"


def get_architectures() -> list[tuple[int, int, str]]:
    return [
        (32, 2, "w32_d2"),
        (32, 3, "w32_d3"),
        (32, 4, "w32_d4"),
        (64, 2, "w64_d2"),
        (64, 3, "w64_d3"),
        (64, 4, "w64_d4"),
        (64, 5, "w64_d5"),
        (128, 2, "w128_d2"),
        (128, 3, "w128_d3"),
        (128, 4, "w128_d4"),
        (256, 2, "w256_d2"),
        (256, 3, "w256_d3"),
    ]


def read_new_dat_to_long(path: str | os.PathLike[str]):
    return read_wide_curve_table(path, curve_axis_name="disp", value_name="F")


def train_single_architecture(
    arch_config,
    data_path,
    output_dir,
    *,
    batch_size: int = 128,
    lr: float = 5e-4,
    max_epoch: int = 100,
):
    width, depth, name = arch_config
    print(f"\n{'='*70}\nTraining architecture: {name} (width={width}, depth={depth})\n{'='*70}")
    df_long = read_new_dat_to_long(data_path)
    out_path = os.path.join(output_dir, f"{name}.pkl")
    result = train_tabular_surrogate(
        df_long,
        input_cols=["Yt", "kb", "b1", "b2", "a3", "a4", "disp"],
        target_col="F",
        out_path=out_path,
        width=width,
        depth=depth,
        batch_size=batch_size,
        lr=lr,
        max_epoch=max_epoch,
    )
    return {
        "name": name,
        "width": int(width),
        "depth": int(depth),
        "train_loss": float(result["train_loss"]),
        "val_loss": float(result["val_loss"]),
        "model_path": out_path,
    }


def plot_results_summary(results, output_path, diameter):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(f"Compression Surrogate Training Summary - {diameter}um", fontsize=16, fontweight="bold")
    names = [r["name"] for r in results]
    train_losses = [r["train_loss"] for r in results]
    val_losses = [r["val_loss"] for r in results]

    ax = axes[0]
    ax.bar(range(len(names)), train_losses)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Training loss")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(names)), val_losses)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Validation loss")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _resolve_data_path(data_diameter, data_file):
    data_dir = SURROGATE_ROOT / f"diameters/{data_diameter}um/data"
    if data_file:
        return data_dir / data_file
    for candidate in ["F_Delta.dat", "samples_all.dat"]:
        path = data_dir / candidate
        if path.exists():
            return path
    return data_dir / "F_Delta.dat"


def _write_summary_report(path: str | os.PathLike[str], payload: dict[str, object]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finalize_results(
    *,
    diameter: str,
    output_dir: str | os.PathLike[str],
    results: list[dict[str, object]],
    plots_dir: str | os.PathLike[str],
    report_json: str | os.PathLike[str] | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    if not results:
        raise ValueError("No architecture results were provided for finalization.")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    results_sorted = sorted(results, key=lambda x: float(x["val_loss"]))
    training_results_path = output_dir_path / "training_results.csv"
    pd.DataFrame(results_sorted).to_csv(training_results_path, index=False)

    plots_dir_path = Path(plots_dir)
    plots_dir_path.mkdir(parents=True, exist_ok=True)
    effective_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_plot_path = plots_dir_path / f"training_summary_{effective_timestamp}.png"
    plot_results_summary(results_sorted, str(summary_plot_path), diameter)

    best = results_sorted[0]
    best_dest = SURROGATE_ROOT / f"diameters/{diameter}um/trained/microbubble_force_BEST.pkl"
    best_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best["model_path"], best_dest)
    print(f"Best model: {best['name']} -> {best_dest}")

    payload = {
        "diameter": diameter,
        "output_dir": str(output_dir_path),
        "best": best,
        "training_results_csv": str(training_results_path),
        "summary_plot": str(summary_plot_path),
        "best_dest": str(best_dest),
        "results": results_sorted,
    }
    if report_json:
        _write_summary_report(report_json, payload)
    return payload


def main():
    ap = argparse.ArgumentParser(description="Train multiple NN architectures for the compression surrogate")
    ap.add_argument("--diameter", type=str, required=True)
    ap.add_argument("--data-diameter", type=str, default=None)
    ap.add_argument("--data-file", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epoch", type=int, default=100)
    ap.add_argument("--arch-index", type=int, default=None)
    ap.add_argument("--arch-name", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--result-json", type=str, default=None)
    ap.add_argument("--report-json", type=str, default=None)
    ap.add_argument("--collect-only", action="store_true", default=False)
    args = ap.parse_args()

    data_diameter = args.data_diameter if args.data_diameter else args.diameter
    data_path = _resolve_data_path(data_diameter, args.data_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(SURROGATE_ROOT / f"diameters/{args.diameter}um/trained/multi_arch_{timestamp}")
    plots_dir = SURROGATE_ROOT / f"diameters/{args.diameter}um/plots"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    output_dir_path = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.collect_only:
        result_files = sorted(output_dir_path.glob("result_*.json"))
        if not result_files:
            raise FileNotFoundError(f"No per-architecture result JSONs found in {output_dir_path}")
        results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
        payload = finalize_results(
            diameter=args.diameter,
            output_dir=output_dir_path,
            results=results,
            plots_dir=plots_dir,
            report_json=args.report_json,
            timestamp=timestamp,
        )
        payload["data_path"] = str(data_path)
        if args.report_json:
            _write_summary_report(args.report_json, payload)
        return

    architectures = get_architectures()
    if args.arch_index is not None or args.arch_name is not None:
        if args.arch_index is not None and args.arch_name is not None:
            raise ValueError("Use only one of --arch-index or --arch-name")
        if args.arch_index is not None:
            arch = architectures[args.arch_index]
        else:
            matches = [a for a in architectures if a[2] == args.arch_name]
            if not matches:
                raise ValueError(f"Unknown arch-name '{args.arch_name}'")
            arch = matches[0]
        result = train_single_architecture(
            arch,
            str(data_path),
            output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            max_epoch=args.max_epoch,
        )
        result_path = args.result_json or os.path.join(output_dir, f"{arch[2]}_result.json")
        _write_summary_report(result_path, result)
        print(f"Saved single-arch result to: {result_path}")
        return

    results = []
    for arch in architectures:
        results.append(
            train_single_architecture(
                arch,
                str(data_path),
                output_dir,
                batch_size=args.batch_size,
                lr=args.lr,
                max_epoch=args.max_epoch,
            )
        )
    payload = finalize_results(
        diameter=args.diameter,
        output_dir=output_dir_path,
        results=results,
        plots_dir=plots_dir,
        report_json=args.report_json,
        timestamp=timestamp,
    )
    payload["data_path"] = str(data_path)
    if args.report_json:
        _write_summary_report(args.report_json, payload)


if __name__ == "__main__":
    main()
