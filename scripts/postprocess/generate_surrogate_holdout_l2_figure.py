#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from meso_uq.postprocess.paper_figures import configure_matplotlib


def _parse_metric(summary: dict[str, object], metric: str) -> float:
    key_map = {
        "mean": "best_mean_curve_rel_l2_pct",
        "median": "best_median_curve_rel_l2_pct",
        "max": "best_max_curve_rel_l2_pct",
    }
    key = key_map[metric]
    value = summary.get(key)
    if value is None:
        raise KeyError(f"Missing metric key {key!r} in summary payload.")
    return float(value)


def collect_holdout_summary_rows(input_root: Path, metric: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(input_root.glob("**/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        modality = str(payload.get("modality", "")).strip().lower()
        family = str(payload.get("surrogate_family", "")).strip().lower()
        diameter = str(payload.get("diameter_um", "")).strip()
        if modality not in {"compression", "indentation"}:
            continue
        if family not in {"dnn", "bnn"}:
            continue
        if not diameter:
            continue
        rows.append(
            {
                "modality": modality,
                "surrogate_family": family,
                "diameter_um": diameter,
                "metric_rel_l2_pct": _parse_metric(payload, metric),
                "summary_path": str(summary_path),
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No holdout summary rows discovered under {input_root}. "
            "Expected files named summary.json with modality/surrogate_family keys."
        )
    df = pd.DataFrame(rows)
    df["diameter_um_f"] = df["diameter_um"].astype(float)
    return df.sort_values(["modality", "diameter_um_f", "surrogate_family"]).reset_index(drop=True)


def plot_holdout_l2(df: pd.DataFrame, *, metric: str, output_png: Path, output_pdf: Path) -> None:
    configure_matplotlib()
    families = ["dnn", "bnn"]
    family_colors = {"dnn": "#2a7fc0", "bnn": "#c07b00"}
    family_labels = {"dnn": "DNN", "bnn": "BNN"}
    metric_label = {
        "mean": "Mean Curve Relative L2 Error (%)",
        "median": "Median Curve Relative L2 Error (%)",
        "max": "Max Curve Relative L2 Error (%)",
    }[metric]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    modalities = ["compression", "indentation"]
    for ax, modality in zip(axes, modalities):
        sub = df[df["modality"] == modality].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        diameters = sorted(sub["diameter_um_f"].unique())
        labels = [f"{diam:.1f}um" for diam in diameters]
        x = list(range(len(diameters)))
        width = 0.36
        for idx, family in enumerate(families):
            fam_sub = sub[sub["surrogate_family"] == family]
            y_values = []
            for diam in diameters:
                row = fam_sub[fam_sub["diameter_um_f"] == diam]
                y_values.append(float(row["metric_rel_l2_pct"].iloc[0]) if not row.empty else float("nan"))
            offset = -0.5 * width if idx == 0 else 0.5 * width
            ax.bar(
                [val + offset for val in x],
                y_values,
                width=width,
                label=family_labels[family],
                color=family_colors[family],
                alpha=0.9,
            )
        ax.set_title(modality.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(metric_label)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-facing DNN vs BNN held-out relative L2 comparison figure."
    )
    parser.add_argument("--input-root", required=True, help="Root directory containing grouped holdout summary.json files.")
    parser.add_argument("--output-dir", default=None, help="Output directory for figure and summary csv.")
    parser.add_argument("--metric", choices=["mean", "median", "max"], default="median")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else input_root / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_holdout_summary_rows(input_root, args.metric)
    summary_csv = output_dir / "surrogate_holdout_l2_summary.csv"
    output_png = output_dir / "surrogate_holdout_l2_comparison.png"
    output_pdf = output_dir / "surrogate_holdout_l2_comparison.pdf"
    df.to_csv(summary_csv, index=False)
    plot_holdout_l2(df, metric=args.metric, output_png=output_png, output_pdf=output_pdf)

    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote figure PNG: {output_png}")
    print(f"Wrote figure PDF: {output_pdf}")


if __name__ == "__main__":
    main()

