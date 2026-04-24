#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from meso_uq.postprocess.paper_figures import configure_matplotlib

_DIAM_RE = re.compile(r"(?P<diam>\d+(?:\.\d+)?)um", re.IGNORECASE)


def _infer_modality(path: Path) -> str | None:
    parts = [part.lower() for part in path.parts]
    if "compression" in parts:
        return "compression"
    if "indentation" in parts:
        return "indentation"
    return None


def _infer_family(path: Path) -> str:
    stem = path.stem.lower()
    parts = [part.lower() for part in path.parts]
    if "bnn" in stem or "bnn" in parts:
        return "bnn"
    if "dnn" in stem or "dnn" in parts:
        return "dnn"
    return "dnn"


def _infer_diameter(path: Path) -> str | None:
    for candidate in [path.stem, *path.parts]:
        match = _DIAM_RE.search(str(candidate))
        if match is not None:
            return match.group("diam")
    return None


def collect_sobol_rows(input_root: Path, *, index_type: str, parameters: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for csv_path in sorted(input_root.glob("**/*.csv")):
        name = csv_path.name.lower()
        if "sobol" not in name:
            continue
        modality = _infer_modality(csv_path)
        if modality is None:
            continue
        family = _infer_family(csv_path)
        diameter = _infer_diameter(csv_path)
        if diameter is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        required = {"axis", "parameter", "index_type", "value"}
        if not required.issubset(set(df.columns)):
            continue
        sub = df[(df["index_type"] == index_type) & (df["parameter"].isin(parameters))].copy()
        if sub.empty:
            continue
        sub["modality"] = modality
        sub["surrogate_family"] = family
        sub["diameter_um"] = float(diameter)
        sub["source_path"] = str(csv_path)
        rows.append(sub)
    if not rows:
        raise FileNotFoundError(
            f"No Sobol rows discovered under {input_root} with index_type={index_type!r}."
        )
    all_rows = pd.concat(rows, ignore_index=True)
    grouped = (
        all_rows.groupby(["modality", "surrogate_family", "parameter", "axis"], as_index=False)[
            "value"
        ]
        .mean()
        .rename(columns={"value": "mean_value"})
    )
    return grouped


def plot_sensitivity_panels(
    df: pd.DataFrame,
    *,
    index_type: str,
    parameters: list[str],
    output_png: Path,
    output_pdf: Path,
) -> None:
    configure_matplotlib()
    parameter_colors = {
        "Yt": "#1f77b4",
        "kb": "#ff7f0e",
        "b1": "#2ca02c",
        "b2": "#d62728",
        "a3": "#9467bd",
        "a4": "#8c564b",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharey=True)
    modalities = ["compression", "indentation"]
    families = ["dnn", "bnn"]
    for row_idx, modality in enumerate(modalities):
        for col_idx, family in enumerate(families):
            ax = axes[row_idx, col_idx]
            sub = df[(df["modality"] == modality) & (df["surrogate_family"] == family)]
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                continue
            for param in parameters:
                series = sub[sub["parameter"] == param].sort_values("axis")
                if series.empty:
                    continue
                ax.plot(
                    series["axis"].to_numpy(float),
                    series["mean_value"].to_numpy(float),
                    label=param,
                    color=parameter_colors.get(param, None),
                )
            if row_idx == 1:
                ax.set_xlabel("Axis")
            if col_idx == 0:
                ax.set_ylabel(f"{index_type} index")
            ax.set_title(f"{modality.capitalize()} / {family.upper()}")
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-facing DNN vs BNN Sobol sensitivity comparison figure."
    )
    parser.add_argument("--input-root", required=True, help="Root directory containing Sobol CSV files.")
    parser.add_argument("--output-dir", default=None, help="Output directory for figure and summary csv.")
    parser.add_argument("--index-type", default="ST", choices=["S1", "ST"])
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=["Yt", "kb", "b1", "b2", "a3", "a4"],
        help="Parameter names to include in the figure.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else input_root / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_sobol_rows(
        input_root,
        index_type=args.index_type,
        parameters=list(args.parameters),
    )
    summary_csv = output_dir / "surrogate_sensitivity_summary.csv"
    output_png = output_dir / "surrogate_sensitivity_comparison.png"
    output_pdf = output_dir / "surrogate_sensitivity_comparison.pdf"
    df.to_csv(summary_csv, index=False)
    plot_sensitivity_panels(
        df,
        index_type=args.index_type,
        parameters=list(args.parameters),
        output_png=output_png,
        output_pdf=output_pdf,
    )

    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote figure PNG: {output_png}")
    print(f"Wrote figure PDF: {output_pdf}")


if __name__ == "__main__":
    main()

