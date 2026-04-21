#!/usr/bin/env python3
"""Build paired DNN/BNN comparison table and emit a GO/NO-GO insertion decision.

Reads the holdout L2 summary CSV produced by generate_surrogate_holdout_l2_figure.py
and applies the explicit decision rule:

  GO  if:
        - all 6 paired runs completed (both families present for every diameter)
        - BNN is not worse than DNN by >10% median L2 on ANY dataset
        - BNN is better or equal on AT LEAST 4/6 datasets

  NO-GO otherwise.

Outputs
-------
  <output-dir>/surrogate_family_comparison.csv
  <output-dir>/bnn_insertion_decision.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"modality", "surrogate_family", "diameter_um", "metric_rel_l2_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV missing columns: {sorted(missing)}")
    return df


def _build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["modality", "diameter_um"],
        columns="surrogate_family",
        values="metric_rel_l2_pct",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    for col in ("dnn", "bnn"):
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot = pivot.rename(columns={"dnn": "dnn_median_rel_l2_pct", "bnn": "bnn_median_rel_l2_pct"})
    pivot["delta_abs"] = pivot["bnn_median_rel_l2_pct"] - pivot["dnn_median_rel_l2_pct"]
    pivot["delta_pct"] = (
        100.0 * pivot["delta_abs"] / pivot["dnn_median_rel_l2_pct"].replace(0, float("nan"))
    )
    pivot["winner"] = pivot.apply(
        lambda r: "bnn" if r["bnn_median_rel_l2_pct"] <= r["dnn_median_rel_l2_pct"] else "dnn",
        axis=1,
    )
    return pivot.sort_values(["modality", "diameter_um"]).reset_index(drop=True)


def _decision(comp: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []

    n_pairs = len(comp)
    n_complete = comp[["dnn_median_rel_l2_pct", "bnn_median_rel_l2_pct"]].notna().all(axis=1).sum()
    if n_complete < 6:
        reasons.append(f"Only {n_complete}/6 paired runs completed (need 6).")

    worst_delta = comp["delta_pct"].max()
    if worst_delta > 10.0:
        row = comp.loc[comp["delta_pct"].idxmax()]
        reasons.append(
            f"BNN exceeds DNN by >{10}% on {row['modality']} {row['diameter_um']}um "
            f"(delta={worst_delta:.1f}%)."
        )

    bnn_wins = int((comp["winner"] == "bnn").sum())
    if bnn_wins < 4:
        reasons.append(
            f"BNN is better or equal on only {bnn_wins}/6 datasets (need ≥4)."
        )

    verdict = "GO" if not reasons else "NO-GO"
    return verdict, reasons


def _write_decision_md(
    path: Path,
    comp: pd.DataFrame,
    verdict: str,
    reasons: list[str],
) -> None:
    lines = [
        "# BNN Insertion Decision",
        "",
        f"**Verdict: {verdict}**",
        "",
    ]
    if reasons:
        lines += ["## Reasons for NO-GO", ""]
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")
    lines += [
        "## Paired Comparison",
        "",
        comp.to_markdown(index=False, floatfmt=".3f"),
        "",
        "### Decision rule",
        "GO requires ALL of:",
        "1. All 6 paired runs completed.",
        "2. BNN not worse than DNN by >10% on any single dataset.",
        "3. BNN better or equal on ≥4/6 datasets.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        required=True,
        help="surrogate_holdout_l2_summary.csv from generate_surrogate_holdout_l2_figure.py",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_summary(input_csv)
    comp = _build_comparison(df)
    verdict, reasons = _decision(comp)

    comp_csv = output_dir / "surrogate_family_comparison.csv"
    decision_md = output_dir / "bnn_insertion_decision.md"

    comp.to_csv(comp_csv, index=False)
    _write_decision_md(decision_md, comp, verdict, reasons)

    print(f"Wrote comparison CSV: {comp_csv}")
    print(f"Wrote decision report: {decision_md}")
    print(f"BNN insertion decision: {verdict}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")

    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
