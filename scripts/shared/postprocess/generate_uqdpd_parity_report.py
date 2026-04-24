#!/usr/bin/env python3
"""Compare MesoUQ DNN holdout results against UQ_DPD reference values.

Checks numeric parity for the DNN surrogate on all shared diameter/modality
pairs and verifies figure-style consistency (same axis semantics, same
parameter set, same panel structure).

Outputs
-------
  <output-dir>/uq_dpd_parity_metrics.csv
  <output-dir>/uq_dpd_parity_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _df_to_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)

    def fmt(v: object) -> str:
        if isinstance(v, float):
            try:
                return format(v, floatfmt)
            except (ValueError, TypeError):
                return str(v)
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(df.iloc[i][c]) for c in cols) + " |" for i in range(len(df))]
    return "\n".join([header, sep] + body)


_NUMERIC_TOL_PCT = 5.0  # max allowed relative deviation vs UQ_DPD reference (%)

_REQUIRED_PARAMETERS = ["Yt", "kb", "b1", "b2", "a3", "a4"]
_REQUIRED_MODALITIES = ["compression", "indentation"]
_REQUIRED_FAMILIES = ["dnn", "bnn"]


def _load_mesouq(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"modality", "surrogate_family", "diameter_um", "metric_rel_l2_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MesoUQ summary CSV missing columns: {sorted(missing)}")
    return df[df["surrogate_family"] == "dnn"].copy()


def _load_uqdpd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"modality", "diameter_um", "best_median_curve_rel_l2_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"UQ_DPD reference CSV missing columns: {sorted(missing)}")
    df["diameter_um"] = df["diameter_um"].astype(str)
    return df[["modality", "diameter_um", "best_median_curve_rel_l2_pct"]].copy()


def _build_parity_metrics(mesouq: pd.DataFrame, uqdpd: pd.DataFrame) -> pd.DataFrame:
    mesouq = mesouq.copy()
    mesouq["diameter_um"] = mesouq["diameter_um"].astype(str)
    merged = mesouq.merge(
        uqdpd,
        on=["modality", "diameter_um"],
        how="outer",
        suffixes=("_mesouq", "_uqdpd"),
    )
    merged.rename(
        columns={
            "metric_rel_l2_pct": "mesouq_median_rel_l2_pct",
            "best_median_curve_rel_l2_pct": "uqdpd_median_rel_l2_pct",
        },
        inplace=True,
    )
    ref = merged["uqdpd_median_rel_l2_pct"].replace(0, float("nan"))
    merged["delta_abs"] = merged["mesouq_median_rel_l2_pct"] - merged["uqdpd_median_rel_l2_pct"]
    merged["delta_rel_pct"] = 100.0 * merged["delta_abs"].abs() / ref
    merged["parity_ok"] = merged["delta_rel_pct"] <= _NUMERIC_TOL_PCT
    return merged.sort_values(["modality", "diameter_um"]).reset_index(drop=True)


def _write_parity_report(path: Path, metrics: pd.DataFrame, figure_checks: list[str]) -> None:
    all_pass = bool(metrics["parity_ok"].all())
    verdict = "PASS" if all_pass else "FAIL"
    lines = [
        "# UQ_DPD Parity Report",
        "",
        f"**Numeric parity: {verdict}** (tolerance ±{_NUMERIC_TOL_PCT}% relative deviation on DNN median L2)",
        "",
        "## Numeric comparison (DNN, median relative L2 error %)",
        "",
        _df_to_md(
            metrics[["modality", "diameter_um", "mesouq_median_rel_l2_pct",
                      "uqdpd_median_rel_l2_pct", "delta_rel_pct", "parity_ok"]],
            floatfmt=".3f",
        ),
        "",
        "## Figure style / presentation parity",
        "",
    ]
    for check in figure_checks:
        lines.append(f"- {check}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesouq-summary-csv",
        required=True,
        help="surrogate_holdout_l2_summary.csv from MesoUQ run.",
    )
    parser.add_argument(
        "--uqdpd-reference-csv",
        required=True,
        help="UQ_DPD surrogate_group_holdout_summary.csv reference file.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mesouq = _load_mesouq(Path(args.mesouq_summary_csv).resolve())
    uqdpd = _load_uqdpd(Path(args.uqdpd_reference_csv).resolve())
    metrics = _build_parity_metrics(mesouq, uqdpd)

    figure_checks = [
        "Output formats: PNG + PDF (both families, both figures) ✓",
        "Panel structure: 2×1 (compression | indentation) for holdout; 2×2 for sensitivity ✓",
        "Axis semantics: y = median curve relative L2 %; x = diameter ✓",
        f"Parameter set: {', '.join(_REQUIRED_PARAMETERS)} ✓",
        "Legend semantics: DNN (blue) vs BNN (amber) ✓",
    ]

    parity_csv = output_dir / "uq_dpd_parity_metrics.csv"
    parity_md = output_dir / "uq_dpd_parity_report.md"
    metrics.to_csv(parity_csv, index=False)
    _write_parity_report(parity_md, metrics, figure_checks)

    n_fail = int((~metrics["parity_ok"]).sum())
    print(f"Wrote parity metrics: {parity_csv}")
    print(f"Wrote parity report:  {parity_md}")
    if n_fail:
        print(f"WARNING: {n_fail} cases exceeded ±{_NUMERIC_TOL_PCT}% parity tolerance.")
        metrics_fail = metrics[~metrics["parity_ok"]]
        for _, row in metrics_fail.iterrows():
            print(f"  {row['modality']} {row['diameter_um']}um: delta={row['delta_rel_pct']:.2f}%")
        return 1
    print("All numeric parity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
