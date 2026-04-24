#!/usr/bin/env python3
"""Collect all stage outputs and write final_report.md.

Reads artefacts from prior pipeline stages and emits a single
concise report with pass/fail status per stage, the BNN insertion
recommendation, and paths to final figures and CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _check(path: Path) -> str:
    return "PASS" if path.exists() else "FAIL (missing)"


def _read_verdict(md_path: Path, keyword: str) -> str:
    if not md_path.exists():
        return "UNKNOWN (file missing)"
    text = md_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if keyword in line:
            return line.strip().lstrip("#").strip()
    return "UNKNOWN"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Root directory for this run.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_root / "comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate stage artifacts
    bnn_report = run_root / "bnn_training" / "bnn_training_matrix_report.json"
    holdout_csv = run_root / "group_holdout" / "surrogate_group_holdout_report.csv"
    holdout_l2_csv = run_root / "figures" / "holdout_l2" / "surrogate_holdout_l2_summary.csv"
    holdout_png = run_root / "figures" / "holdout_l2" / "surrogate_holdout_l2_comparison.png"
    holdout_pdf = run_root / "figures" / "holdout_l2" / "surrogate_holdout_l2_comparison.pdf"
    sobol_csv = run_root / "figures" / "sensitivity" / "surrogate_sensitivity_summary.csv"
    sobol_png = run_root / "figures" / "sensitivity" / "surrogate_sensitivity_comparison.png"
    sobol_pdf = run_root / "figures" / "sensitivity" / "surrogate_sensitivity_comparison.pdf"
    comparison_csv = output_dir / "surrogate_family_comparison.csv"
    decision_md = output_dir / "bnn_insertion_decision.md"
    parity_csv = output_dir / "uq_dpd_parity_metrics.csv"
    parity_md = output_dir / "uq_dpd_parity_report.md"

    bnn_status = "PASS" if bnn_report.exists() else "FAIL (missing)"
    if bnn_report.exists():
        try:
            payload = json.loads(bnn_report.read_text(encoding="utf-8"))
            bnn_status = payload.get("status", "unknown").upper()
        except Exception:
            bnn_status = "FAIL (unreadable)"

    bnn_verdict = _read_verdict(decision_md, "Verdict")
    parity_verdict = _read_verdict(parity_md, "Numeric parity")

    lines = [
        "# Final Run Report",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Stage pass/fail",
        "",
        f"| Stage | Status |",
        f"|---|---|",
        f"| BNN training (indentation) | {bnn_status} |",
        f"| Group holdout (DNN+BNN, all 6 cases) | {_check(holdout_csv)} |",
        f"| Holdout L2 figure | {_check(holdout_png)} |",
        f"| Sobol sensitivity figure | {_check(sobol_png)} |",
        f"| Comparison table | {_check(comparison_csv)} |",
        f"| UQ_DPD parity | {_check(parity_csv)} |",
        "",
        "## BNN insertion recommendation",
        "",
        f"**{bnn_verdict}**",
        "",
        "## UQ_DPD parity",
        "",
        f"**{parity_verdict}**",
        "",
        "## Output paths",
        "",
        f"| Artefact | Path |",
        f"|---|---|",
        f"| Holdout L2 summary CSV | `{holdout_l2_csv}` |",
        f"| Holdout L2 figure PNG | `{holdout_png}` |",
        f"| Holdout L2 figure PDF | `{holdout_pdf}` |",
        f"| Sensitivity summary CSV | `{sobol_csv}` |",
        f"| Sensitivity figure PNG | `{sobol_png}` |",
        f"| Sensitivity figure PDF | `{sobol_pdf}` |",
        f"| Family comparison CSV | `{comparison_csv}` |",
        f"| BNN insertion decision | `{decision_md}` |",
        f"| UQ_DPD parity metrics CSV | `{parity_csv}` |",
        f"| UQ_DPD parity report | `{parity_md}` |",
    ]

    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote final report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
