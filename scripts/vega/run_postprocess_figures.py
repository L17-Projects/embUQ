#!/usr/bin/env python3
"""Run all postprocessing steps for a completed surrogate validation run.

Steps:
  1. Holdout L2 figure
  2. Sensitivity figure
  3. Family comparison + BNN insertion decision
  4. UQ_DPD parity report
  5. Final report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTPROCESS = REPO_ROOT / "scripts" / "postprocess"

UQDPD_REFERENCE_CSV = (
    REPO_ROOT.parent
    / "UQ_DPD"
    / "Hierarchical_UQ_compression_dev"
    / "_paper"
    / "v4"
    / "generated"
    / "figures"
    / "surrogate_group_holdout_summary.csv"
)


def _run(cmd: list[str], label: str) -> int:
    print(f"\n[postprocess] {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  FAILED (returncode={result.returncode})", file=sys.stderr)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Run root created by the pipeline.")
    parser.add_argument("--metric", default="median", choices=["mean", "median", "max"])
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--uqdpd-reference-csv",
        default=str(UQDPD_REFERENCE_CSV),
        help="Path to UQ_DPD surrogate_group_holdout_summary.csv reference.",
    )
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve()
    py = args.python_bin

    holdout_input = run_root / "group_holdout"
    sobol_input = run_root / "sobol"
    figures_holdout = run_root / "figures" / "holdout_l2"
    figures_sobol = run_root / "figures" / "sensitivity"
    comparison_dir = run_root / "comparison"
    holdout_l2_csv = figures_holdout / "surrogate_holdout_l2_summary.csv"

    failures = 0

    failures += _run(
        [py, str(POSTPROCESS / "generate_surrogate_holdout_l2_figure.py"),
         "--input-root", str(holdout_input),
         "--output-dir", str(figures_holdout),
         "--metric", args.metric],
        "Holdout L2 figure",
    )

    failures += _run(
        [py, str(POSTPROCESS / "generate_surrogate_sensitivity_figure.py"),
         "--input-root", str(sobol_input),
         "--output-dir", str(figures_sobol),
         "--index-type", "ST",
         "--parameters", "Yt", "kb", "b1", "b2", "a3", "a4"],
        "Sensitivity figure",
    )

    failures += _run(
        [py, str(POSTPROCESS / "generate_surrogate_comparison.py"),
         "--input-csv", str(holdout_l2_csv),
         "--output-dir", str(comparison_dir)],
        "Family comparison + BNN decision",
    )

    failures += _run(
        [py, str(POSTPROCESS / "generate_uqdpd_parity_report.py"),
         "--mesouq-summary-csv", str(holdout_l2_csv),
         "--uqdpd-reference-csv", args.uqdpd_reference_csv,
         "--output-dir", str(comparison_dir)],
        "UQ_DPD parity report",
    )

    failures += _run(
        [py, str(POSTPROCESS / "generate_final_report.py"),
         "--run-root", str(run_root),
         "--output-dir", str(comparison_dir)],
        "Final report",
    )

    if failures:
        print(f"\n{failures} postprocessing step(s) failed.", file=sys.stderr)
        return 1
    print("\nAll postprocessing steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
