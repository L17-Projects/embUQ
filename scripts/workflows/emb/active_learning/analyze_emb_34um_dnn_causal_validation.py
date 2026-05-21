#!/usr/bin/env python3
from __future__ import annotations

"""Write EMB 3.4um DNN causal AL-vs-LHS validation artifacts."""

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_report import (  # noqa: E402
    write_emb_34um_dnn_causal_validation_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build EMB 3.4um DNN causal AL-vs-LHS validation artifacts.")
    parser.add_argument("--rows", required=True, help="JSON/CSV file with validation rows.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for report, CSV, and plots. Defaults to <rows-parent>/analyze.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026202405)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--no-plots", action="store_true", help="Write fallback PNGs without matplotlib.")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit 0 even when the validation status is not passed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    rows_path = Path(args.rows)
    output_root = Path(args.output_root) if args.output_root else rows_path.parent / "analyze"
    command = " ".join(shlex.quote(item) for item in [Path(__file__).name, *raw_args])
    artifacts = write_emb_34um_dnn_causal_validation_artifacts(
        rows=rows_path,
        output_root=output_root,
        metadata={"ensemble_size": 10},
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
        include_plot=not args.no_plots,
        generation_command=command,
    )

    print(artifacts.report_path)
    print(f"status={artifacts.report['status']}")
    if artifacts.report["decision"]["passed"] or args.allow_blocked:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
