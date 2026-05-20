#!/usr/bin/env python3
from __future__ import annotations

"""Write EMB 3.4um AL-vs-LHS validation artifacts from curve rows."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_al_vs_lhs_validation import (  # noqa: E402
    write_emb_34um_al_vs_lhs_validation_artifacts,
)


def _parse_prefix_counts(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute EMB 3.4um grouped-holdout AL-vs-LHS median curve relative L2 validation artifacts."
    )
    parser.add_argument(
        "--curve-rows",
        required=True,
        help="JSON or CSV rows with strategy plus curve metric inputs.",
    )
    parser.add_argument("--output-root", required=True, help="Directory for PNG, JSON, and CSV artifacts.")
    parser.add_argument(
        "--prefix-counts",
        default=None,
        help="Optional comma-separated curve prefix counts. Defaults to cumulative AL round counts.",
    )
    parser.add_argument("--acquisition-engine", default="", help="Name of the adaptive acquisition engine, if used.")
    parser.add_argument(
        "--adaptive-acquisition-available",
        action="store_true",
        help="Mark the manifest as using a real adaptive acquisition engine.",
    )
    parser.add_argument(
        "--runtime-rows",
        default=None,
        help="JSON or CSV with per-curve runtime evidence.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Write a fallback PNG without importing matplotlib.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = write_emb_34um_al_vs_lhs_validation_artifacts(
        curve_rows=Path(args.curve_rows),
        output_root=Path(args.output_root),
        prefix_curve_counts=_parse_prefix_counts(args.prefix_counts),
        adaptive_acquisition_available=bool(args.adaptive_acquisition_available),
        acquisition_engine_name=args.acquisition_engine,
        runtime_rows=Path(args.runtime_rows) if args.runtime_rows else None,
        include_plot=not bool(args.no_plot),
    )
    print(artifacts.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
