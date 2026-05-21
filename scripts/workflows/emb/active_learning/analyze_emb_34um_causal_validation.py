#!/usr/bin/env python3
from __future__ import annotations

"""Write EMB 3.4um causal AL-vs-LHS validation artifacts."""

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

from meso_uq.active_learning.emb_34um_causal_validation_report import (  # noqa: E402
    write_emb_34um_causal_validation_artifacts,
)
from meso_uq.active_learning.emb_34um_causal_validation_ingestion import (  # noqa: E402
    EMB_34UM_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build EMB 3.4um causal AL-vs-LHS causal validation artifacts.")
    parser.add_argument(
        "--rows",
        default=None,
        help="JSON/CSV with precomputed validation rows. For campaign analysis, prefer --ingestion-manifest.",
    )
    parser.add_argument(
        "--ingestion-manifest",
        default=None,
        help="Ingestion manifest/report JSON with completed curve records.",
    )
    parser.add_argument(
        "--campaign-root",
        default=None,
        help="Campaign root containing an ingest/ingest manifest.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for report, CSV, and plots. Defaults to <campaign-root>/analyze when provided.",
    )
    parser.add_argument("--runtime-rows", default=None, help="Optional JSON/CSV for per-row runtime/replacement inputs.")
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
    if args.ingestion_manifest:
        source_rows_path = Path(args.ingestion_manifest)
    elif args.campaign_root:
        source_rows_path = Path(args.campaign_root) / "ingest" / EMB_34UM_CAUSAL_VALIDATION_INGESTION_MANIFEST_FILENAME
    elif args.rows:
        source_rows_path = Path(args.rows)
    else:
        raise ValueError("Either --rows, --ingestion-manifest, or --campaign-root must be provided.")

    output_root = Path(args.output_root) if args.output_root else Path(args.campaign_root or ".") / "analyze"
    command = " ".join(shlex.quote(item) for item in [Path(__file__).name, *raw_args])
    artifacts = write_emb_34um_causal_validation_artifacts(
        rows=source_rows_path,
        output_root=output_root,
        runtime_rows=Path(args.runtime_rows) if args.runtime_rows else None,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
        include_plot=not args.no_plots,
        generation_command=command,
    )

    print(artifacts.report_path)
    print(f"status={artifacts.report['decision']['status']}")
    if artifacts.report["decision"]["passed"] or args.allow_blocked:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
