#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meso_uq.active_learning.emb_34um_final_gate_ingestion import (  # noqa: E402
    write_emb_34um_final_gate_ingestion_artifacts,
)


DEFAULT_CAMPAIGN_MANIFEST = Path(
    "/scratch/project/eu-26-17/eubrieucb/mesouq/runs/active_learning/emb_indentation_3p4_final_gate/"
    "20260519T181300Z/emb_34um_final_gate_campaign_manifest.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest EMB 3.4um final-gate DPD F_Delta.dat outputs and write a gate report."
    )
    parser.add_argument(
        "--campaign-manifest",
        default=str(DEFAULT_CAMPAIGN_MANIFEST),
        help="Path to emb_34um_final_gate_campaign_manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory for ingestion artifacts. Defaults to <campaign_root>/ingestion_report.",
    )
    parser.add_argument(
        "--f-delta-file",
        action="append",
        default=[],
        help="Additional F_Delta.dat file to search for candidate rows. May be passed multiple times.",
    )
    parser.add_argument("--expected-full-count", type=int, default=90)
    parser.add_argument("--expected-canary-count", type=int, default=1)
    parser.add_argument("--retry-limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Write fallback PNGs without importing matplotlib.")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Return exit code 0 even when outputs are missing or final pass/fail is blocked.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = write_emb_34um_final_gate_ingestion_artifacts(
        campaign_manifest_path=Path(args.campaign_manifest),
        output_root=Path(args.output_root) if args.output_root else None,
        extra_f_delta_files=[Path(item) for item in args.f_delta_file],
        expected_full_count=args.expected_full_count,
        expected_canary_count=args.expected_canary_count,
        retry_limit=args.retry_limit,
        include_plot=not args.no_plots,
    )
    print(f"ingestion_report={artifacts.report_path}")
    print(f"ingestion_manifest={artifacts.manifest_path}")
    print(f"summary_csv={artifacts.summary_csv_path}")
    print(f"validation_plot={artifacts.plot_path}")
    print(f"quarantine_manifest={artifacts.quarantine_path}")
    print(f"status={artifacts.report['status']}")
    if artifacts.report["passed"] or args.allow_blocked:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
