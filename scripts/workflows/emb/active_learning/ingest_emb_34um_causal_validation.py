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

from meso_uq.active_learning.emb_34um_causal_validation_ingestion import (  # noqa: E402
    write_emb_34um_causal_validation_ingestion_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest EMB 3.4um causal validation outputs and write audit/replacement artifacts."
    )
    parser.add_argument("--campaign-manifest", required=True, help="Path to emb_34um_causal_validation manifest JSON.")
    parser.add_argument("--output-root", default=None, help="Artifact output directory. Defaults to <campaign_root>/ingest.")
    parser.add_argument(
        "--replacement-manifest-output",
        default=None,
        help="Optional path to write replacement plan JSON (no manifest mutation).",
    )
    parser.add_argument(
        "--success-filename",
        action="append",
        default=[],
        help="Success marker filename expected under output_root. May be passed multiple times.",
    )
    parser.add_argument(
        "--status-filename",
        action="append",
        default=[],
        help="Status JSON filename expected under output_root. May be passed multiple times.",
    )
    parser.add_argument("--allow-legacy-provenance", action="store_true", help="Disable fresh-only provenance guardrails.")
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when audit is blocked.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = write_emb_34um_causal_validation_ingestion_artifacts(
        campaign_manifest_path=Path(args.campaign_manifest),
        output_root=Path(args.output_root) if args.output_root else None,
        write_replacement_plan=args.replacement_manifest_output is not None,
        replacement_manifest_output_path=(Path(args.replacement_manifest_output) if args.replacement_manifest_output else None),
        success_filenames=tuple(args.success_filename) if args.success_filename else ("F_Delta.dat",),
        status_filenames=tuple(args.status_filename) if args.status_filename else ("runtime_status.json", "result_status.json", "emb_34um_runtime_status.json"),
        fresh_only=not args.allow_legacy_provenance,
    )
    print(f"ingestion_report={artifacts.report_path}")
    print(f"ingestion_manifest={artifacts.manifest_path}")
    print(f"summary_csv={artifacts.summary_csv_path}")
    print(f"status={artifacts.report['status']}")
    if artifacts.replacement_plan_path is not None:
        print(f"replacement_plan={artifacts.replacement_plan_path}")
    if artifacts.report["passed"] or args.allow_blocked:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
