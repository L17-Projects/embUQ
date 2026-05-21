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

from meso_uq.active_learning.emb_34um_dnn_causal_validation_ingestion import (  # noqa: E402
    write_emb_34um_dnn_causal_validation_ingestion_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest EMB 3.4um DNN causal-validation DPD outputs and write campaign ingestion artifacts."
    )
    parser.add_argument(
        "--campaign-manifest",
        default=None,
        help="Path to emb_34um_dnn_causal_validation_manifest.json.",
    )
    parser.add_argument(
        "--campaign-root",
        default=None,
        help="Campaign root (used to resolve campaign manifest when --campaign-manifest is omitted).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Artifact output directory. Defaults to <campaign_root>/ingest.",
    )
    parser.add_argument(
        "--status-filename",
        action="append",
        default=[],
        help="Runtime status JSON filename under each candidate output root. May be passed multiple times.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit 0 even when ingestion status is blocked.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = write_emb_34um_dnn_causal_validation_ingestion_artifacts(
        campaign_manifest_path=Path(args.campaign_manifest) if args.campaign_manifest else None,
        campaign_root=Path(args.campaign_root) if args.campaign_root else None,
        output_root=Path(args.output_root) if args.output_root else None,
        status_filenames=tuple(args.status_filename)
        if args.status_filename
        else (
            "emb_34um_runtime_status.json",
            "runtime_status.json",
            "result_status.json",
        ),
    )
    print(f"ingestion_report={artifacts.report_path}")
    print(f"ingestion_manifest={artifacts.manifest_path}")
    print(f"completed_rows={artifacts.completed_rows_path}")
    print(f"summary_csv={artifacts.summary_csv_path}")
    print(f"status={artifacts.report['status']}")
    if artifacts.report["passed"] or args.allow_blocked:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
