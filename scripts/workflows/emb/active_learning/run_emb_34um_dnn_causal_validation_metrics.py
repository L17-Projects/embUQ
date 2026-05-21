#!/usr/bin/env python3
from __future__ import annotations

"""Generate EMB 3.4um DNN causal AL-vs-LHS metric rows."""

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to locate repository root.")


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_ingestion import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_COMPLETED_ROWS_FILENAME,
    write_emb_34um_dnn_causal_validation_ingestion_artifacts,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_metrics import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_REPORT_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_FILENAME,
    write_emb_34um_dnn_causal_validation_metric_artifacts,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", default=None, help="Campaign root to ingest before scoring.")
    parser.add_argument("--completed-rows", default=None, help="Completed rows JSON from DNN campaign ingestion.")
    parser.add_argument("--output-root", default=None, help="Metric output directory. Defaults to <campaign-root>/ingest.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ensemble-size", type=int, default=EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _completed_rows_source(args: argparse.Namespace, output_root: Path) -> object:
    if args.completed_rows:
        return Path(args.completed_rows)
    if not args.campaign_root:
        raise ValueError("Either --campaign-root or --completed-rows is required.")
    artifacts = write_emb_34um_dnn_causal_validation_ingestion_artifacts(
        campaign_root=Path(args.campaign_root),
        output_root=output_root,
    )
    return artifacts.completed_rows_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_root:
        output_root = Path(args.output_root)
    elif args.campaign_root:
        output_root = Path(args.campaign_root) / "ingest"
    elif args.completed_rows:
        output_root = Path(args.completed_rows).parent / "metrics"
    else:
        raise ValueError("Either --campaign-root or --completed-rows is required.")

    completed_rows_source = _completed_rows_source(args, output_root)
    train_kwargs = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "validation_fraction": float(args.validation_fraction),
    }
    artifacts = write_emb_34um_dnn_causal_validation_metric_artifacts(
        completed_rows_source=completed_rows_source,
        output_root=output_root,
        rows_filename=EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_ROWS_FILENAME,
        report_filename=EMB_34UM_DNN_CAUSAL_VALIDATION_METRIC_REPORT_FILENAME,
        dry_run=bool(args.dry_run),
        ensemble_size=int(args.ensemble_size),
        device=str(args.device),
        train_kwargs=train_kwargs,
    )
    print(json.dumps({"rows_path": str(artifacts.rows_path), "report_path": str(artifacts.report_path)}, sort_keys=True))
    return 0 if artifacts.report_payload.get("passed") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
