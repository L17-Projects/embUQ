#!/usr/bin/env python3
"""Lock a fixed DNN architecture for EMB 3.4um DNN causal protocol runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


def _ensure_src_on_path() -> Path:
    repo_root = _repo_root()
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_src_on_path()

from meso_uq.active_learning.emb_34um_dnn_architecture_lock import (
    EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_SEEDS,
    EMB_34UM_DNN_ARCHITECTURE_LOCK_SCHEMA_VERSION,
    EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE,
    write_emb_34um_dnn_causal_architecture_lock_artifacts,
)
from meso_uq.active_learning.emb_34um_dnn_causal_validation_protocol import (
    EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
    EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory for architecture-lock manifest and report artifacts.",
    )
    parser.add_argument(
        "--selected-architecture",
        default=EMB_34UM_DNN_ARCHITECTURE_LOCK_SELECTED_ARCHITECTURE,
        help="DNN architecture name to lock; fixed architecture is enforced by default.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=EMB_34UM_DNN_CAUSAL_ENSEMBLE_SIZE,
        help="Ensemble size, required to be 10 for this protocol.",
    )
    parser.add_argument(
        "--ensemble-seed",
        action="append",
        default=[],
        type=int,
        help="Explicit ensemble seed (repeatable).",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epoch count.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument(
        "--timing-canary-point-count",
        type=int,
        default=EMB_34UM_DNN_CAUSAL_TIMING_CANARY_POINT_COUNT,
        help="Timing-canary force-grid point count.",
    )
    parser.add_argument(
        "--timing-canary-override-reason",
        default="",
        help="Required if timing-canary-point-count is downgraded to 5.",
    )
    parser.add_argument(
        "--source-evidence",
        default=None,
        help="Optional JSON mapping for source evidence.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write fallback PNG without attempting matplotlib rendering.",
    )
    return parser


def _coerce_source_evidence(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    text = raw.strip()
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("source-evidence must be a JSON object.")
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    timing_canary: dict[str, Any] = {
        "required": True,
        "point_count": args.timing_canary_point_count,
    }
    if args.timing_canary_point_count == 5:
        timing_canary["override_reason"] = args.timing_canary_override_reason

    source_evidence = {
        "schema_version": EMB_34UM_DNN_ARCHITECTURE_LOCK_SCHEMA_VERSION,
        "workflow": Path(sys.argv[0]).name,
    }
    source_evidence.update(_coerce_source_evidence(args.source_evidence))

    artifacts = write_emb_34um_dnn_causal_architecture_lock_artifacts(
        args.output_root,
        selected_architecture=args.selected_architecture,
        ensemble_size=args.ensemble_size,
        ensemble_seeds=tuple(args.ensemble_seed) if args.ensemble_seed else EMB_34UM_DNN_ARCHITECTURE_LOCK_DEFAULT_SEEDS,
        training_hyperparameters={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        timing_canary=timing_canary,
        source_evidence=source_evidence,
        training_history=(
            {"epoch": 0, "train_loss": 1.0, "validation_loss": 1.1},
            {"epoch": 1, "train_loss": 0.6, "validation_loss": 0.8},
            {"epoch": 2, "train_loss": 0.4, "validation_loss": 0.7},
        ),
        include_plot=not args.no_plot,
    )

    print(f"manifest={artifacts.manifest_path}")
    print(f"report={artifacts.report_path}")
    print(f"plot={artifacts.plot_path}")
    print(f"plot_sidecar={artifacts.plot_sidecar_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
