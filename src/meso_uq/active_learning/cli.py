"""Package-native Active Learning CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from meso_uq.active_learning.acquisition import SUPPORTED_POLICIES

from . import cli_adapters


def build_parser() -> argparse.ArgumentParser:
    """Build the Active Learning CLI parser.

    The interface is intentionally render-only; commands generate local artifacts and never
    submit scheduler jobs.
    """

    parser = argparse.ArgumentParser(
        description="Active-Learning utility command line tools (render-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Dry-run/render-only mode is always active: no scheduler jobs are submitted.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate generation or candidate active-learning inputs."
    )
    validate_parser.add_argument(
        "input",
        type=Path,
        help="Path to a generation config or candidate payload.",
    )
    validate_parser.add_argument(
        "--kind",
        choices=("auto", "generation", "candidate"),
        default="auto",
        help="Explicit validation kind (default: auto).",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a candidate batch, validate candidates, and write artifacts.",
    )
    generate_parser.add_argument(
        "config_path",
        type=Path,
        help="Path to candidate-generation config.",
    )
    generate_parser.add_argument("--run-id", default="active_learning", help="Run identifier used for artifact paths.")
    generate_parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration index for artifact namespace (non-negative).",
    )
    generate_parser.add_argument(
        "--output-root",
        default="_runs/active_learning",
        help="Output directory for generated artifacts.",
    )
    generate_parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        dest="include_plot",
        default=True,
        help="Control generation validation plot output.",
    )

    acquire_parser = subparsers.add_parser(
        "acquire",
        help="Score candidates and write acquisition artifacts.",
    )
    acquire_parser.add_argument(
        "candidates_path",
        type=Path,
        help="Path to candidate/result payload used for acquisition scoring.",
    )
    acquire_parser.add_argument(
        "--policy",
        default="uncertainty",
        choices=tuple(sorted(SUPPORTED_POLICIES)),
        help="Acquisition scoring policy.",
    )
    acquire_parser.add_argument("--run-id", default="active_learning", help="Run identifier used for artifact paths.")
    acquire_parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration index for artifact namespace (non-negative).",
    )
    acquire_parser.add_argument(
        "--output-root",
        default="_runs/active_learning",
        help="Output directory for acquisition artifacts.",
    )
    acquire_parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable weight entry for weighted policies (for example --weight uncertainty=1.0).",
    )
    acquire_parser.add_argument(
        "--expected-improvement-baseline",
        type=float,
        default=None,
        help="Expected-improvement baseline when using expected_improvement policy.",
    )
    acquire_parser.add_argument(
        "--diversity-path",
        action="append",
        default=[],
        help="Repeatable dotted path to candidate field used by diversity policy.",
    )
    acquire_parser.add_argument(
        "--reference-candidates",
        type=Path,
        default=None,
        help="Optional reference candidates for diversity policy.",
    )
    acquire_parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        dest="include_plot",
        default=True,
        help="Control acquisition validation plot output.",
    )

    return parser


def _emit_json(result: object) -> None:
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def _normalize_weights(weights: list[str]) -> tuple[str, ...] | None:
    if not weights:
        return None
    return tuple(weights)


def _normalize_diversity_paths(paths: list[str]) -> tuple[str, ...] | None:
    if not paths:
        return None
    return tuple(paths)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        try:
            kind, payload = cli_adapters.validate_active_learning_config(
                input_path=args.input,
                kind=args.kind,
            )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        result = {"command": "validate", "kind": kind, "result": payload}
        _emit_json(result)
        if kind == cli_adapters.CANDIDATE_INPUT_CONFIG_KIND and payload.get("summary") != "ok":
            return 1
        return 0

    if args.command == "generate":
        try:
            result = cli_adapters.run_generate_command(
                config_path=args.config_path,
                output_root=args.output_root,
                run_id=args.run_id,
                iteration=args.iteration,
                include_plot=args.include_plot,
            )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        _emit_json(result)
        return 0

    if args.command == "acquire":
        try:
            result = cli_adapters.run_acquire_command(
                candidates_path=args.candidates_path,
                policy=args.policy,
                output_root=args.output_root,
                run_id=args.run_id,
                iteration=args.iteration,
                weights=_normalize_weights(args.weight),
                expected_improvement_baseline=args.expected_improvement_baseline,
                diversity_paths=_normalize_diversity_paths(args.diversity_path),
                reference_candidates_path=args.reference_candidates,
                include_plot=args.include_plot,
            )
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        _emit_json(result)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
