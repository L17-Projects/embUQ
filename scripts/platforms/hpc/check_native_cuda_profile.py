#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.inference.native_cuda_performance import (  # noqa: E402
    DEFAULT_NATIVE_CUDA_PERFORMANCE_THRESHOLDS,
    NativeCudaPerformanceThresholds,
    check_native_cuda_profile,
)


def _parser() -> argparse.ArgumentParser:
    defaults = DEFAULT_NATIVE_CUDA_PERFORMANCE_THRESHOLDS
    parser = argparse.ArgumentParser(
        description=(
            "Check a HUQ_PSI_NATIVE_CUDA_PROFILE_JSONL file against fixed "
            "NativeCuda Phase 2 performance thresholds."
        )
    )
    parser.add_argument("profile_jsonl", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-setup-count", type=int, default=defaults.min_setup_count)
    parser.add_argument("--min-batch-count", type=int, default=defaults.min_batch_count)
    parser.add_argument("--max-setup-total-seconds", type=float, default=defaults.max_setup_total_seconds)
    parser.add_argument("--max-setup-compile-seconds", type=float, default=defaults.max_setup_compile_seconds)
    parser.add_argument("--max-batch-total-seconds", type=float, default=defaults.max_batch_total_seconds)
    parser.add_argument("--max-batch-alloc-seconds", type=float, default=defaults.max_batch_alloc_seconds)
    parser.add_argument("--max-batch-h2d-seconds", type=float, default=defaults.max_batch_h2d_seconds)
    parser.add_argument("--max-batch-compute-seconds", type=float, default=defaults.max_batch_compute_seconds)
    parser.add_argument("--max-batch-d2h-seconds", type=float, default=defaults.max_batch_d2h_seconds)
    parser.add_argument(
        "--max-batch-host-reduce-seconds",
        type=float,
        default=defaults.max_batch_host_reduce_seconds,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    thresholds = NativeCudaPerformanceThresholds(
        min_setup_count=args.min_setup_count,
        min_batch_count=args.min_batch_count,
        max_setup_total_seconds=args.max_setup_total_seconds,
        max_setup_compile_seconds=args.max_setup_compile_seconds,
        max_batch_total_seconds=args.max_batch_total_seconds,
        max_batch_alloc_seconds=args.max_batch_alloc_seconds,
        max_batch_h2d_seconds=args.max_batch_h2d_seconds,
        max_batch_compute_seconds=args.max_batch_compute_seconds,
        max_batch_d2h_seconds=args.max_batch_d2h_seconds,
        max_batch_host_reduce_seconds=args.max_batch_host_reduce_seconds,
    )
    report = check_native_cuda_profile(args.profile_jsonl, thresholds=thresholds)
    payload = report.to_manifest()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    if report.passed:
        return 0
    for mismatch in report.mismatches:
        print(f"NativeCuda performance threshold exceeded: {mismatch}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
