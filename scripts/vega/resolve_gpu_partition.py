#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.scheduler_routing import GpuPartitionPolicy, route_gpu_partition  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve GPU partition from time limit using strict HUQ-EMB routing policy."
    )
    parser.add_argument("--time-limit", required=True, help="SLURM walltime (e.g. 00:15:00)")
    parser.add_argument("--short-partition", default="dev")
    parser.add_argument("--long-partition", default="gpu")
    parser.add_argument("--threshold-seconds", type=int, default=1800)
    args = parser.parse_args(argv)

    partition = route_gpu_partition(
        args.time_limit,
        policy=GpuPartitionPolicy(
            short_partition=args.short_partition,
            long_partition=args.long_partition,
            threshold_seconds=args.threshold_seconds,
        ),
    )
    print(partition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
