#!/usr/bin/env python3
from __future__ import annotations

"""Write the EMB 3.4um retrospective active-design benchmark plot."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning.emb_34um_retrospective_al_benchmark import (  # noqa: E402
    write_emb_34um_retrospective_al_benchmark_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-all",
        default="emb/indentation/surrogate/diameters/3.4um/data/samples_all.dat",
        help="EMB 3.4um samples_all.dat table.",
    )
    parser.add_argument("--output-root", required=True, help="Directory for benchmark artifacts.")
    parser.add_argument("--candidate-pool-size", type=int, default=500)
    parser.add_argument("--max-executed-curves", type=int, default=300)
    parser.add_argument("--round-size", type=int, default=30)
    parser.add_argument("--validation-count", type=int, default=3000)
    parser.add_argument("--lhs-replicates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = write_emb_34um_retrospective_al_benchmark_artifacts(
        samples_all_path=Path(args.samples_all),
        output_root=Path(args.output_root),
        candidate_pool_size=args.candidate_pool_size,
        max_executed_curves=args.max_executed_curves,
        round_size=args.round_size,
        validation_count=args.validation_count,
        lhs_replicates=args.lhs_replicates,
        seed=args.seed,
        neighbors=args.neighbors,
        include_plot=not args.no_plot,
    )
    print(f"manifest={artifacts.manifest_path}")
    print(f"plot={artifacts.plot_path}")
    print(f"summary_csv={artifacts.summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
