#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.workflows.legacy import (
    prepend_legacy_evalkit_paths,
    resolve_legacy_relative_path,
    resolve_legacy_surrogate_backend,
    warn_legacy_workflow_surface,
)

prepend_legacy_evalkit_paths(PROJECT_ROOT)

from emb.compression.evalkit.posterior_compression import (
    compute_compression_surrogate,
    preload_compression_surrogate,
)
from emb.indentation.evalkit.posterior_indentation import (
    compute_indentation_surrogate,
    preload_indentation_surrogate,
)
from meso_uq.experiments import load_experiments
from meso_uq.postprocess.propagation import propagate_run_directory


def _reference_csv_for_experiment(exp, diameter_um: float, output_dir: Path) -> Path:
    data_file = exp.data_file(diameter_um)
    reference_df = pd.read_csv(data_file, sep=r"\s+", engine="python", comment="#")
    reference_csv = output_dir / "references" / f"{exp.dataset_name(diameter_um)}.csv"
    reference_csv.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(reference_csv, index=False)
    return reference_csv


def main() -> int:
    warn_legacy_workflow_surface("propagation/scripts/run_phase3b_propagation.py")
    parser = argparse.ArgumentParser(
        description="Run lightweight propagation from Phase 3b posterior samples"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="_setup")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Surrogate device: cpu (default) or gpu (cuda)",
    )
    args = parser.parse_args()

    config_path = resolve_legacy_relative_path(PROJECT_ROOT, args.config)
    os.environ["HUQ_INFERENCE_CONFIG"] = str(config_path)
    output_root = resolve_legacy_relative_path(PROJECT_ROOT, args.output_dir)

    with open(config_path, "rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)
    experiments = [exp for exp in load_experiments(config, PROJECT_ROOT) if exp.enabled]
    surrogate_backend = resolve_legacy_surrogate_backend(config)

    for exp in experiments:
        preload_map = {
            "compression": preload_compression_surrogate,
            "indentation": preload_indentation_surrogate,
        }
        preload_fn = preload_map.get(exp.name)
        if preload_fn is not None:
            for diameter_um in exp.diameters:
                preload_fn(diameter_um, device=args.device, backend=surrogate_backend)

    eval_map = {
        "compression": lambda sample, pts, d: compute_compression_surrogate(
            sample, pts, d, device=args.device
        ),
        "indentation": lambda sample, pts, d: compute_indentation_surrogate(
            sample, pts, d, device=args.device
        ),
    }

    for exp in experiments:
        for diameter_um in exp.diameters:
            run_dir = output_root / "results_phase_3b" / exp.dataset_name(diameter_um)
            summary_csv = (
                output_root / "propagation_phase3b" / exp.dataset_name(diameter_um) / "summary.csv"
            )
            reference_points = exp.get_reference_points(diameter_um)
            evaluate = lambda sample, pts, name=exp.name, d=diameter_um: eval_map[name](
                sample, pts, d
            )
            result = propagate_run_directory(
                run_dir,
                reference_points=reference_points,
                evaluate_sample=evaluate,
                output_csv=summary_csv,
            )
            reference_csv = _reference_csv_for_experiment(exp, diameter_um, output_root)
            print(
                f"{exp.dataset_name(diameter_um)} -> {result['summary_csv']} | ref={reference_csv} | samples={result['num_samples']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
