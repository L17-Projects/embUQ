#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG_FILES = [
    "inference/configs/production/inference_config_compression.yaml",
    "inference/configs/production/inference_config_indentation.yaml",
    "inference/configs/validation/validation_config_compression.yaml",
    "inference/configs/validation/validation_config_indentation.yaml",
    "reduced/configs/production/reduced_config_compression.yaml",
    "reduced/configs/production/reduced_config_indentation.yaml",
    "reduced/configs/validation/validation_config_compression.yaml",
    "reduced/configs/validation/validation_config_indentation.yaml",
    "reduced/configs/ci/ci_canary_config_compression.yaml",
]

SURROGATE_BLOCK_PATTERN = re.compile(
    r"(?ms)^surrogate:\n\s+backend:\s*dnn\n\s+predictive_mc_samples:\s*\d+\n\s+predictive_mc_chunk_size:\s*\d+"
)

REQUIRED_SNIPPETS = [
    (
        "inference/scripts/run_phase_1.py",
        "preload_fn(diameter_um, device=device, backend=surrogate_backend)",
        "phase1 preload must honor configured surrogate backend",
    ),
    (
        "inference/scripts/run_phase_3b.py",
        "preload_fn(diameter_um, device=device, backend=surrogate_backend)",
        "phase3b preload must honor configured surrogate backend",
    ),
    (
        "propagation/scripts/run_phase1_propagation.py",
        "preload_fn(diameter_um, device=args.device, backend=surrogate_backend)",
        "phase1 propagation preload must honor configured surrogate backend",
    ),
    (
        "propagation/scripts/run_phase3b_propagation.py",
        "preload_fn(diameter_um, device=args.device, backend=surrogate_backend)",
        "phase3b propagation preload must honor configured surrogate backend",
    ),
    (
        "compression/evalkit/posterior_compression.py",
        "def compute_compression_surrogate(\n    sample: Dict[str, Any], displ: List[float], diameter_um: float, device: str = \"cpu\"\n)",
        "compression single-sample surrogate path must accept explicit device",
    ),
    (
        "indentation/evalkit/posterior_indentation.py",
        "def compute_indentation_surrogate(\n    sample: Dict[str, Any], forces: List[float], diameter_um: float, device: str = \"cpu\"\n)",
        "indentation single-sample surrogate path must accept explicit device",
    ),
]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def main() -> int:
    failures: list[str] = []

    for rel_path in CONFIG_FILES:
        content = _read(rel_path)
        if SURROGATE_BLOCK_PATTERN.search(content) is None:
            failures.append(
                f"{rel_path}: missing explicit surrogate backend block "
                "(backend + predictive_mc_samples + predictive_mc_chunk_size)."
            )

    for rel_path, snippet, reason in REQUIRED_SNIPPETS:
        content = _read(rel_path)
        if snippet not in content:
            failures.append(f"{rel_path}: {reason}.")

    if failures:
        print("BNN backend canary failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print("BNN backend canary passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
