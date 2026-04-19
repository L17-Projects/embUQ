#!/usr/bin/env python3
"""Convert a phase3b_map_manifest.json → per-diameter MAP JSON files.

The per-diameter JSON files are consumed by evaluate_map_mirheo_optimized*.py
to run Mirheo at MAP parameters.

Input manifest format (produced by scripts/vega/extract_map.py):
    {
      "experiment": "indentation",
      "model_family": "reduced-model",
      "datasets": {
        "indentation_3.2um": {
          "Yt": ..., "kb": ..., "d0": ..., "sigma": ...,
          "logLikelihood": ..., "logPrior": ..., "logPosterior": ...,
          "diameter_um": 3.2,
          "run_dir": "...", "output_csv": "..."
        }
      }
    }

Output per-diameter JSON (e.g. indentation_3.2um_map.json):
    {
      "diameter_um": 3.2,
      "sample_id": 0,
      "generation": -1,
      "logPosterior": 16.94,
      "parameter_names": ["Yt", "kb", "d0", "[Sigma]"],
      "parameters": [...]
    }

Parameter name detection:
    - Full model   : Yt, kb, b1, b2, a3, a4, d0, sigma  →  8 params
    - Reduced model: Yt, kb, d0, sigma                  →  4 params
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ordered parameter name sequences for each model family
_FULL_PARAM_KEYS = ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "sigma"]
_REDUCED_PARAM_KEYS = ["Yt", "kb", "d0", "sigma"]

# Output names (sigma → "[Sigma]" to match Korali convention)
_FULL_OUTPUT_NAMES = ["Yt", "kb", "b1", "b2", "a3", "a4", "d0", "[Sigma]"]
_REDUCED_OUTPUT_NAMES = ["Yt", "kb", "d0", "[Sigma]"]

_SKIP_KEYS = {"logLikelihood", "logPrior", "logPosterior", "diameter_um", "run_dir", "output_csv"}


def detect_param_layout(dataset: dict) -> tuple[list[str], list[str]]:
    """Return (source_keys, output_names) for the parameter vector.

    Raises ValueError if the dataset doesn't match a known layout.
    """
    keys = set(dataset) - _SKIP_KEYS
    if all(k in keys for k in _FULL_PARAM_KEYS):
        return _FULL_PARAM_KEYS, _FULL_OUTPUT_NAMES
    if all(k in keys for k in _REDUCED_PARAM_KEYS) and "b1" not in keys:
        return _REDUCED_PARAM_KEYS, _REDUCED_OUTPUT_NAMES
    raise ValueError(
        f"Cannot determine parameter layout from dataset keys: {sorted(keys)}. "
        f"Expected one of {_FULL_PARAM_KEYS} or {_REDUCED_PARAM_KEYS}."
    )


def convert_dataset(dataset_name: str, dataset: dict) -> dict:
    """Convert one manifest dataset entry to the per-diameter MAP JSON dict."""
    source_keys, output_names = detect_param_layout(dataset)
    parameters = [float(dataset[k]) for k in source_keys]
    return {
        "diameter_um": float(dataset["diameter_um"]),
        "sample_id": 0,
        "generation": -1,
        "logPosterior": float(dataset["logPosterior"]),
        "parameter_names": output_names,
        "parameters": parameters,
    }


def convert_manifest(manifest_path: Path, output_dir: Path) -> list[Path]:
    """Convert all datasets in a manifest to per-diameter JSON files.

    Returns the list of written file paths.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    datasets = manifest.get("datasets", {})
    if not datasets:
        raise ValueError(f"Manifest has no 'datasets' entries: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for dataset_name, dataset in datasets.items():
        entry = convert_dataset(dataset_name, dataset)
        out_path = output_dir / f"{dataset_name}_map.json"
        with open(out_path, "w") as f:
            json.dump(entry, f, indent=2)
        written.append(out_path)
        print(f"  {dataset_name} -> {out_path}")

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert phase3b_map_manifest.json → per-diameter MAP JSON files"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to *_map_manifest.json (produced by scripts/vega/extract_map.py)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write per-diameter MAP JSON files",
    )
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    written = convert_manifest(args.manifest, args.output_dir)
    print(f"Wrote {len(written)} MAP JSON file(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
