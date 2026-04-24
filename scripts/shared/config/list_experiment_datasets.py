#!/usr/bin/env python3
"""
List enabled experiment datasets from the inference config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="inference/configs/production/inference_config_compression.yaml",
        help="Path to inference config YAML.",
    )
    parser.add_argument(
        "--format",
        choices=["dataset", "name", "name-diameter"],
        default="name-diameter",
        help="Output format for each dataset.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(project_root, args.config)
    with open(config_path, "rb") as handle:
        config = yaml.load(handle, Loader=yaml.CLoader)

    raw_experiments = config.get("experiments")
    if not raw_experiments:
        name = config.get("experiment", "compression")
        diameters = config.get("emb_diameters", [])
        raw_experiments = [{"name": name, "enabled": True, "diameters": diameters}]

    for exp in raw_experiments:
        if not exp.get("enabled", True):
            continue
        name = exp.get("name")
        diameters = exp.get("diameters", config.get("emb_diameters", []))
        for diameter in sorted(diameters):
            if args.format == "dataset":
                print(f"{name}_{diameter}um")
            elif args.format == "name":
                print(name)
            else:
                print(f"{name} {diameter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
