#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.platforms.site_selector import resolve_hpc_site  # noqa: E402
from meso_uq.hpc_paths import detect_hpc_site  # noqa: E402
from meso_uq.postprocess import extract_map_from_directory  # noqa: E402
from meso_uq.vega_workflows import (  # noqa: E402
    ALL_WORKFLOW_EXPERIMENTS,
    VALID_MAP_STAGES,
    VALID_MODEL_FAMILIES,
    VALID_PROFILES,
    VALID_STRUCTURES,
    VegaWorkflowSelection,
    load_workflow_datasets,
    resolve_map_output_root,
    resolve_map_stage_input_root,
    resolve_workflow_config_path,
    resolve_workflow_output_root,
    select_workflow_datasets,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract MAP outputs for Vega workflows with explicit workflow selection.")
    parser.add_argument("--structure", choices=VALID_STRUCTURES, default=None)
    parser.add_argument("--experiment", choices=ALL_WORKFLOW_EXPERIMENTS, required=True)
    parser.add_argument("--model-family", choices=VALID_MODEL_FAMILIES, required=True)
    parser.add_argument("--profile", choices=VALID_PROFILES, required=True)
    parser.add_argument("--stage", choices=VALID_MAP_STAGES, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--site", choices=["vega", "karolina"], default=None)
    parser.add_argument("--maps-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--output", type=str, default=None, help="Single-dataset CSV output path.")
    args = parser.parse_args(argv)

    selection = VegaWorkflowSelection(
        args.experiment, args.model_family, args.profile, structure=args.structure
    )
    config_path = resolve_workflow_config_path(REPO_ROOT, selection, args.config)
    resolved_site = resolve_hpc_site(cli_site=args.site, env=os.environ, allow_hostname=True, default="vega")
    output_root = resolve_workflow_output_root(
        REPO_ROOT,
        selection,
        args.output_dir,
        run_tag=args.run_tag,
        site=resolved_site,
    )
    map_input_root = resolve_map_stage_input_root(output_root, args.stage)
    maps_root = resolve_map_output_root(output_root, args.stage, args.maps_dir)
    maps_root.mkdir(parents=True, exist_ok=True)

    datasets = load_workflow_datasets(
        REPO_ROOT, config_path, selection.experiment, structure=selection.structure
    )
    selected = select_workflow_datasets(datasets, dataset_name=args.dataset, diameter=args.diameter)

    if args.output is not None and len(selected) != 1:
        raise ValueError("--output can only be used when selecting exactly one dataset.")

    manifest: dict[str, object] = {
        "structure": selection.structure,
        "experiment": selection.experiment,
        "model_family": selection.model_family,
        "profile": selection.profile,
        "stage": args.stage,
        "config": str(config_path),
        "output_root": str(output_root),
        "maps_root": str(maps_root),
        "datasets": {},
    }

    for diameter, dataset_name in selected:
        run_dir = map_input_root / dataset_name
        output_csv = Path(args.output).resolve() if args.output else maps_root / f"{dataset_name}.csv"
        row = extract_map_from_directory(run_dir, output_csv=str(output_csv))
        payload = row.iloc[0].to_dict()
        payload["diameter_um"] = diameter
        payload["run_dir"] = str(run_dir)
        payload["output_csv"] = str(output_csv)
        manifest["datasets"][dataset_name] = payload
        print(f"{dataset_name} -> {output_csv}")

    manifest_path = maps_root / f"{args.stage}_map_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
