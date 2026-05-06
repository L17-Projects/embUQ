#!/usr/bin/env python3
"""CLI wrapper to run (or dry-run) GV sampling via :func:`sample_gv`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY, build_geometry
from meso_uq.structures.gv.material_parameters import validate_material_parameter_overrides
from meso_uq.structures.gv.sampling import GVRuntimeOptions, sample_gv
from meso_uq.structures.registry import GeometrySpec, StructureSpec, get_structure

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="gv", choices=("gv",))
    parser.add_argument(
        "--selection",
        default=None,
        help="Optional structure-qualified selection, for example gv:stretching.",
    )
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--geometry-id", default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable control override, for example --control tot_force=600.",
    )
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable material override, for example --material ka=1.1.",
    )
    parser.add_argument("--fixture-path", default=None, help="Optional provenance JSON recorded with the sampling run.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Runtime staging root. Numerical HDF5 artifacts are written to the canonical _runs/gv/numerical_data tree.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def _resolve_selected_experiment(args: argparse.Namespace) -> tuple[str, str | None]:
    selection = getattr(args, "selection", None)
    if selection is None:
        return str(args.structure), None if args.experiment is None else str(args.experiment)

    parts = str(selection).split(":")
    if len(parts) != 2:
        raise ValueError("GV sampling selection must use the form gv:<experiment>.")
    selected_structure, selected_experiment = parts
    if selected_structure != "gv":
        raise ValueError(
            f"GV sampling selection must target structure 'gv'."
        )
    if args.experiment is not None and str(args.experiment) != selected_experiment:
        raise ValueError(
            f"Conflicting selection values: --experiment={args.experiment!r} and --selection={selection!r}."
        )
    if str(args.structure) != selected_structure:
        raise ValueError(
            f"Conflicting structure values: --structure={args.structure!r} and --selection={selection!r}."
        )
    return selected_structure, selected_experiment


def _parse_material_overrides(material_items: list[str]) -> dict[str, float] | None:
    if not material_items:
        return None

    raw_overrides: dict[str, str] = {}
    for item in material_items:
        if "=" not in item:
            raise ValueError(f"Invalid material override '{item}'. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        raw_name = name.strip()
        if raw_name in raw_overrides:
            raise ValueError(f"Duplicate material override '{raw_name}'.")
        raw_overrides[raw_name] = raw_value
    return validate_material_parameter_overrides(raw_overrides)


def _parse_control_value(raw_value: str) -> float | tuple[float, ...]:
    if "," not in raw_value:
        return float(raw_value)
    values = tuple(float(item.strip()) for item in raw_value.split(",") if item.strip())
    if not values:
        raise ValueError("Comma-separated control sweeps must contain at least one value.")
    return values


def _parse_controls(control_items: list[str], structure: StructureSpec, experiment_name: str) -> dict[str, object]:
    experiment = structure.get_experiment(experiment_name, include_experimental=True)
    allowed = set(experiment.control_names)
    parsed: dict[str, object] = {}
    for item in control_items:
        if "=" not in item:
            raise ValueError(f"Invalid control override '{item}'. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        normalized_name = name.strip()
        if normalized_name not in allowed:
            raise ValueError(
                f"Unknown control '{normalized_name}' for experiment '{experiment_name}'. "
                f"Expected one of: {sorted(allowed)}"
            )
        value = _parse_control_value(raw_value)
        if normalized_name in parsed:
            previous = parsed[normalized_name]
            previous_values = previous if isinstance(previous, tuple) else (float(previous),)
            value_values = value if isinstance(value, tuple) else (float(value),)
            parsed[normalized_name] = tuple((*previous_values, *value_values))
        else:
            parsed[normalized_name] = value
    if not parsed:
        raise ValueError("At least one control must be provided via --control.")
    if not any(isinstance(value, tuple) for value in parsed.values()):
        for control_name in experiment.control_names:
            if control_name in parsed:
                parsed[control_name] = (float(parsed[control_name]),)
                break
    return parsed


def _resolve_geometry(args: argparse.Namespace, structure: StructureSpec) -> GeometrySpec:
    if args.geometry_id is not None:
        return structure.get_geometry(args.geometry_id)
    if args.radius is None and args.height is None:
        return DEFAULT_GV_GEOMETRY
    if args.radius is None or args.height is None:
        raise ValueError("Custom geometry requires both --radius and --height.")
    return build_geometry(radius=args.radius, height=args.height, source="scripts/workflows/gv/run_gv_sampling.py")


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return Path(args.output_root).expanduser().resolve()
    return (REPO_ROOT / "_runs").resolve()


def _load_fixture(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Fixture file must contain a JSON object.")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    structure_name, experiment_name = _resolve_selected_experiment(args)
    if experiment_name is None:
        if args.experiment is None:
            raise ValueError("Either --experiment or --selection must be provided.")
        experiment_name = str(args.experiment)
    structure = get_structure(structure_name)
    structure.get_experiment(experiment_name, include_experimental=True)

    geometry = _resolve_geometry(args, structure=structure)
    controls = _parse_controls(args.control, structure=structure, experiment_name=experiment_name)
    material_overrides = _parse_material_overrides(args.material)
    if material_overrides is None:
        raise ValueError("All GV material parameters must be provided with --material.")

    fixture_like = _load_fixture(args.fixture_path)
    output_root = _resolve_output_root(args)
    sample_payload = sample_gv(
        campaign_id=args.campaign_id,
        experiment=experiment_name,
        geometry={
            "radGV": float(geometry.parameters["radius"]),
            "height": float(geometry.parameters["height"]),
        },
        material_parameters=material_overrides,
        controls=controls,
        runtime_options=GVRuntimeOptions(
            controls=controls,
            output_root=output_root,
            timeout_seconds=args.timeout_seconds,
        ),
        dry_run=args.dry_run,
        write_artifacts=not args.dry_run,
    )

    print(f"manifest_path={sample_payload.manifest_path}")
    print(f"hdf5_path={sample_payload.hdf5_path}")
    print(f"status={getattr(sample_payload, 'status', 'completed')}")
    if fixture_like is not None:
        print(f"provenance_fixture={args.fixture_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through unit tests.
    raise SystemExit(main())
