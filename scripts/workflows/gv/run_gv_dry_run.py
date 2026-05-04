#!/usr/bin/env python3
"""Prepare a GV runtime dry-run without importing runtime dependencies at CLI import time."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from meso_uq.hpc_paths import default_runs_root  # noqa: E402
from meso_uq.structures.gv import DEFAULT_GV_GEOMETRY, build_geometry  # noqa: E402
from meso_uq.structures.registry import GeometrySpec, StructureSpec, get_structure  # noqa: E402
from meso_uq.structures.gv.material_parameters import validate_material_parameter_overrides

DEFAULT_RUNTIME_MODULE_ROOT = "meso_uq.structures.gv.runtime"
DEFAULT_WORKFLOW_NAME = "gv_runtime"
GV_SIMULATION_ROOT = (REPO_ROOT / "gv_simulation_files").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="gv", choices=("gv",))
    parser.add_argument("--selection", default=None, help="Optional structure-qualified selection, for example gv:stretching.")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--geometry-id", default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable control override, for example --control bpress=-0.001.",
    )
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable GV material override, for example --material ka=1.2.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--site", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--include-experimental", action="store_true", default=False)
    parser.add_argument("--runtime-module-root", default=DEFAULT_RUNTIME_MODULE_ROOT)
    return parser


def _parse_controls(control_items: list[str], experiment_name: str, structure: StructureSpec) -> dict[str, float]:
    experiment = structure.get_experiment(experiment_name, include_experimental=True)
    allowed = set(experiment.control_names)
    parsed: dict[str, float] = {}
    for item in control_items:
        if "=" not in item:
            raise ValueError(f"Invalid control override '{item}'. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name not in allowed:
            raise ValueError(
                f"Unsupported control '{name}' for experiment '{experiment_name}'. "
                f"Expected one of: {sorted(allowed)}"
            )
        try:
            parsed[name] = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value for control '{name}': {raw_value!r}. Expected a float."
            ) from exc
    return parsed


def _parse_material_overrides(material_items: list[str]) -> dict[str, float] | None:
    if not material_items:
        return None

    raw_overrides: dict[str, str] = {}
    seen_names: set[str] = set()
    for item in material_items:
        if "=" not in item:
            raise ValueError(f"Invalid material override '{item}'. Expected NAME=VALUE.")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Material parameter '{name}' is duplicated in overrides.")
        seen_names.add(name)
        raw_overrides[name] = raw_value
    return validate_material_parameter_overrides(raw_overrides)


def _resolve_selected_experiment(args: argparse.Namespace) -> tuple[str, str]:
    selection = getattr(args, "selection", None)
    if selection is None:
        if not args.experiment:
            raise ValueError("Either --experiment or --selection gv:<experiment> is required.")
        return str(args.structure), str(args.experiment)

    parts = str(selection).split(":")
    if len(parts) != 2:
        raise ValueError(
            "GV workflow selection must use the form gv:<experiment>."
        )
    selected_structure, selected_experiment = parts
    if selected_structure != "gv":
        raise ValueError(
            f"GV workflow selection does not accept structure '{selected_structure}'. Use gv:<experiment>."
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


def _resolve_geometry(args: argparse.Namespace, structure: StructureSpec) -> GeometrySpec:
    if args.geometry_id is not None:
        return structure.get_geometry(args.geometry_id)
    if args.radius is None and args.height is None:
        return DEFAULT_GV_GEOMETRY
    if args.radius is None or args.height is None:
        raise ValueError("Custom geometry requires both --radius and --height.")
    return build_geometry(radius=args.radius, height=args.height, source="scripts/workflows/gv/run_gv_dry_run.py")


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = (
            default_runs_root(
                REPO_ROOT,
                DEFAULT_WORKFLOW_NAME,
                site=args.site,
                run_tag=args.run_tag,
            )
        ).resolve()
    if output_root == GV_SIMULATION_ROOT or GV_SIMULATION_ROOT in output_root.parents:
        raise ValueError(
            "GV dry-run outputs must live under _runs or an explicit non-staging output root, "
            f"not inside {GV_SIMULATION_ROOT}."
        )
    return output_root


def _load_experiment_module(module_name: str):
    return importlib.import_module(module_name)


def _resolve_runtime_descriptor(experiment_module):
    for name in ("DESCRIPTOR", "RUNTIME_DESCRIPTOR", "runtime_descriptor"):
        descriptor = getattr(experiment_module, name, None)
        if descriptor is not None:
            break
    else:
        descriptor = None
        for name in ("build_runtime_descriptor", "build_dry_run_descriptor"):
            factory = getattr(experiment_module, name, None)
            if callable(factory):
                descriptor = factory()
                break
    if descriptor is None or not callable(getattr(descriptor, "plan", None)):
        raise AttributeError(
            f"Experiment runtime module {experiment_module.__name__!r} does not expose a RuntimeDescriptor-compatible descriptor."
        )
    return descriptor


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    to_manifest = getattr(value, "to_manifest", None)
    if callable(to_manifest):
        return _jsonify(to_manifest())
    if hasattr(value, "_asdict"):
        return _jsonify(value._asdict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _jsonify(vars(value))
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _plan_runtime_dry_run(descriptor, request: dict[str, Any]) -> Any:
    plan_kwargs = {
        "output_root": request["output_root"],
        "geometry": request["geometry"],
        "controls": request["controls"],
        "include_experimental": request["include_experimental"],
    }
    if request.get("material_parameter_overrides") is not None:
        plan_kwargs["material_parameter_overrides"] = request["material_parameter_overrides"]
    return descriptor.plan(**plan_kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        structure_name, experiment_name = _resolve_selected_experiment(args)
        structure = get_structure(structure_name)
        structure.get_experiment(
            experiment_name,
            include_experimental=args.include_experimental,
        )
        geometry = _resolve_geometry(args, structure)
        controls = _parse_controls(args.control, experiment_name, structure)
        material_parameter_overrides = _parse_material_overrides(args.material)
        output_root = _resolve_output_root(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    experiment_module_name = f"{args.runtime_module_root}.{experiment_name}"
    experiment_module = _load_experiment_module(experiment_module_name)
    descriptor = _resolve_runtime_descriptor(experiment_module)

    request = {
        "structure": structure.name,
        "experiment": experiment_name,
        "geometry": geometry.id,
        "controls": controls,
        "material_parameter_overrides": material_parameter_overrides,
        "output_root": output_root,
        "include_experimental": args.include_experimental,
        "dry_run": True,
    }
    runtime_payload = _plan_runtime_dry_run(descriptor, request)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _jsonify(runtime_payload)
    if not isinstance(manifest, dict):
        raise TypeError("RuntimeDescriptor.plan(...) must return a manifest-capable dry-run payload.")
    manifest.setdefault("structure", structure.name)
    manifest.setdefault("experiment", experiment_name)
    manifest.setdefault("geometry", geometry.id)
    manifest.setdefault("controls", controls)
    manifest["geometry_spec"] = {
        "id": geometry.id,
        "label": geometry.label,
        "parameters": dict(geometry.parameters),
        "source": geometry.source,
    }
    manifest["runtime_module"] = experiment_module_name
    manifest["selection"] = f"{structure.name}:{experiment_name}"
    manifest_path = output_root / "gv_runtime_dry_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Dry-run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
