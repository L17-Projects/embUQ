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

DEFAULT_RUNTIME_MODULE_ROOT = "meso_uq.structures.gv.runtime"
DEFAULT_WORKFLOW_NAME = "gv_runtime"
GV_SIMULATION_ROOT = (REPO_ROOT / "gv_simulation_files").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="gv", choices=("gv",))
    parser.add_argument("--experiment", required=True)
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
    return descriptor.plan(
        output_root=request["output_root"],
        geometry=request["geometry"],
        controls=request["controls"],
        include_experimental=request["include_experimental"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        structure = get_structure(args.structure)
        structure.get_experiment(
            args.experiment,
            include_experimental=args.include_experimental,
        )
        geometry = _resolve_geometry(args, structure)
        controls = _parse_controls(args.control, args.experiment, structure)
        output_root = _resolve_output_root(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    experiment_module_name = f"{args.runtime_module_root}.{args.experiment}"
    experiment_module = _load_experiment_module(experiment_module_name)
    descriptor = _resolve_runtime_descriptor(experiment_module)

    request = {
        "structure": structure.name,
        "experiment": args.experiment,
        "geometry": geometry.id,
        "controls": controls,
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
    manifest.setdefault("experiment", args.experiment)
    manifest.setdefault("geometry", geometry.id)
    manifest.setdefault("controls", controls)
    manifest["geometry_spec"] = {
        "id": geometry.id,
        "label": geometry.label,
        "parameters": dict(geometry.parameters),
        "source": geometry.source,
    }
    manifest["runtime_module"] = experiment_module_name
    manifest_path = output_root / "gv_runtime_dry_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Dry-run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
