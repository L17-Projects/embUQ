#!/usr/bin/env python3
"""Render the EMB 3.4um DNN causal-validation controller manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _script_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Unable to resolve repository root.")


_REPO_ROOT = _script_root()
_SRC_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from meso_uq.active_learning.emb_34um_dnn_causal_validation_design import (  # noqa: E402
    EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME,
    EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME,
)


CONTROLLER_SCHEMA_VERSION = "meso_uq.active_learning.emb_34um_dnn_causal_validation_controller.v1"
CONTROLLER_MANIFEST_FILENAME = "emb_34um_dnn_causal_validation_controller_manifest.json"
DEFAULT_EXECUTION_MODE = "render-only"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _coerce_stage_entries(entries: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(entries, list):
        raise ValueError("command_inventory.entries must be a list.")
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("command_inventory.entries items must be mappings.")
        normalized.append(dict(entry))
    return tuple(normalized)


def _stage_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(entry.get("mode", "")),
        "replica": int(entry.get("replica", 0)),
        "cycle": int(entry.get("cycle", 0)),
        "candidate_count": int(entry.get("candidate_count", 0)),
        "array": str(entry.get("array", "")),
        "execution_mode": str(entry.get("execution_mode", DEFAULT_EXECUTION_MODE)),
        "batch_summary_path": str(entry.get("batch_summary_path", "")),
        "command": str(entry.get("command", "")),
    }


def build_emb_34um_dnn_causal_validation_controller(
    *,
    campaign_root: Path,
    design_manifest_path: Path | None = None,
    execution_mode: str = DEFAULT_EXECUTION_MODE,
) -> dict[str, Any]:
    campaign_root = Path(campaign_root)
    design_manifest_path = Path(design_manifest_path or campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_MANIFEST_FILENAME)
    design_manifest = _load_json(design_manifest_path)
    command_inventory = design_manifest.get("command_inventory", {})
    if not isinstance(command_inventory, Mapping):
        raise ValueError("command_inventory must be a mapping.")

    entries = _coerce_stage_entries(command_inventory.get("entries", []))
    stage_records = tuple(_stage_record(entry) for entry in entries)
    controller_manifest = {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "execution_mode": execution_mode,
        "render_only": execution_mode == DEFAULT_EXECUTION_MODE,
        "campaign_root": str(campaign_root),
        "design_manifest_path": str(design_manifest_path),
        "design_command_inventory_path": str(
            campaign_root / EMB_34UM_DNN_CAUSAL_VALIDATION_COMMAND_INVENTORY_FILENAME
        ),
        "design_schema_version": str(design_manifest.get("schema_version", "")),
        "command_inventory": {
            "count": len(entries),
            "walltime": str(command_inventory.get("walltime", "")),
            "concurrent_jobs": int(command_inventory.get("concurrent_jobs", 0)),
            "retry_limit": int(command_inventory.get("retry_limit", 0)),
            "entries": [dict(entry) for entry in entries],
        },
        "parallel_arrays": list(stage_records),
        "stage_order": [record["mode"] for record in stage_records],
        "submission": {
            "submitted": False,
            "submission_commands": [],
            "render_only": True,
        },
    }

    manifest_path = campaign_root / CONTROLLER_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(controller_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest_path": manifest_path, "manifest": controller_manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--design-manifest")
    parser.add_argument("--execution-mode", default=DEFAULT_EXECUTION_MODE)
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    args = build_parser().parse_args(argv)
    build_emb_34um_dnn_causal_validation_controller(
        campaign_root=Path(args.campaign_root),
        design_manifest_path=Path(args.design_manifest) if args.design_manifest else None,
        execution_mode=str(args.execution_mode),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
