#!/usr/bin/env python3
"""
Validate the machine-readable report produced by run_local_validation_matrix.py.

Usage:
    python scripts/platforms/workstation/validate_local_outputs.py \
        --report _o369_runs/local_validation_report.json

Checks:
  - Report is valid JSON with required top-level keys.
  - `status` is a recognised value.
  - `selections` is a non-empty list.
  - `phase2_backend_contract` and `phase2_backend_effective` are recognised.
  - `overlays` contains entries for every declared selection.
  - Propagation and MAP overlay files exist on disk for each selection.

Prints a concise summary and exits non-zero on any validation failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_KEYS = {
    "status",
    "selections",
    "overlays",
    "phase2_backend_contract",
    "phase2_backend_effective",
}
ALLOWED_STATUSES = {"passed", "failed", "partial"}
ALLOWED_PHASE2_BACKEND_CONTRACTS = {"dual_backend"}
ALLOWED_PHASE2_BACKENDS = {"cpu-mpi", "native-cuda"}


def _load_report(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"ERROR: report file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: report is not valid JSON: {exc}")


def validate_report(report: dict[str, Any], check_files: bool = True) -> list[str]:
    errors: list[str] = []

    missing_keys = REQUIRED_KEYS - set(report)
    if missing_keys:
        errors.append(f"Missing required keys: {sorted(missing_keys)}")
        return errors  # can't proceed without structure

    status = report["status"]
    if status not in ALLOWED_STATUSES:
        errors.append(f"status '{status}' not in {sorted(ALLOWED_STATUSES)}")

    backend_contract = report["phase2_backend_contract"]
    if backend_contract not in ALLOWED_PHASE2_BACKEND_CONTRACTS:
        errors.append(
            "phase2_backend_contract "
            f"'{backend_contract}' not in {sorted(ALLOWED_PHASE2_BACKEND_CONTRACTS)}"
        )

    backend_effective = report["phase2_backend_effective"]
    if backend_effective not in ALLOWED_PHASE2_BACKENDS:
        errors.append(
            "phase2_backend_effective "
            f"'{backend_effective}' not in {sorted(ALLOWED_PHASE2_BACKENDS)}"
        )

    selections = report["selections"]
    if not isinstance(selections, list) or not selections:
        errors.append("selections must be a non-empty list")
        return errors

    overlays = report.get("overlays", {})
    if not isinstance(overlays, dict):
        errors.append("overlays must be a mapping")
        return errors

    for sel in selections:
        if not isinstance(sel, str):
            errors.append(
                "selection entries must be strings in experiment:model-family:profile form; "
                f"got {type(sel).__name__}"
            )
            continue
        if sel not in overlays:
            errors.append(f"selection '{sel}' missing from overlays")
            continue
        payload = overlays[sel]
        if not isinstance(payload, dict):
            errors.append(f"overlays['{sel}'] must be a mapping")
            continue

        prop_plots = payload.get("propagation_vs_reference_plots", [])
        map_plots = payload.get("map_vs_reference_plots", [])

        if not prop_plots:
            errors.append(f"{sel}: no propagation_vs_reference_plots listed")
        if not map_plots:
            errors.append(f"{sel}: no map_vs_reference_plots listed")

        if check_files:
            for p in prop_plots:
                if not Path(p).exists():
                    errors.append(f"{sel}: missing propagation plot: {p}")
            for p in map_plots:
                if not Path(p).exists():
                    errors.append(f"{sel}: missing MAP plot: {p}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the local validation report from run_local_validation_matrix.py."
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Path to local_validation_report.json",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        default=False,
        help="Skip checking that overlay plot files exist on disk.",
    )
    args = parser.parse_args(argv)

    report = _load_report(args.report)
    errors = validate_report(report, check_files=not args.no_check_files)

    status = report.get("status", "unknown")
    selections = report.get("selections", [])
    selection_text = (
        ", ".join(str(selection) for selection in selections) if selections else "(none)"
    )
    print(f"Report:     {args.report}")
    print(f"Status:     {status}")
    print(f"Selections: {selection_text}")

    if errors:
        print(f"\nValidation FAILED ({len(errors)} error(s)):")
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        return 1

    overlays = report.get("overlays", {})
    total_prop = sum(len(v.get("propagation_vs_reference_plots", [])) for v in overlays.values())
    total_map = sum(len(v.get("map_vs_reference_plots", [])) for v in overlays.values())
    print("\nValidation PASSED")
    print(f"  {len(selections)} lane(s), {total_prop} propagation plot(s), {total_map} MAP plot(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
