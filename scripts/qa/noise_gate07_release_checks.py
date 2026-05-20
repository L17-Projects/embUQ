#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MappingLike = dict[str, Any]


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, payload: MappingLike) -> None:
    lines = [
        "# NOISE Gate 07 Release Checks",
        "",
        f"Status: {payload['status']}",
        "",
        f"Source manifest: `{payload.get('source_release_manifest')}`",
        "",
        f"Source commit: `{payload.get('source_commit')}`",
        "",
        f"Merge boundary: `{payload.get('merge_boundary')}`",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {failure}" for failure in payload["failures"] or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in payload["warnings"] or ["None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gate07(release_manifest: MappingLike) -> MappingLike:
    failures: list[str] = []
    warnings: list[str] = []
    gate07 = release_manifest.get("gate07")
    if not isinstance(gate07, dict) or gate07.get("pass") is not True:
        failures.append("release manifest gate07.pass is not true.")
    if release_manifest.get("config_validation", {}).get("passed") is not True:
        failures.append("config validation did not pass.")
    entries = release_manifest.get("artifact_index", {}).get("entries")
    if not isinstance(entries, dict) or not entries:
        failures.append("artifact index is missing or empty.")
    if release_manifest.get("merge_boundary") not in {"merged", "human_review_required"}:
        failures.append("release manifest must record merged or human_review_required merge boundary.")
    if release_manifest.get("merge_boundary") == "human_review_required":
        warnings.append("PR stack is at a human review/merge boundary.")
    if release_manifest.get("no_karolina_interaction") is not True:
        failures.append("release manifest must record no active Karolina interaction.")
    return {
        "schema_version": 1,
        "gate": "NOISE Gate 07 Release checks",
        "pass": not failures,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "warnings": warnings,
        "source_commit": release_manifest.get("provenance", {}).get("git_commit"),
        "merge_boundary": release_manifest.get("merge_boundary"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate NOISE Gate 07 release-readiness manifest.")
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    release_manifest = json.loads(args.release_manifest.read_text(encoding="utf-8"))
    payload = evaluate_gate07(release_manifest)
    payload["source_release_manifest"] = args.release_manifest.as_posix()
    _write_json(args.output_root / "noise_gate07_manifest.json", payload)
    _write_report(args.output_root / "noise_gate07_report.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
