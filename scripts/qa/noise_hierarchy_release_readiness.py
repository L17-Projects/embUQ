#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from meso_uq.noise.release_readiness import (
    build_noise_artifact_index,
    evaluate_release_gate,
    resolve_noise_mode,
    supported_noise_modes,
    validate_noise_config_file,
)


MappingLike = dict[str, Any]


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(("git", *args), check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _provenance(start_time: float) -> MappingLike:
    status = _git_output("status", "--short") or ""
    return {
        "argv": list(sys.argv),
        "command_line": " ".join(sys.argv),
        "cwd": Path.cwd().as_posix(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status_clean": status == "",
        "git_status_short": status,
        "runtime_seconds": float(time.perf_counter() - start_time),
        "karolina_interaction": "none",
    }


def _load_json(path: Path) -> MappingLike:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_template(mode: str) -> str:
    if mode == "legacy":
        return "legacy"
    if mode == "full_hierarchy":
        return "full"
    return "staged"


def _write_markdown_report(path: Path, manifest: MappingLike) -> None:
    lines = [
        "# Noise Hierarchy Release Readiness",
        "",
        f"Status: {'pass' if manifest['gate07']['pass'] else 'fail'}",
        "",
        f"Commit: `{manifest['provenance']['git_commit']}`",
        "",
        f"Command: `{manifest['provenance']['command_line']}`",
        "",
        f"Report template: `{manifest['report_template']}`",
        "",
        f"Merge boundary: `{manifest['merge_boundary']}`",
        "",
        "## Mode Surface",
        "",
        "| mode | stage | components |",
        "| --- | --- | --- |",
    ]
    for mode in manifest["modes"]:
        lines.append(f"| {mode['mode']} | {mode['stage']} | {', '.join(mode['components'])} |")
    lines.extend(["", "## Config Validation", "", "| config | status | warnings | errors |", "| --- | --- | --- | --- |"])
    for result in manifest["config_validation"]["results"]:
        status = "pass" if result["passed"] else "fail"
        warnings = "; ".join(result["warnings"]) or "None"
        errors = "; ".join(result["errors"]) or "None"
        lines.append(f"| `{result['path']}` | {status} | {warnings} | {errors} |")
    lines.extend(["", "## Evidence", ""])
    for label, entry in manifest["artifact_index"]["entries"].items():
        lines.append(f"- `{label}`: `{entry.get('path')}`; all_scenarios_passed={entry.get('all_scenarios_passed')}")
        if entry.get("commands"):
            lines.append(f"  - commands: `{json.dumps(entry['commands'], sort_keys=True)}`")
        if entry.get("metric_summary"):
            lines.append(f"  - metrics: `{json.dumps(entry['metric_summary'], sort_keys=True)}`")
        if entry.get("residual_risk"):
            lines.append(f"  - residual risk: {entry['residual_risk']}")
        for artifact_name, artifact_path in sorted((entry.get("artifacts") or {}).items()):
            if isinstance(artifact_path, str):
                exists = (entry.get("artifact_existence") or {}).get(artifact_name)
                lines.append(f"  - `{artifact_name}`: `{artifact_path}`; exists={exists}")
    lines.extend(["", "## Sidecars", ""])
    for artifact_name, artifact_path in sorted(manifest["artifacts"].items()):
        lines.append(f"- `{artifact_name}`: `{artifact_path}`")
    lines.extend(
        [
            "",
            "## Migration Notes",
            "",
            "- Legacy mode remains explicitly selectable and preserves the M1 compatibility wrappers.",
            "- Staged modes expose M2-M7 components without changing legacy defaults implicitly.",
            "- Full hierarchy mode resolves the shared M5 covariance assembly used by downstream validation diagnostics.",
            "- This release-readiness report records human review/merge boundaries instead of claiming merged PR state early.",
        ]
    )
    lines.extend(["", "## Gate 07", ""])
    lines.extend(f"- failure: {failure}" for failure in manifest["gate07"]["failures"] or ["None"])
    lines.extend(f"- warning: {warning}" for warning in manifest["gate07"]["warnings"] or ["None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Validate noise hierarchy configs and assemble release-readiness artifacts.")
    parser.add_argument("--mode", default="full_hierarchy", help="Noise hierarchy mode to resolve for CLI smoke evidence.")
    parser.add_argument("--config", action="append", type=Path, help="Noise config file to validate. Defaults to configs/noise/*.example.yaml.")
    parser.add_argument("--synthetic-manifest", type=Path)
    parser.add_argument("--predictive-manifest", type=Path)
    parser.add_argument("--emb-manifest", type=Path)
    parser.add_argument("--gate06-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--merge-boundary", choices=("merged", "human_review_required", "blocked"), default="human_review_required")
    parser.add_argument(
        "--confirm-no-karolina-interaction",
        action="store_true",
        help="Required operator confirmation that this validation did not touch active Karolina worktrees or sessions.",
    )
    args = parser.parse_args(argv)
    if not args.confirm_no_karolina_interaction:
        parser.error("--confirm-no-karolina-interaction is required for release-readiness evidence.")
    args.output_root.mkdir(parents=True, exist_ok=True)

    config_paths = args.config or sorted(Path("configs/noise").glob("*.example.yaml"))
    try:
        selected_mode = resolve_noise_mode(args.mode).mode
    except ValueError as exc:
        parser.error(str(exc))
    config_results = [validate_noise_config_file(path) for path in config_paths]
    mode_names = (selected_mode,) + tuple(mode for mode in supported_noise_modes() if mode != selected_mode)
    mode_specs = [resolve_noise_mode(mode).as_dict() for mode in mode_names]
    manifests = {
        "synthetic_recovery": args.synthetic_manifest,
        "predictive_checks": args.predictive_manifest,
        "emb_comparison": args.emb_manifest,
    }
    provided_manifests = {label: path for label, path in manifests.items() if path is not None}
    artifact_index = build_noise_artifact_index(provided_manifests)
    gate06_manifest = _load_json(args.gate06_manifest) if args.gate06_manifest else {"pass": False}
    gate07 = evaluate_release_gate(
        config_results=config_results,
        artifact_index=artifact_index,
        gate06_manifest=gate06_manifest,
        merge_boundary=args.merge_boundary,
        no_karolina_interaction=args.confirm_no_karolina_interaction,
    )
    config_validation = {
        "passed": all(result.passed for result in config_results),
        "results": [result.as_dict() for result in config_results],
        "config_paths": [path.as_posix() for path in config_paths],
    }
    artifact_index_path = args.output_root / "noise_artifact_index.json"
    config_validation_path = args.output_root / "noise_config_validation.json"
    manifest_path = args.output_root / "noise_release_readiness_manifest.json"
    report_path = args.output_root / "noise_release_readiness_report.md"
    manifest: MappingLike = {
        "schema_version": 1,
        "description": "M8 noise hierarchy release-readiness manifest for MES-38/MES-39/MES-47.",
        "provenance": _provenance(start_time),
        "commands": {
            "executed_argv": list(sys.argv),
            "regenerate": "python scripts/qa/noise_hierarchy_release_readiness.py --output-root <output-root> --confirm-no-karolina-interaction ...",
            "gate07": "python scripts/qa/noise_gate07_release_checks.py --release-manifest <manifest> --output-root <gate-root> --github-pr <PR number> --repo BrieucB/MesoUQ",
        },
        "modes": mode_specs,
        "selected_mode": selected_mode,
        "report_template": _report_template(selected_mode),
        "config_validation": config_validation,
        "artifact_index": artifact_index,
        "artifacts": {
            "artifact_index": artifact_index_path.as_posix(),
            "config_validation": config_validation_path.as_posix(),
            "release_manifest": manifest_path.as_posix(),
            "release_report": report_path.as_posix(),
        },
        "merge_boundary": args.merge_boundary,
        "no_karolina_interaction": args.confirm_no_karolina_interaction,
        "karolina_interaction_confirmation": {
            "operator_confirmed": args.confirm_no_karolina_interaction,
            "scope": "No active Karolina worktree or running Karolina session was touched for this evidence run.",
        },
        "gate06_manifest": None if args.gate06_manifest is None else args.gate06_manifest.as_posix(),
        "gate07": gate07.as_dict(),
        "skips": [],
        "residual_risk": "Required PRs are stacked and recorded at a human review/merge boundary until reviewer approval and merge.",
    }
    _write_json(artifact_index_path, artifact_index)
    _write_json(config_validation_path, config_validation)
    _write_json(manifest_path, manifest)
    _write_markdown_report(report_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if gate07.passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
