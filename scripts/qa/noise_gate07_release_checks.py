#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MappingLike = dict[str, Any]
_REQUIRED_EVIDENCE = {"synthetic_recovery", "predictive_checks", "emb_comparison"}
_REQUIRED_CONFIG_NAMES = {"legacy", "synthetic_recovery", "predictive_checks", "emb_comparison"}


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> MappingLike:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "## GitHub Checks",
        "",
        f"- status: `{payload.get('github_checks', {}).get('status')}`",
        "",
        "## Failures",
        "",
    ]
    lines.extend(f"- {failure}" for failure in payload["failures"] or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in payload["warnings"] or ["None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_path(path_value: Any, failures: list[str], label: str) -> None:
    if not isinstance(path_value, str) or not path_value:
        failures.append(f"{label} path is missing.")
        return
    if not Path(path_value).exists():
        failures.append(f"{label} path does not exist: {path_value}.")


def _github_check_summary(*, pr: str | None, repo: str | None, allow_missing: bool) -> tuple[MappingLike, tuple[str, ...], tuple[str, ...]]:
    if not pr:
        if allow_missing:
            return {"status": "skipped_by_explicit_flag", "checks": []}, (), ("GitHub checks were skipped by explicit CLI flag.",)
        return {"status": "not_checked", "checks": []}, ("GitHub PR checks were not inspected; pass --github-pr or --allow-missing-github-checks explicitly.",), ()
    command = ["gh", "pr", "checks", pr, "--json", "bucket,name,state,workflow"]
    if repo:
        command.extend(["--repo", repo])
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode not in {0, 8} and not completed.stdout.strip():
        return {"status": "error", "stderr": completed.stderr.strip(), "checks": []}, ("GitHub PR checks could not be read.",), ()
    checks = json.loads(completed.stdout or "[]")
    if not checks:
        return {"status": "no_checks", "checks": []}, ("GitHub reports no checks for the PR branch.",), ()
    blocking = [check for check in checks if check.get("bucket") in {"fail", "pending", "cancel"}]
    status = "passed" if not blocking else "blocked"
    failures = tuple(f"GitHub check {check.get('name')} is {check.get('bucket')} ({check.get('state')})." for check in blocking)
    return {"status": status, "checks": checks}, failures, ()


def evaluate_gate07(
    release_manifest: MappingLike,
    *,
    github_checks: MappingLike | None = None,
    github_failures: tuple[str, ...] = (),
    github_warnings: tuple[str, ...] = (),
) -> MappingLike:
    failures: list[str] = list(github_failures)
    warnings: list[str] = list(github_warnings)
    gate07 = release_manifest.get("gate07")
    if not isinstance(gate07, dict) or gate07.get("pass") is not True:
        failures.append("release manifest gate07.pass is not true.")

    config_validation = release_manifest.get("config_validation")
    if not isinstance(config_validation, dict) or config_validation.get("passed") is not True:
        failures.append("config validation did not pass.")
    config_results = config_validation.get("results", []) if isinstance(config_validation, dict) else []
    if not isinstance(config_results, list) or not config_results:
        failures.append("config validation results are missing.")
    else:
        failed_configs = [result.get("path") for result in config_results if result.get("passed") is not True]
        if failed_configs:
            failures.append(f"config validation contains failed configs: {failed_configs}.")
        config_names = {result.get("config_name") for result in config_results}
        missing_configs = sorted(_REQUIRED_CONFIG_NAMES - config_names)
        if missing_configs:
            failures.append(f"config validation is missing required config names: {missing_configs}.")

    artifact_index = release_manifest.get("artifact_index")
    entries = artifact_index.get("entries") if isinstance(artifact_index, dict) else None
    if not isinstance(entries, dict) or not entries:
        failures.append("artifact index is missing or empty.")
        entries = {}
    missing_entries = sorted(_REQUIRED_EVIDENCE - set(entries))
    if missing_entries:
        failures.append(f"artifact index is missing required evidence entries: {missing_entries}.")
    for label in sorted(_REQUIRED_EVIDENCE & set(entries)):
        entry = entries[label]
        if entry.get("exists") is not True:
            failures.append(f"evidence entry {label} does not exist.")
        _check_path(entry.get("path"), failures, f"evidence entry {label}")
        if entry.get("all_scenarios_passed") is not True:
            failures.append(f"evidence entry {label} does not report all_scenarios_passed=true.")
        statuses = entry.get("scenario_gate_statuses")
        if not isinstance(statuses, dict) or not statuses:
            failures.append(f"evidence entry {label} does not record per-scenario gate statuses.")
        elif any(status != "pass" for status in statuses.values()):
            failures.append(f"evidence entry {label} has non-pass scenario gate statuses: {statuses}.")
        if entry.get("git_status_clean") is not True:
            failures.append(f"evidence entry {label} does not record git_status_clean=true.")
        if not entry.get("git_commit"):
            failures.append(f"evidence entry {label} does not record a git commit.")
        if not entry.get("commands"):
            failures.append(f"evidence entry {label} does not record regeneration commands.")
        missing_sidecars = [name for name, exists in (entry.get("artifact_existence") or {}).items() if exists is not True]
        if missing_sidecars:
            failures.append(f"evidence entry {label} has missing artifact sidecars: {missing_sidecars}.")

    gate06_path = release_manifest.get("gate06_manifest")
    _check_path(gate06_path, failures, "Gate06 manifest")
    if isinstance(gate06_path, str) and Path(gate06_path).exists():
        gate06_manifest = _load_json(Path(gate06_path))
        if gate06_manifest.get("pass") is not True:
            failures.append("Gate06 manifest does not pass.")

    for artifact_name, artifact_path in (release_manifest.get("artifacts") or {}).items():
        _check_path(artifact_path, failures, f"release artifact {artifact_name}")

    provenance = release_manifest.get("provenance", {})
    if provenance.get("git_status_clean") is not True:
        failures.append("release manifest provenance does not record git_status_clean=true.")
    if not provenance.get("git_commit"):
        failures.append("release manifest provenance is missing git_commit.")

    if release_manifest.get("merge_boundary") not in {"merged", "human_review_required"}:
        failures.append("release manifest must record merged or human_review_required merge boundary.")
    if release_manifest.get("merge_boundary") == "human_review_required":
        warnings.append("PR stack is at a human review/merge boundary.")
    confirmation = release_manifest.get("karolina_interaction_confirmation")
    if release_manifest.get("no_karolina_interaction") is not True:
        failures.append("release manifest must record no active Karolina interaction.")
    if not isinstance(confirmation, dict) or confirmation.get("operator_confirmed") is not True:
        failures.append("release manifest is missing explicit Karolina non-interaction confirmation.")

    return {
        "schema_version": 1,
        "gate": "NOISE Gate 07 Release checks",
        "pass": not failures,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "warnings": warnings,
        "source_commit": provenance.get("git_commit"),
        "merge_boundary": release_manifest.get("merge_boundary"),
        "github_checks": github_checks or {"status": "not_checked", "checks": []},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate NOISE Gate 07 release-readiness manifest.")
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--github-pr", help="GitHub PR number, URL, or branch to verify checks for.")
    parser.add_argument("--repo", default="BrieucB/MesoUQ", help="GitHub repository for --github-pr checks.")
    parser.add_argument("--allow-missing-github-checks", action="store_true", help="Explicitly allow local-only Gate07 validation without GitHub checks.")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    release_manifest = _load_json(args.release_manifest)
    github_checks, github_failures, github_warnings = _github_check_summary(pr=args.github_pr, repo=args.repo, allow_missing=args.allow_missing_github_checks)
    payload = evaluate_gate07(release_manifest, github_checks=github_checks, github_failures=github_failures, github_warnings=github_warnings)
    payload["source_release_manifest"] = args.release_manifest.as_posix()
    _write_json(args.output_root / "noise_gate07_manifest.json", payload)
    _write_report(args.output_root / "noise_gate07_report.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
