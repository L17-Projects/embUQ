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
_REQUIRED_RELEASE_ARTIFACTS = {
    "artifact_index",
    "config_validation",
    "release_manifest",
    "release_report",
}
_REQUIRED_ARTIFACT_KEYS = {
    "synthetic_recovery": {
        "metrics",
        "summary_csv",
        "covariance_heatmap",
        "recovery_parameter_intervals",
        "residual_whitened_hist",
        "synthetic_observable_overlay",
    },
    "predictive_checks": {
        "metrics",
        "summary_csv",
        "calibration_summary",
        "ppc_observable_overlay",
        "ppc_summary_intervals",
        "sbc_rank_histogram",
    },
    "emb_comparison": {
        "metrics",
        "summary_csv",
        "gate06_summary",
        "report_md",
        "metrics_table",
        "posterior_intervals",
        "predictive_bands",
        "residual_diagnostics",
    },
}


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> MappingLike:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: MappingLike) -> None:
    github_checks = payload.get("github_checks", {})
    workflow_runs = github_checks.get("workflow_runs") or []
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
        f"- status: `{github_checks.get('status')}`",
        "",
    ]
    pr_checks = github_checks.get("pr_checks")
    if isinstance(pr_checks, dict) and pr_checks:
        lines.append(f"- PR checks: `{pr_checks.get('status')}`")
    pr_head = github_checks.get("pr_head")
    if isinstance(pr_head, dict) and pr_head:
        lines.append(f"- PR head: `{pr_head.get('headRefOid')}`")
    if workflow_runs:
        lines.extend(["", "### Workflow Runs", ""])
        for run in workflow_runs:
            lines.append(
                f"- `{run.get('id')}` {run.get('name')}: `{run.get('status')}` / "
                f"`{run.get('conclusion')}` at `{run.get('headSha')}`"
            )
    lines.extend(["", "## Failures", ""])
    lines.extend(f"- {failure}" for failure in payload["failures"] or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in payload["warnings"] or ["None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _resolve_release_path(base_dir: Path | None, path_value: Any) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _check_path(path_value: Any, failures: list[str], label: str, *, base_dir: Path | None = None) -> Path | None:
    path = _resolve_release_path(base_dir, path_value)
    if path is None:
        failures.append(f"{label} path is missing.")
        return None
    if not path.exists():
        failures.append(f"{label} path does not exist: {path_value}.")
    return path


def _resolve_manifest_artifact(manifest_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _validate_evidence_manifest(label: str, manifest_path: Path, entry: MappingLike, failures: list[str]) -> None:
    try:
        evidence_manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"evidence manifest {label} could not be read: {exc}.")
        return

    artifacts = evidence_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        failures.append(f"evidence manifest {label} does not record artifact sidecars.")
        artifacts = {}
    missing_required = sorted(_REQUIRED_ARTIFACT_KEYS[label] - set(artifacts))
    if missing_required:
        failures.append(f"evidence manifest {label} is missing required artifact sidecars: {missing_required}.")
    for artifact_name in sorted(_REQUIRED_ARTIFACT_KEYS[label] & set(artifacts)):
        artifact_path = _resolve_manifest_artifact(manifest_path, artifacts.get(artifact_name))
        if artifact_path is None:
            failures.append(f"evidence manifest {label} artifact {artifact_name} path is missing.")
        elif not artifact_path.exists():
            failures.append(f"evidence manifest {label} artifact {artifact_name} path does not exist: {artifact_path}.")

    metrics_path = _resolve_manifest_artifact(manifest_path, artifacts.get("metrics"))
    metrics: MappingLike = {}
    if metrics_path is None or not metrics_path.exists():
        failures.append(f"evidence manifest {label} metrics artifact is missing.")
    else:
        try:
            metrics = _load_json(metrics_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"evidence manifest {label} metrics could not be read: {exc}.")
            metrics = {}

    if metrics.get("all_scenarios_passed") is not True:
        failures.append(f"evidence manifest {label} metrics does not report all_scenarios_passed=true.")
    statuses = metrics.get("scenario_gate_statuses")
    if not isinstance(statuses, dict) or not statuses:
        failures.append(f"evidence manifest {label} metrics does not record per-scenario gate statuses.")
    elif any(status != "pass" for status in statuses.values()):
        failures.append(f"evidence manifest {label} metrics has non-pass scenario statuses: {statuses}.")

    configs = evidence_manifest.get("configs", evidence_manifest.get("config"))
    if not configs:
        failures.append(f"evidence manifest {label} does not record config references.")
    residual_risk = (
        evidence_manifest.get("residual_risk")
        or evidence_manifest.get("residual_risk_notes")
        or evidence_manifest.get("known_limitations")
        or metrics.get("residual_risk_notes")
        or metrics.get("known_limitations")
    )
    if not residual_risk:
        failures.append(f"evidence manifest {label} does not record residual risk or limitations.")
    provenance = evidence_manifest.get("provenance") if isinstance(evidence_manifest.get("provenance"), dict) else {}
    if provenance.get("git_status_clean") is not True:
        failures.append(f"evidence manifest {label} does not record git_status_clean=true.")
    if not provenance.get("git_commit"):
        failures.append(f"evidence manifest {label} does not record a git commit.")
    if not evidence_manifest.get("commands"):
        failures.append(f"evidence manifest {label} does not record regeneration commands.")

    artifact_existence = entry.get("artifact_existence")
    if not isinstance(artifact_existence, dict) or not artifact_existence:
        failures.append(f"artifact index entry {label} does not record artifact sidecar existence.")
        artifact_existence = {}
    missing_index_keys = sorted(_REQUIRED_ARTIFACT_KEYS[label] - set(artifact_existence))
    if missing_index_keys:
        failures.append(f"artifact index entry {label} is missing required artifact sidecars: {missing_index_keys}.")
    missing_index_artifacts = [name for name, exists in artifact_existence.items() if exists is not True]
    if missing_index_artifacts:
        failures.append(f"artifact index entry {label} has missing artifact sidecars: {missing_index_artifacts}.")


def _github_pr_head_summary(*, pr: str, repo: str | None, expected_commit: str | None) -> tuple[MappingLike, tuple[str, ...]]:
    command = ["gh", "pr", "view", pr, "--json", "headRefOid,headRefName,url"]
    if repo:
        command.extend(["--repo", repo])
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"status": "error", "stderr": completed.stderr.strip()}, ("GitHub PR head commit could not be read.",)
    head = json.loads(completed.stdout)
    failures: list[str] = []
    if expected_commit and head.get("headRefOid") != expected_commit:
        failures.append(f"GitHub PR head SHA {head.get('headRefOid')} does not match release manifest commit {expected_commit}.")
    head["status"] = "passed" if not failures else "blocked"
    return head, tuple(failures)


def _github_check_summary(
    *,
    pr: str | None,
    repo: str | None,
    allow_missing: bool,
    allow_no_checks: bool = False,
) -> tuple[MappingLike, tuple[str, ...], tuple[str, ...]]:
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
        if allow_no_checks:
            return {"status": "no_pr_checks", "checks": []}, (), ("GitHub reports no PR checks; explicit workflow run evidence is being used.",)
        return {"status": "no_checks", "checks": []}, ("GitHub reports no checks for the PR branch.",), ()
    blocking = [check for check in checks if check.get("bucket") in {"fail", "pending", "cancel"}]
    status = "passed" if not blocking else "blocked"
    failures = tuple(f"GitHub check {check.get('name')} is {check.get('bucket')} ({check.get('state')})." for check in blocking)
    return {"status": status, "checks": checks}, failures, ()


def _github_run_summary(
    *,
    run_ids: list[str],
    repo: str | None,
    expected_commit: str | None,
) -> tuple[MappingLike, tuple[str, ...], tuple[str, ...]]:
    runs: list[MappingLike] = []
    failures: list[str] = []
    for run_id in run_ids:
        command = [
            "gh",
            "run",
            "view",
            run_id,
            "--json",
            "status,conclusion,headSha,headBranch,name,url,jobs",
        ]
        if repo:
            command.extend(["--repo", repo])
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode != 0 or not completed.stdout.strip():
            failures.append(f"GitHub workflow run {run_id} could not be read.")
            runs.append({"id": run_id, "status": "error", "stderr": completed.stderr.strip()})
            continue
        run = json.loads(completed.stdout)
        run["id"] = run_id
        runs.append(run)
        if expected_commit and run.get("headSha") != expected_commit:
            failures.append(
                f"GitHub workflow run {run_id} is for head SHA {run.get('headSha')}, expected {expected_commit}."
            )
        if run.get("status") != "completed":
            failures.append(f"GitHub workflow run {run_id} is {run.get('status')}, not completed.")
        if run.get("conclusion") != "success":
            failures.append(f"GitHub workflow run {run_id} concluded {run.get('conclusion')}, not success.")
        failed_jobs = [
            job.get("name")
            for job in run.get("jobs", [])
            if job.get("status") == "completed" and job.get("conclusion") not in {"success", "skipped"}
        ]
        if failed_jobs:
            failures.append(f"GitHub workflow run {run_id} has failed jobs: {failed_jobs}.")
    status = "passed" if runs and not failures else "blocked"
    return {"status": status, "runs": runs}, tuple(failures), ()


def _github_evidence_summary(
    *,
    pr: str | None,
    run_ids: list[str],
    repo: str | None,
    allow_missing: bool,
    expected_commit: str | None,
) -> tuple[MappingLike, tuple[str, ...], tuple[str, ...]]:
    failures: list[str] = []
    warnings: list[str] = []
    pr_summary: MappingLike = {"status": "not_requested", "checks": []}
    run_summary: MappingLike = {"status": "not_requested", "runs": []}

    pr_head: MappingLike = {}
    if pr:
        pr_head, pr_head_failures = _github_pr_head_summary(pr=pr, repo=repo, expected_commit=expected_commit)
        failures.extend(pr_head_failures)
        pr_summary, pr_failures, pr_warnings = _github_check_summary(
            pr=pr,
            repo=repo,
            allow_missing=allow_missing,
            allow_no_checks=bool(run_ids),
        )
        failures.extend(pr_failures)
        warnings.extend(pr_warnings)
    if run_ids:
        run_summary, run_failures, run_warnings = _github_run_summary(
            run_ids=run_ids,
            repo=repo,
            expected_commit=expected_commit,
        )
        failures.extend(run_failures)
        warnings.extend(run_warnings)
    elif not pr:
        if allow_missing:
            warnings.append("GitHub checks were skipped by explicit CLI flag.")
        else:
            failures.append(
                "GitHub evidence was not inspected; pass --github-pr, --github-run-id, or --allow-missing-github-checks explicitly."
            )

    if allow_missing and not pr and not run_ids:
        status = "skipped_by_explicit_flag"
    else:
        status = "passed" if not failures else "blocked"
    return {
        "status": status,
        "pr_checks": pr_summary,
        "pr_head": pr_head,
        "workflow_runs": run_summary.get("runs", []),
    }, tuple(failures), tuple(warnings)

def evaluate_gate07(
    release_manifest: MappingLike,
    *,
    release_manifest_path: Path | None = None,
    github_checks: MappingLike | None = None,
    github_failures: tuple[str, ...] = (),
    github_warnings: tuple[str, ...] = (),
) -> MappingLike:
    failures: list[str] = list(github_failures)
    warnings: list[str] = list(github_warnings)
    release_base_dir = release_manifest_path.parent if release_manifest_path is not None else None
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
        entry_path = _check_path(entry.get("path"), failures, f"evidence entry {label}", base_dir=release_base_dir)
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
        if not entry.get("configs"):
            failures.append(f"evidence entry {label} does not record config references.")
        if not entry.get("residual_risk"):
            failures.append(f"evidence entry {label} does not record residual risk or limitations.")
        if entry_path is not None and entry_path.exists():
            _validate_evidence_manifest(label, entry_path, entry, failures)

    gate06_path = release_manifest.get("gate06_manifest")
    resolved_gate06_path = _check_path(gate06_path, failures, "Gate06 manifest", base_dir=release_base_dir)
    if resolved_gate06_path is not None and resolved_gate06_path.exists():
        gate06_manifest = _load_json(resolved_gate06_path)
        if gate06_manifest.get("pass") is not True:
            failures.append("Gate06 manifest does not pass.")

    release_artifacts = release_manifest.get("artifacts")
    if not isinstance(release_artifacts, dict) or not release_artifacts:
        failures.append("release manifest artifacts are missing.")
        release_artifacts = {}
    missing_release_artifacts = sorted(_REQUIRED_RELEASE_ARTIFACTS - set(release_artifacts))
    if missing_release_artifacts:
        failures.append(f"release manifest artifacts are missing required entries: {missing_release_artifacts}.")
    for artifact_name, artifact_path in release_artifacts.items():
        _check_path(artifact_path, failures, f"release artifact {artifact_name}", base_dir=release_base_dir)

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
    parser.add_argument("--repo", default="BrieucB/MesoUQ", help="GitHub repository for GitHub checks or workflow runs.")
    parser.add_argument("--github-run-id", action="append", default=[], help="GitHub Actions workflow run ID to verify as explicit release evidence; may be repeated.")
    parser.add_argument("--allow-missing-github-checks", action="store_true", help="Explicitly allow local-only Gate07 validation without GitHub checks.")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    release_manifest = _load_json(args.release_manifest)
    expected_commit = release_manifest.get("provenance", {}).get("git_commit")
    github_checks, github_failures, github_warnings = _github_evidence_summary(
        pr=args.github_pr,
        run_ids=args.github_run_id,
        repo=args.repo,
        allow_missing=args.allow_missing_github_checks,
        expected_commit=expected_commit,
    )
    payload = evaluate_gate07(
        release_manifest,
        release_manifest_path=args.release_manifest,
        github_checks=github_checks,
        github_failures=github_failures,
        github_warnings=github_warnings,
    )
    payload["source_release_manifest"] = args.release_manifest.as_posix()
    _write_json(args.output_root / "noise_gate07_manifest.json", payload)
    _write_report(args.output_root / "noise_gate07_report.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
