#!/usr/bin/env python3
"""
Validate a machine-readable release evidence manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_STATUSES = {"draft", "ready", "blocked"}
REQUIRED_REPORT_KEYS = {
    "vega_validation_matrix",
    "vega_acceptance",
    "vega_production_sanity",
    "workstation_acceptance",
}
REQUIRED_DOCUMENT_KEYS = {
    "release_notes",
    "validation_matrix",
    "workstation_acceptance_checklist",
    "security_policy",
}


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _resolve_local_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_manifest(manifest: dict[str, Any], must_exist: bool = False) -> list[str]:
    errors: list[str] = []

    release = manifest.get("release")
    if not isinstance(release, str) or not release:
        errors.append("manifest.release must be a non-empty string")

    status = manifest.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append(f"manifest.status must be one of {sorted(ALLOWED_STATUSES)}")

    candidate_commit = manifest.get("candidate_commit")
    if not isinstance(candidate_commit, str) or not candidate_commit:
        errors.append("manifest.candidate_commit must be a non-empty string")

    claims = manifest.get("claims")
    if not isinstance(claims, dict):
        errors.append("manifest.claims must be a mapping")
    else:
        native_cuda_claim = claims.get("native_cuda_phase2_public_claim")
        if not isinstance(native_cuda_claim, bool):
            errors.append("manifest.claims.native_cuda_phase2_public_claim must be a boolean")
        elif release == "v0.1.0" and native_cuda_claim is not False:
            errors.append("v0.1.0 must keep native_cuda_phase2_public_claim set to false")

    github_ci = manifest.get("github_ci")
    if not isinstance(github_ci, dict):
        errors.append("manifest.github_ci must be a mapping")
    else:
        for key in ("ci_workflow_url", "release_smoke_url"):
            value = github_ci.get(key)
            if not isinstance(value, str) or not value:
                errors.append(f"manifest.github_ci.{key} must be a non-empty string")

    reports = manifest.get("reports")
    if not isinstance(reports, dict):
        errors.append("manifest.reports must be a mapping")
    else:
        missing_report_keys = sorted(REQUIRED_REPORT_KEYS - set(reports))
        if missing_report_keys:
            errors.append(f"manifest.reports is missing keys: {missing_report_keys}")
        for key in REQUIRED_REPORT_KEYS & set(reports):
            value = reports[key]
            if not isinstance(value, str) or not value:
                errors.append(f"manifest.reports.{key} must be a non-empty string")
            elif must_exist and not _looks_like_url(value) and not _resolve_local_path(value).exists():
                errors.append(f"missing report path: {_resolve_local_path(value)}")

    documents = manifest.get("documents")
    if not isinstance(documents, dict):
        errors.append("manifest.documents must be a mapping")
    else:
        missing_document_keys = sorted(REQUIRED_DOCUMENT_KEYS - set(documents))
        if missing_document_keys:
            errors.append(f"manifest.documents is missing keys: {missing_document_keys}")
        for key in REQUIRED_DOCUMENT_KEYS & set(documents):
            value = documents[key]
            if not isinstance(value, str) or not value:
                errors.append(f"manifest.documents.{key} must be a non-empty string")
            elif must_exist and not _looks_like_url(value) and not _resolve_local_path(value).exists():
                errors.append(f"missing document path: {_resolve_local_path(value)}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a release evidence manifest.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--must-exist", action="store_true", help="Require referenced report and document paths to exist locally.")
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.report)
    errors = validate_manifest(manifest, must_exist=args.must_exist)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated release evidence manifest: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
