#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MappingLike = dict[str, Any]


_ALLOWED_EVIDENCE_CLASSES = {"validation_fixture", "validation_pass", "production_pass"}


def _load_json(path: Path, label: str) -> MappingLike:
    if not path.exists():
        raise ValueError(f"{label} manifest does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} manifest is not valid JSON: {path}") from exc


def _resolve_artifact(manifest_path: Path, artifact: str) -> Path:
    candidate = Path(artifact)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return manifest_path.parent / candidate


def _load_metrics(manifest_path: Path, manifest: MappingLike, label: str) -> MappingLike:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts.get("metrics"):
        raise ValueError(f"{label} manifest must contain artifacts.metrics.")
    path = _resolve_artifact(manifest_path, str(artifacts["metrics"]))
    return _load_json(path, f"{label} metrics")


def _check_manifest(label: str, manifest_path: Path, require_clean_git: bool) -> tuple[list[str], list[str], MappingLike, MappingLike]:
    failures: list[str] = []
    warnings: list[str] = []
    manifest = _load_json(manifest_path, label)
    metrics = _load_metrics(manifest_path, manifest, label)
    if metrics.get("all_scenarios_passed") is not True:
        failures.append(f"{label} metrics do not report all_scenarios_passed=true.")
    statuses = metrics.get("scenario_gate_statuses")
    if not isinstance(statuses, dict) or not statuses:
        failures.append(f"{label} metrics must include non-empty scenario_gate_statuses.")
    elif any(status != "pass" for status in statuses.values()):
        failures.append(f"{label} has non-passing scenario gate statuses: {statuses}.")
    provenance = manifest.get("provenance", {})
    if require_clean_git and provenance.get("git_status_clean") is not True:
        failures.append(f"{label} manifest does not record git_status_clean=true.")
    for field in ("git_commit", "git_branch"):
        if provenance.get(field) in (None, ""):
            failures.append(f"{label} manifest provenance is missing {field}.")
    if not manifest.get("commands"):
        failures.append(f"{label} manifest is missing regeneration command metadata.")
    if not manifest.get("required_scenarios"):
        failures.append(f"{label} manifest is missing required_scenarios.")
    if not manifest.get("scenario_artifacts"):
        failures.append(f"{label} manifest is missing scenario_artifacts.")
    return failures, warnings, manifest, metrics


def evaluate_gate06(
    *,
    synthetic_manifest: Path,
    predictive_manifest: Path,
    emb_manifest: Path,
    require_production: bool = False,
    require_clean_git: bool = True,
) -> MappingLike:
    failures: list[str] = []
    warnings: list[str] = []
    payloads: dict[str, MappingLike] = {}
    for label, path in (
        ("synthetic_recovery", synthetic_manifest),
        ("predictive_checks", predictive_manifest),
        ("emb_comparison", emb_manifest),
    ):
        current_failures, current_warnings, manifest, metrics = _check_manifest(label, path, require_clean_git)
        failures.extend(current_failures)
        warnings.extend(current_warnings)
        payloads[label] = {"manifest": manifest, "metrics": metrics}

    emb = payloads.get("emb_comparison", {}).get("manifest", {})
    evidence_class = str(emb.get("evidence_class", "")).strip()
    production_claim = bool(emb.get("production_claim", False))
    if evidence_class not in _ALLOWED_EVIDENCE_CLASSES:
        failures.append(f"emb_comparison evidence_class must be one of {sorted(_ALLOWED_EVIDENCE_CLASSES)}, got {evidence_class!r}.")
    if require_production and evidence_class != "production_pass":
        failures.append("Gate 06 was run with --require-production, but EMB evidence is not production_pass.")
    if production_claim and evidence_class != "production_pass":
        failures.append("EMB comparison manifest claims production evidence without evidence_class=production_pass.")
    if evidence_class == "validation_fixture" and not production_claim:
        warnings.append("EMB evidence is validation_fixture with production_claim=false; acceptable for validation closeout, not a production posterior claim.")

    passed = not failures
    return {
        "schema_version": 1,
        "gate": "NOISE Gate 06 Integrated EMB comparison",
        "pass": passed,
        "status": "pass" if passed else "fail",
        "evidence_class": evidence_class,
        "production_claim": production_claim,
        "require_production": bool(require_production),
        "require_clean_git": bool(require_clean_git),
        "inputs": {
            "synthetic_manifest": synthetic_manifest.as_posix(),
            "predictive_manifest": predictive_manifest.as_posix(),
            "emb_manifest": emb_manifest.as_posix(),
        },
        "source_commits": {
            label: payload["manifest"].get("provenance", {}).get("git_commit")
            for label, payload in payloads.items()
        },
        "scenario_gate_statuses": {
            label: payload["metrics"].get("scenario_gate_statuses")
            for label, payload in payloads.items()
        },
        "failures": tuple(failures),
        "warnings": tuple(warnings),
    }


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, payload: MappingLike) -> None:
    lines = [
        "# NOISE Gate 06 Integrated EMB Comparison",
        "",
        f"Status: {payload['status']}",
        "",
        f"Evidence class: {payload['evidence_class']}",
        "",
        f"Production claim: {str(payload['production_claim']).lower()}",
        "",
        "## Inputs",
        "",
    ]
    for key, value in payload["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Failures", ""])
    lines.extend(f"- {failure}" for failure in payload["failures"] or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in payload["warnings"] or ["None"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate NOISE Gate 06 evidence manifests.")
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--predictive-manifest", type=Path, required=True)
    parser.add_argument("--emb-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-production", action="store_true")
    parser.add_argument("--allow-dirty-git", action="store_true")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = evaluate_gate06(
        synthetic_manifest=args.synthetic_manifest,
        predictive_manifest=args.predictive_manifest,
        emb_manifest=args.emb_manifest,
        require_production=args.require_production,
        require_clean_git=not args.allow_dirty_git,
    )
    _write_json(args.output_root / "noise_gate06_manifest.json", payload)
    _write_report(args.output_root / "noise_gate06_report.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
