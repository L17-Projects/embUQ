from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = REPO_ROOT / "configs" / "audit_polishing"
ALLOWED_ORPHAN_CLASSIFICATIONS = {
    "keep",
    "document",
    "deprecate",
    "delete",
    "false_positive",
    "needs_owner_review",
}
ALLOWED_GATE_STATUSES = {"advisory", "deferred", "enforced"}


def _load_json(relative_path: str) -> dict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def test_orphan_triage_manifest_classifies_baseline_without_deletion() -> None:
    manifest = _load_json("configs/audit_polishing/orphan_python_triage.example.json")

    assert manifest["schema_version"] == "1.0"
    assert manifest["source"]["linear_issue"] == "MES-264"
    assert manifest["summary"]["candidate_count"] == sum(
        group["candidate_count"] for group in manifest["groups"]
    )
    assert manifest["summary"]["candidate_count"] == 97
    assert manifest["classification_policy"]["delete_requires_owner_confirmation"] is True
    assert manifest["classification_policy"]["delete_requires_focused_tests"] is True

    groups_by_subsystem = {group["subsystem"]: group for group in manifest["groups"]}
    assert groups_by_subsystem["scripts/platforms/karolina"]["recommended_action"] == (
        "preserve-until-platform-owner-review"
    )
    assert groups_by_subsystem["scripts/platforms/vega"]["recommended_action"] == (
        "preserve-until-platform-owner-review"
    )
    assert groups_by_subsystem["scripts/workflows/emb"]["recommended_action"] == (
        "preserve-active-learning-and-platform-entrypoints"
    )

    for group in manifest["groups"]:
        assert group["classification"] in ALLOWED_ORPHAN_CLASSIFICATIONS
        assert group["recommended_action"] != "delete"
        assert group["owner"]


def test_large_tracked_payload_manifest_has_existing_owned_paths() -> None:
    manifest = _load_json("configs/audit_polishing/large_tracked_payloads.example.json")

    assert manifest["schema_version"] == "1.0"
    assert manifest["source"]["linear_issue"] == "MES-265"
    assert manifest["policy"]["generated_root_candidates_found"] == 0
    assert manifest["policy"]["removal_requires_reproducibility_plan"] is True

    payloads = manifest["tracked_payloads"]
    assert payloads
    assert payloads[0]["path"] == "emb/indentation/surrogate/diameters/3.4um/data/samples_all.dat"
    assert payloads[0]["bytes"] > 20_000_000

    for payload in payloads:
        path = REPO_ROOT / payload["path"]
        assert path.is_file(), payload["path"]
        assert path.stat().st_size == payload["bytes"]
        assert payload["owner"]
        assert payload["retention_policy"]
        assert payload["storage_location"] in {"source_tree", "generated_root", "hpc_output", "external"}
        assert payload["status"] in {"owned", "needs_owner_review", "externalization_candidate"}


def test_static_quality_policy_has_explicit_gate_decisions() -> None:
    manifest = _load_json("configs/audit_polishing/static_quality_policy.example.json")

    assert manifest["schema_version"] == "1.0"
    assert manifest["source"]["linear_issue"] == "MES-266"
    gates = {gate["name"]: gate for gate in manifest["gates"]}

    assert gates["docs-link-check"]["status"] == "enforced"
    assert gates["governance-structural-tests"]["status"] == "enforced"
    assert gates["workflow-action-pin-policy"]["status"] == "enforced"
    assert gates["ruff"]["status"] == "deferred"
    assert gates["mypy"]["status"] == "deferred"
    assert gates["pip-audit"]["status"] == "advisory"

    for gate in gates.values():
        assert gate["status"] in ALLOWED_GATE_STATUSES
        assert gate["owner"]
        assert gate["review_condition"]
        if gate["status"] == "enforced":
            assert gate["command"]


def test_audit_polishing_policy_is_linked_from_docs() -> None:
    docs_index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    artifact_policy = (REPO_ROOT / "docs" / "ARTIFACT_POLICY.md").read_text(encoding="utf-8")
    policy = (REPO_ROOT / "docs" / "AUDIT_POLISHING_POLICY.md").read_text(encoding="utf-8")

    assert "AUDIT_POLISHING_POLICY.md" in docs_index
    assert "configs/audit_polishing/large_tracked_payloads.example.json" in artifact_policy
    assert "Dependabot GitHub Actions PRs" in policy
    assert "needs_owner_review" in policy
