from pathlib import Path

import yaml


def _load_workflow(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    workflow_path = repo_root / ".github" / "workflows" / name
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


def _uses_by_step(workflow):
    mapping = {}
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                mapping[step["name"]] = step["uses"]
    return mapping


def test_ci_workflow_has_concurrency_timeouts_and_canary_artifacts():
    workflow = _load_workflow("ci.yml")

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "github.workflow" in workflow["concurrency"]["group"]
    assert workflow["jobs"]["package-and-tests"]["permissions"] == {"contents": "read", "id-token": "write"}

    expected_timeouts = {
        "package-and-tests": 15,
        "docs": 5,
        "mpi-smoke": 10,
        "retraining-canary": 15,
        "workflow-canary": 20,
    }
    for job_name, timeout in expected_timeouts.items():
        assert workflow["jobs"][job_name]["timeout-minutes"] == timeout

    package_steps = workflow["jobs"]["package-and-tests"]["steps"]
    coverage_upload = next(step for step in package_steps if step["name"] == "Upload coverage artifacts")
    codecov_upload = next(step for step in package_steps if step["name"] == "Upload coverage to Codecov")
    assert coverage_upload["if"] == "always()"
    assert coverage_upload["uses"] == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    assert coverage_upload["with"]["name"] == "coverage-report"
    assert coverage_upload["with"]["retention-days"] == 14
    assert codecov_upload["uses"] == "codecov/codecov-action@57e3a136b779b570ffcdbf80b3bdc90e7fab3de2"
    assert codecov_upload["with"]["use_oidc"] is True
    assert codecov_upload["with"]["files"] == "coverage.xml"
    assert codecov_upload["with"]["disable_search"] is True
    assert codecov_upload["with"]["fail_ci_if_error"] is False

    workflow_steps = workflow["jobs"]["workflow-canary"]["steps"]
    workflow_summary = next(step for step in workflow_steps if step["name"] == "Summarize workflow canary outputs")
    workflow_upload = next(step for step in workflow_steps if step["name"] == "Upload workflow canary artifacts")
    assert workflow_summary["if"] == "always()"
    assert workflow_upload["if"] == "always()"
    assert workflow_upload["uses"] == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    assert workflow_upload["with"]["path"] == "_ci/workflow_canary"
    assert workflow_upload["with"]["retention-days"] == 14

    retraining_steps = workflow["jobs"]["retraining-canary"]["steps"]
    retraining_upload = next(step for step in retraining_steps if step["name"] == "Upload retraining canary artifacts")
    assert retraining_upload["if"] == "always()"
    assert retraining_upload["uses"] == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    assert retraining_upload["with"]["path"] == "_ci/surrogate_retraining"
    assert retraining_upload["with"]["retention-days"] == 14

    action_refs = set(_uses_by_step(workflow).values())
    assert action_refs == {
        "actions/cache@668228422ae6a00e4ad889ee87cd7109ec5666a7",
        "actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd",
        "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
        "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f",
        "codecov/codecov-action@57e3a136b779b570ffcdbf80b3bdc90e7fab3de2",
    }


def test_release_smoke_workflow_has_concurrency_timeouts_and_dist_artifact():
    workflow = _load_workflow("release-smoke.yml")

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "github.workflow" in workflow["concurrency"]["group"]
    assert workflow["jobs"]["release-smoke"]["timeout-minutes"] == 15
    assert workflow["jobs"]["docs-link-check"]["timeout-minutes"] == 5

    release_steps = workflow["jobs"]["release-smoke"]["steps"]
    release_upload = next(step for step in release_steps if step["name"] == "Upload release smoke dist artifacts")
    assert release_upload["if"] == "always()"
    assert release_upload["uses"] == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    assert release_upload["with"]["path"] == "dist"
    assert release_upload["with"]["retention-days"] == 14

    action_refs = set(_uses_by_step(workflow).values())
    assert action_refs == {
        "actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd",
        "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
        "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f",
    }
