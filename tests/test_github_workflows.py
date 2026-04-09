from pathlib import Path

import yaml


def _load_workflow(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    workflow_path = repo_root / ".github" / "workflows" / name
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


def test_ci_workflow_has_concurrency_timeouts_and_canary_artifacts():
    workflow = _load_workflow("ci.yml")

    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "github.workflow" in workflow["concurrency"]["group"]

    expected_timeouts = {
        "package-and-tests": 15,
        "docs": 5,
        "mpi-smoke": 10,
        "retraining-canary": 15,
        "workflow-canary": 20,
    }
    for job_name, timeout in expected_timeouts.items():
        assert workflow["jobs"][job_name]["timeout-minutes"] == timeout

    workflow_steps = workflow["jobs"]["workflow-canary"]["steps"]
    workflow_summary = next(step for step in workflow_steps if step["name"] == "Summarize workflow canary outputs")
    workflow_upload = next(step for step in workflow_steps if step["name"] == "Upload workflow canary artifacts")
    assert workflow_summary["if"] == "always()"
    assert workflow_upload["if"] == "always()"
    assert workflow_upload["uses"] == "actions/upload-artifact@v4"
    assert workflow_upload["with"]["path"] == "_ci/workflow_canary"
    assert workflow_upload["with"]["retention-days"] == 14

    retraining_steps = workflow["jobs"]["retraining-canary"]["steps"]
    retraining_upload = next(step for step in retraining_steps if step["name"] == "Upload retraining canary artifacts")
    assert retraining_upload["if"] == "always()"
    assert retraining_upload["uses"] == "actions/upload-artifact@v4"
    assert retraining_upload["with"]["path"] == "_ci/surrogate_retraining"
    assert retraining_upload["with"]["retention-days"] == 14


def test_release_smoke_workflow_has_concurrency_timeouts_and_dist_artifact():
    workflow = _load_workflow("release-smoke.yml")

    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "github.workflow" in workflow["concurrency"]["group"]
    assert workflow["jobs"]["release-smoke"]["timeout-minutes"] == 15
    assert workflow["jobs"]["docs-link-check"]["timeout-minutes"] == 5

    release_steps = workflow["jobs"]["release-smoke"]["steps"]
    release_upload = next(step for step in release_steps if step["name"] == "Upload release smoke dist artifacts")
    assert release_upload["if"] == "always()"
    assert release_upload["uses"] == "actions/upload-artifact@v4"
    assert release_upload["with"]["path"] == "dist"
    assert release_upload["with"]["retention-days"] == 14
