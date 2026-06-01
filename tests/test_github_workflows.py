from pathlib import Path

import yaml

UPLOAD_ARTIFACT_SHA = "actions/upload-artifact@" "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
CODECOV_ACTION_SHA = "codecov/codecov-action@" "e79a6962e0d4c0c17b229090214935d2e33f8354"
RUN_COVERAGE_SCOPE_IF = (
    "success() && github.event_name == 'pull_request' && "
    "steps.coverage-scope.outputs.run_coverage_gate == 'true'"
)
FULL_CI_PR_IF = (
    "github.event_name == 'pull_request' && "
    "contains(github.event.pull_request.labels.*.name, 'full-ci')"
)
FULL_CI_JOB_IF = (
    "github.event_name == 'workflow_dispatch' || github.ref == 'refs/heads/main' || "
    "(github.event_name == 'pull_request' && "
    "contains(github.event.pull_request.labels.*.name, 'full-ci'))"
)


def _load_workflow(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    workflow_path = repo_root / ".github" / "workflows" / name
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


def _workflow_triggers(workflow):
    return workflow.get("on", workflow.get(True))


def _uses_by_step(workflow):
    mapping = {}
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                mapping[step["name"]] = step["uses"]
    return mapping


def _step_by_name(steps, step_name: str):
    return next(step for step in steps if step["name"] == step_name)


def test_ci_workflow_has_concurrency_timeouts_and_canary_artifacts():
    workflow = _load_workflow("ci.yml")
    triggers = _workflow_triggers(workflow)

    assert workflow["permissions"] == {"contents": "read"}
    assert triggers["pull_request"]["types"] == ["opened", "synchronize", "reopened", "labeled"]
    assert "workflow_dispatch" in triggers
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "github.workflow" in workflow["concurrency"]["group"]
    assert workflow["jobs"]["package-and-tests"]["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }

    expected_timeouts = {
        "package-and-tests": 15,
        "docs": 5,
        "platform-interface-gate": 10,
        "mpi-smoke": 20,
        "retraining-canary": 15,
        "workflow-canary": 20,
    }
    for job_name, timeout in expected_timeouts.items():
        assert workflow["jobs"][job_name]["timeout-minutes"] == timeout

    package_steps = workflow["jobs"]["package-and-tests"]["steps"]
    coverage_upload = _step_by_name(package_steps, "Upload coverage artifacts")
    codecov_upload = _step_by_name(package_steps, "Upload coverage to Codecov")
    preserve_head = _step_by_name(package_steps, "Preserve head coverage report")
    compute_base = _step_by_name(package_steps, "Compute base branch coverage")
    coverage_delta = _step_by_name(package_steps, "Enforce feedback coverage increase")
    coverage_scope = _step_by_name(package_steps, "Detect coverage-sensitive changes")
    installed_smoke = _step_by_name(package_steps, "Run installed package smoke")
    assert preserve_head["if"] == "always()"
    assert "test -f coverage.json" in preserve_head["run"]
    assert installed_smoke["run"] == "python scripts/ci/run_installed_package_smoke.py --wheel dist"
    assert compute_base["if"] == f"success() && {FULL_CI_PR_IF}"
    assert "${{ github.event.pull_request.base.sha }}" in compute_base["run"]
    assert "${{ github.base_ref }}" not in compute_base["run"]
    assert coverage_scope["if"] == FULL_CI_PR_IF
    assert coverage_scope["id"] == "coverage-scope"
    assert _step_by_name(package_steps, "Restore base coverage cache")["if"] == FULL_CI_PR_IF
    assert coverage_upload["if"] == "always()"
    assert coverage_upload["uses"] == UPLOAD_ARTIFACT_SHA
    assert coverage_upload["with"]["name"] == "coverage-report"
    assert coverage_upload["with"]["retention-days"] == 14
    assert "coverage-head.json" in coverage_upload["with"]["path"]
    assert "coverage-base.json" in coverage_upload["with"]["path"]
    assert "coverage-delta.md" in coverage_upload["with"]["path"]
    assert coverage_delta["if"] == RUN_COVERAGE_SCOPE_IF
    assert "test -f coverage-head.json" in coverage_delta["run"]
    assert "test -f coverage-base.json" in coverage_delta["run"]
    assert "check_coverage_increase.py" in coverage_delta["run"]
    assert codecov_upload["uses"] == CODECOV_ACTION_SHA
    assert codecov_upload["with"]["token"] == "${{ secrets.CODECOV_TOKEN }}"
    assert codecov_upload["with"]["use_oidc"] == "${{ secrets.CODECOV_TOKEN == '' }}"
    assert codecov_upload["with"]["slug"] == "BrieucB/MesoUQ"
    assert codecov_upload["with"]["files"] == "coverage.xml"
    assert codecov_upload["with"]["disable_search"] is True
    assert codecov_upload["with"]["codecov_yml_path"] == "codecov.yml"
    assert codecov_upload["with"]["fail_ci_if_error"] is False

    assert workflow["jobs"]["bnn-backend-canary"]["if"] == FULL_CI_JOB_IF
    assert workflow["jobs"]["mpi-smoke"]["if"] == FULL_CI_JOB_IF
    assert workflow["jobs"]["retraining-canary"]["if"] == FULL_CI_JOB_IF
    assert workflow["jobs"]["workflow-canary"]["if"] == FULL_CI_JOB_IF

    platform_gate = workflow["jobs"]["platform-interface-gate"]
    assert platform_gate["strategy"]["fail-fast"] is False
    assert platform_gate["strategy"]["matrix"]["site"] == ["vega", "karolina"]
    platform_steps = platform_gate["steps"]
    platform_check = _step_by_name(platform_steps, "Run Vega/Karolina platform-interface gate")
    assert platform_check["env"] == {"MESOUQ_SITE": "${{ matrix.site }}"}
    platform_run = platform_check["run"]
    assert "tests/test_hpc_shared_script_governance.py" in platform_run
    assert "tests/test_script_path_governance.py" in platform_run
    assert "tests/test_karolina_validation_matrix.py" in platform_run
    assert "tests/test_vega_matrix_sbatch.py" in platform_run
    assert "tests/unit/test_gv_platform_routing.py" in platform_run
    assert "tests/unit/test_dpd_production_preflight.py" in platform_run

    workflow_steps = workflow["jobs"]["workflow-canary"]["steps"]
    mpi_steps = workflow["jobs"]["mpi-smoke"]["steps"]
    mpi_install = _step_by_name(mpi_steps, "Install MPI stack")
    assert "--no-install-recommends openmpi-bin libopenmpi-dev" in mpi_install["run"]

    korali_cache = _step_by_name(workflow_steps, "Restore Korali runtime cache")
    assert "workflow-canary-v2-korali" in korali_cache["with"]["key"]
    assert ".github/workflows/ci.yml" in korali_cache["with"]["key"]

    workflow_install = _step_by_name(workflow_steps, "Install workflow canary system packages")
    assert "--no-install-recommends openmpi-bin libopenmpi-dev" in workflow_install["run"]

    workflow_bootstrap = _step_by_name(workflow_steps, "Bootstrap canonical Korali runtime")
    assert "--system-site-packages" in workflow_bootstrap["run"]
    assert "--skip-python-deps" in workflow_bootstrap["run"]
    assert "--with-korali" in workflow_bootstrap["run"]

    workflow_validate = _step_by_name(workflow_steps, "Validate canonical Korali runtime")
    assert 'source "${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh"' in workflow_validate["run"]
    assert "import korali" in workflow_validate["run"]
    assert "import torch" in workflow_validate["run"]

    workflow_summary = _step_by_name(workflow_steps, "Summarize workflow canary outputs")
    workflow_upload = _step_by_name(workflow_steps, "Upload workflow canary artifacts")
    assert workflow_summary["if"] == "always()"
    assert workflow_upload["if"] == "always()"
    assert workflow_upload["uses"] == UPLOAD_ARTIFACT_SHA
    assert workflow_upload["with"]["path"] == "_ci/workflow_canary"
    assert workflow_upload["with"]["retention-days"] == 14

    retraining_steps = workflow["jobs"]["retraining-canary"]["steps"]
    retraining_upload = _step_by_name(retraining_steps, "Upload retraining canary artifacts")
    assert retraining_upload["if"] == "always()"
    assert retraining_upload["uses"] == UPLOAD_ARTIFACT_SHA
    assert retraining_upload["with"]["path"] == "_ci/surrogate_retraining"
    assert retraining_upload["with"]["retention-days"] == 14

    action_refs = set(_uses_by_step(workflow).values())
    assert action_refs == {
        "actions/cache@27d5ce7f107fe9357f9df03efb73ab90386fccae",
        "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
        "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
        UPLOAD_ARTIFACT_SHA,
        CODECOV_ACTION_SHA,
    }


def test_release_smoke_workflow_has_concurrency_timeouts_and_dist_artifact():
    workflow = _load_workflow("release-smoke.yml")
    triggers = _workflow_triggers(workflow)

    assert workflow["permissions"] == {"contents": "read"}
    assert triggers["pull_request"]["types"] == ["opened", "synchronize", "reopened", "labeled"]
    assert "workflow_dispatch" in triggers
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert "github.workflow" in workflow["concurrency"]["group"]
    assert workflow["jobs"]["release-smoke"]["timeout-minutes"] == 15
    assert workflow["jobs"]["release-smoke"]["if"] == FULL_CI_JOB_IF
    assert workflow["jobs"]["docs-link-check"]["timeout-minutes"] == 5

    release_steps = workflow["jobs"]["release-smoke"]["steps"]
    release_upload = _step_by_name(release_steps, "Upload release smoke dist artifacts")
    assert release_upload["if"] == "always()"
    assert release_upload["uses"] == UPLOAD_ARTIFACT_SHA
    assert release_upload["with"]["path"] == "dist"
    assert release_upload["with"]["retention-days"] == 14

    docs_steps = workflow["jobs"]["docs-link-check"]["steps"]
    docs_check = next(
        step
        for step in docs_steps
        if step["name"] == "Validate release docs and governance metadata"
    )
    docs_run = docs_check["run"]
    assert "test -f CONTRIBUTING.md" in docs_run
    assert "test -f SECURITY.md" in docs_run
    assert "test -f docs/RELEASE_EVIDENCE_CHECKLIST_v0.1.0.md" in docs_run
    assert "test -f docs/WORKSTATION_ACCEPTANCE_CHECKLIST.md" in docs_run
    assert "test -f .github/CODEOWNERS" in docs_run
    assert "test -f .github/dependabot.yml" in docs_run

    action_refs = set(_uses_by_step(workflow).values())
    assert action_refs == {
        "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
        "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
        UPLOAD_ARTIFACT_SHA,
    }


def test_codecov_patch_status_is_informational():
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo_root / "codecov.yml").read_text(encoding="utf-8"))

    assert config["codecov"]["require_ci_to_pass"] is False
    assert config["coverage"]["status"]["patch"]["default"]["informational"] is True
