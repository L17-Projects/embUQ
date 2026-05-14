# Governance Test Policy

This repository keeps a lightweight structural governance slice under `tests/test_structural_governance_policy.py`.
The slice is intentionally fast and avoids deep vendor-tree inspection, generated-data traversal, or external storage/HPC dependencies.

The shared helper in `tests/support/governance.py` checks:

- `schema_version` presence for structured files under `configs/`
- the example artifact manifest under `configs/artifacts/`
- tracked generated-root files via `git ls-files`
- the narrow `meso_uq.public_api` export surface
- CI workflow references to repo-local generated evidence paths
- docs markers for legacy command surfaces that still need explicit review
- the structure of the import-cycle allowlist entries

Allowlist entries live in `configs/governance_allowlist.example.json`.
Each exception record must carry:

- `owner`
- `reason`
- `review_condition`

The allowlist is meant to be explicit and reviewable:

- `owner` records who is responsible for the exception
- `reason` records why the exception exists
- `review_condition` records when the exception should be revisited

Failure messages should name the exact path and include a remediation hint. The intended remediation is usually one of:

- add or restore `schema_version`
- untrack or relocate generated root content
- update the public API allowlist when exports change intentionally
- update the CI/docs allowlist when legacy references are still required
- remove legacy compatibility text once the compatibility window ends

The helper does not scan `extern/korali` deeply and does not depend on GPU, MPI, or other external runtime services.
