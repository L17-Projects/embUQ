# Audit Polishing Policy

This policy records how baseline weekly-audit signals become actionable cleanup work without turning weak signals into unsafe repository changes.

## Scope

The authoritative MesoUQ target surface is Vega, Karolina, and GitHub Linux CI. Local workstation-only behavior is audit-host context unless it prevents evidence collection. Audit polishing work must not modify active production or active-learning work unless the owning issue explicitly asks for it.

Machine-readable policy examples live under `configs/audit_polishing/`:

- `orphan_python_triage.example.json` for orphan/dead-code candidate classification.
- `large_tracked_payloads.example.json` for large tracked scientific payload ownership.
- `static_quality_policy.example.json` for static quality and dependency gate decisions.

## Orphan Candidate Handling

The orphan scan means only that a Python file had no import or text-reference hit in the baseline scanner. It does not prove the file is unused. Classify each candidate before any deletion:

- `keep`: the entrypoint is active and should remain as-is.
- `document`: the entrypoint is active but needs docs, manifests, or tests to make reachability clear.
- `deprecate`: the entrypoint can be phased out after a documented compatibility window.
- `delete`: the owner approved removal and focused tests prove the public surface is unaffected.
- `false_positive`: the scanner missed an intentional dynamic entrypoint or external scheduler reference.
- `needs_owner_review`: no action is allowed yet.

Vega and Karolina platform wrappers, GV runtime sources, and EMB active-learning workflow entrypoints must be preserved until their owners confirm a narrower action.

## Large Payload Ownership

Large tracked payloads are allowed only when they are curated scientific inputs, reference geometry, vendored source, or release-critical baseline artifacts. Each tracked payload needs an owner, retention policy, storage location, and status. Moving a payload out of the repository requires a reproducibility plan with checksum, retrieval path, and a test or smoke command that proves the replacement works.

No broad size-based deletion is allowed. The baseline found no tracked generated-root drift, so the current action is ownership documentation, not removal.

Create a content-addressed, non-destructive cleanup plan with:

```bash
/usr/bin/python3.11 scripts/qa/collect_open_pr_snapshot.py \
  --output <open-pr-snapshot.json>
/usr/bin/python3.11 scripts/qa/plan_repository_quarantine.py \
  --orphan-report <orphan-candidates.json> \
  --open-pr-snapshot <open-pr-snapshot.json> \
  --manifest-dir papers/UQ_EMB/manifests \
  --quarantine-root <scratch-quarantine-root> \
  --output-json <cleanup-plan.json> \
  --output-markdown <cleanup-plan.md>
```

The planner does not move or delete files. It records hashes, references,
open-PR reachability, protected paths, and destination capacity so that any
later quarantine can be reviewed independently.

## Static Quality And Dependency Gates

The minimum enforced audit-polishing gates are docs link checks, structural governance tests, script path governance, repository governance, and workflow action pin policy tests. General lint, type checking, and dependency security checks remain advisory or deferred until a scoped baseline is accepted.

Dependabot GitHub Actions PRs must keep workflow YAML and workflow-policy tests in the same branch. If Dependabot updates a pinned action SHA, `tests/test_github_workflows.py` must be updated in the same PR or a follow-up branch must supersede the Dependabot PR.

## Closeout

Close polishing issues only after the manifest entry, evidence path, and Linear issue state agree. Accepted risks must include an owner, reason, and review condition.
