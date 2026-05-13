# Example Config Policy

`examples/configs/` is a release-oriented copy bundle. It is not the source of truth for
workflow behavior. The maintained configs live under:

- `inference/configs/production/`
- `reduced/configs/production/`

The bundle is intentionally small and should be treated as policy-managed evidence:

- **maintained**: the example matches its canonical workflow config and can stay in the bundle
- **needs_refresh**: the example is still relevant, but its contents drifted from canonical and should be synced
- **archive_candidate**: the example is historical evidence only and should be moved out of the active bundle after owner review
- **duplicate**: the example repeats another bundle entry or canonical payload
- **owner_decision**: the file is valid enough to inspect, but the retention call is not clear from repository evidence alone

## Refresh vs archival

Refresh when all of the following are true:

- the example still represents an active workflow contract
- the workflow still has a canonical config under `inference/configs/production/` or
  `reduced/configs/production/`
- the example is meant to be a convenience copy, not a divergent fork

Archive when all of the following are true:

- the example is no longer useful as a current copy
- the file exists only as history or release evidence
- a maintainer has confirmed the bundle should stop tracking it

Do not silently rewrite a historical example into a different meaning. If the owner wants a
different meaning, the file should move through a distinct review step.

## Validation rules

The example-policy checks in CI expect each maintained or refreshable example to:

- use a repository-relative path
- live under `examples/configs/`
- end in `.yaml` or `.yml`
- avoid private absolute path literals covered by `FORBIDDEN_PRIVATE_PATHS`
- load as structured YAML

When a canonical counterpart is known, maintained examples must match it byte-for-byte after
YAML parsing. Refresh candidates may differ, but they should still parse and stay within the
bundle path rules.

## CI expectations

The focused unit test file checks:

- manifest parsing
- current inventory coverage for the four tracked example configs
- private path rejection
- missing file detection
- maintained-example loading
- import-light behavior for the policy module

CI should also keep `git diff --check` clean for this slice so manifest and docs edits remain
whitespace-safe.

## Relationship to `configs/`

`configs/` holds schema-versioned example skeletons that are meant to be validated by the
config-policy layer. `examples/configs/` is different:

- it is workflow-oriented
- it mirrors production or reduced workflow configs for release-facing inspection
- it may lag canonical workflow configs and therefore needs periodic refresh or archival review

Keep the two bundles separate. `configs/` is for structured example schemas; `examples/configs/`
is for workflow snapshots and release evidence.
