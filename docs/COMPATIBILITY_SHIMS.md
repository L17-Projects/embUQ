# Compatibility Shims and Deprecation Policy

This document defines the warning policy for legacy wrappers and imports during the Wave 1 migration effort.

For each legacy surface, define the deprecation message contract using these fields:

- legacy path/import
- canonical replacement
- migration window
- removal condition
- CI noise policy
- compatibility-test expectations

## compression

- legacy path/import: `meso_uq/compression`, `compression/` entrypoints, and historical compression convenience imports.
- canonical replacement: current `src/meso_uq/surrogate/*` package interfaces plus future EMB package/workflow runners introduced by the migration.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove only after canonical entrypoints are stable, compatibility tests are green for one full release cycle, and users have had warning exposure.
- CI noise policy: emit at most one deprecation notice per invocation site so CI logs remain stable; suppress duplicate warning bursts in loop-heavy flows.
- compatibility-test expectations: keep wrappers and shims importable and executable by existing coverage tests until the migration window closes.

## indentation

- legacy path/import: `meso_uq/indentation`, `indentation/` entrypoints, and indentation launcher aliases.
- canonical replacement: current `src/meso_uq/surrogate/*` package interfaces plus future EMB package/workflow runners introduced by the migration.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove only after the canonical path demonstrates parity in production-smoke and acceptance-style runs and all migration evidence has been refreshed.
- CI noise policy: warn only when wrapper/import path is used, not during successful direct use of canonical paths.
- compatibility-test expectations: retain wrapper importability and behavioral parity checks in the reduced/indentation workflows and existing public tests.

## inference/scripts

- legacy path/import: `inference/scripts` launchers and module entrypoints.
- canonical replacement: package-owned inference workflow runners under the migrated `src/meso_uq` architecture.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove only when runner discovery and migration docs are updated and direct canonical calls are covered by compatibility smoke tests.
- CI noise policy: emit warning only once per process for this wrapper family.
- compatibility-test expectations: existing inference-script smoke and matrix tests must pass with legacy path still present.

## propagation/scripts

- legacy path/import: `propagation/scripts` launch wrappers and ad-hoc compatibility imports.
- canonical replacement: package-owned propagation/reporting workflow runners under the migrated `src/meso_uq` architecture and canonical script contracts.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove when propagation migration evidence is accepted and no downstream workflow still resolves legacy roots.
- CI noise policy: deprecation text should be visible in test logs and launch traces, but deduplicated per unique call site.
- compatibility-test expectations: keep compatibility script tests green and include explicit migration-window assertions before removal.

## reduced

- legacy path/import: `reduced` configs, scripts, and legacy task runners.
- canonical replacement: architecture-neutral reduced-workflow roots under the migrated package surface.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove after reduced workflow documentation, canonical replacements, and migration validation evidence are complete.
- CI noise policy: preserve one warning per reduced-root invocation path, and avoid repeated spam when invoking multiple profiles.
- compatibility-test expectations: existing reduced wrapper, script, and configuration tests continue to execute through compatibility surfaces until cycle completion.

## scripts/vega

- legacy path/import: `scripts/vega/*` compatibility wrappers and aliases.
- canonical replacement: `scripts/platforms/vega/*` during the platform-policy migration, then any later site-neutral dispatcher once it exists.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove after site-neutral dispatch coverage is green and migration notes explicitly mark Vega wrappers as deprecated-only.
- CI noise policy: emit warnings when these wrapper roots are exercised; avoid warning amplification across repeated matrix rows.
- compatibility-test expectations: keep wrapper delegate behavior intact and covered by existing matrix/acceptance tests as long as compatibility policy is active.

## scripts/ci

- legacy path/import: `scripts/ci/*` compatibility launcher scripts for CI-facing workflows.
- canonical replacement: `scripts/qa/ci/*` during the path-policy migration, then any later package-owned CI helpers once they exist.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove only after CI workflows are migrated without regression and no external callers rely on old paths.
- CI noise policy: warnings are acceptable in CI, but must be constant-time per stage and not generate duplicate noise per shard.
- compatibility-test expectations: existing CI wrapper integration checks must continue to pass while wrappers remain, with explicit deprecation-path assertions added when applicable.

## scripts/karolina

- legacy path/import: `scripts/karolina/*` and related Karolina compatibility shims.
- canonical replacement: `scripts/platforms/karolina/*` during the platform-policy migration, then any later architecture-neutral platform abstraction layer once it exists.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove only after Karolina parity is validated and migration playbook points users to canonical equivalents.
- CI noise policy: keep warning emission to one per active script path during CI runs to preserve signal-to-noise ratio.
- compatibility-test expectations: keep platform-compat tests for Karolina wrappers passing until canonical dispatch is fully authoritative.

## scripts/hpc

- legacy path/import: `scripts/hpc/*` compatibility wrappers and alias entrypoints.
- canonical replacement: `scripts/platforms/hpc/*` during the platform-policy migration, then any later shared platform-neutral dispatch once it exists.
- migration window: one release cycle from first Wave 1 merge.
- removal condition: remove only after HPC launch semantics are stable on canonical dispatcher and deprecation evidence is accepted.
- CI noise policy: emit a single deprecation message per wrapper module and avoid duplicate emission inside looped workflow stages.
- compatibility-test expectations: existing migration-wave harnesses and compatibility tests must continue passing until deprecation conditions are satisfied.
