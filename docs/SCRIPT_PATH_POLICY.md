# Script Path Governance Policy

## Canonical script surface

This repository keeps script execution paths on the following canonical roots:

- CI helpers: `scripts/qa/ci/*.py`
- QA helpers: `scripts/qa/*` and `scripts/qa/release/*`
- Platform helpers: `scripts/platforms/<site>/*`
- Workflow scripts: `scripts/workflows/*`
- Shared workflow helpers: `scripts/shared/*`
- Slurm launchers and templates: `scripts/platforms/<site>/**/` and `scripts/platforms/<site>/sbatch/*.sbatch`

When workflows or platform scripts launch work, they should reference scripts through these canonical roots.

## Compatibility aliases and migration window

The following compatibility aliases are maintained only for migration parity and must stay in place while this wave is active:

- `scripts/ci` -> `scripts/qa/ci`
- `scripts/vega` -> `scripts/platforms/vega`
- `scripts/karolina` -> `scripts/platforms/karolina`
- `scripts/hpc` -> `scripts/platforms/hpc`

Compatibility policy for these roots:

- preserve the alias as a symlink (or equivalent filesystem-forwarding entrypoint equivalent) for one release-cycle migration window from the first Wave-1 merge
- allow callers to keep using old paths while we migrate operators/validators
- prefer direct canonical paths in newly added CI jobs, templates, and launch docs
- remove/retire aliases only after canonical references are green and migration acceptance evidence is accepted

## Governing expectations

Governance checks must enforce:

- referenced scripts in GitHub workflow YAML are present in-repo
- referenced scripts in platform Slurm templates are present in-repo
- legacy roots (`scripts/ci`, `scripts/vega`, `scripts/karolina`, `scripts/hpc`) resolve to their canonical targets when referenced
- unsupported script roots are rejected or migrated into canonical/compatibility families

## Practical parser limits

Governance checks intentionally use conservative static matching, not full shell interpolation.
They only validate references that are explicit repo-relative paths such as:

- `scripts/.../*.py`
- `scripts/.../*.sh`
- `scripts/.../*.sbatch`
- `scripts/.../*.yml` / `scripts/.../*.yaml`

If a workflow/template uses runtime interpolation (`$SITE`, `${SITE}`, matrix expression, etc.) before the `scripts/` path, that indirection is out of scope for deterministic static checks and must be covered by existing runtime/template tests.
