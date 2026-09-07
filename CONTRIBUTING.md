# Contributing to MesoUQ

## General principle

`MesoUQ` is a curated public release line. Contributions should strengthen the public-facing software surface, documentation, tests, and maintainability.

## What kinds of contributions are welcome

Good contributions include:
- improvements to the shared package under `src/meso_uq/`
- workflow hardening and clearer configuration behavior
- better documentation and examples
- focused tests and smoke checks
- cleaner public surrogate, sensitivity, and postprocessing surfaces
- improvements to the focused vendored Korali patch surface when they are well-scoped and well-explained

## What to avoid in regular contributions

Please do not casually add:
- large binary artifacts or generated result trees
- private/local cluster assumptions
- manuscript-only `_paper` material
- broad unreviewed dumps from upstream research repos

## Provenance expectations

When a contribution ports content from an upstream research repository, keep the provenance explicit:
- say where the content came from
- prefer small, reviewable slices
- adapt code to the public release structure when that makes the public repo cleaner
- avoid mixing unrelated migrations into one PR

## PR style

A good PR should:
- have a narrow purpose
- state what user-facing capability it adds or improves
- avoid bundling generated artifacts with source changes
- keep docs and code consistent
- preserve the release-grade structure of the public repo

Review ownership for the release line is tracked in `.github/CODEOWNERS`.
Automated dependency maintenance is tracked in `.github/dependabot.yml`.

## Security issues

Do not use a public bug report for a suspected vulnerability.
Follow the private-first reporting path in `SECURITY.md`.

## Documentation expectations

If a PR adds a new public workflow surface, it should usually also update at least one of:
- `docs/GETTING_STARTED.md`
- `docs/WORKFLOWS.md`
- `docs/RELEASE_SCOPE.md`

## Testing expectations

Where practical, contributions should keep the public CI surface green and add focused tests rather than environment-heavy brittle checks.
