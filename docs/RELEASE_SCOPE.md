# Release scope

`MesoUQ` is a curated public release line, not a verbatim dump of the private research repository.

## Included now

The public line already includes:
- the shared Python package `meso_uq`
- the supported workflow spine for `Phase 1`, `Phase 2`, and `Phase 3b` plus canonical production configs
- reduced-model public wrappers for `Phase 1`, `Phase 2`, and `Phase 3b`
- compression and indentation evalkit support needed by the workflow spine
- provisional non-shear GV workflow scaffolding (`stretching`, `buckling`, `torsion`, `eigenmodes`) under the MES-78 hardening slice
- surrogate retraining and evaluation entrypoints for compression and indentation
- Sobol sensitivity helpers and a lightweight Latin-hypercube design generator
- MAP extraction and plotting/postprocessing helpers
- a focused vendored Korali patch surface under `extern/korali/`
- a repo-local Mirheo bootstrap surface driven by `extern/mirheo.lock.json`

## Intentionally deferred

The following remain intentionally outside the current public boundary:
- heavy generated artifacts and large trained model payloads
- every internal/private convenience script from the research line
- full propagation execution pipelines beyond the current public plotting/postprocessing layer
- manuscript-specific `_paper` content
- `shear_flow` production acceptance in GV
- active-learning rollout decisions and deferred canary experiments that are not yet merged as evidence

## Why the scope is curated

The release line is curated so that:
- the repo remains understandable to outside users
- the public surface stays reviewable and maintainable
- provenance remains explicit
- heavy or highly local research infrastructure does not pollute the public package

## Practical interpretation

If you are using the repo now, the intended public strengths are:
- surrogate retraining/evaluation
- sensitivity support
- full-model and reduced-model hierarchical inference through `Phase 1`, `Phase 2`, and `Phase 3b`
- MAP extraction and plotting
- understanding the vendored Korali patch surface that supports those workflows
- the provisional GV closeout slice documented in `GV_EXTENSION_CLOSEOUT.md`

If you need large-scale data generation or cluster-specific Mirheo production pipelines, treat those as later extension work rather than a guaranteed part of the current public alpha line.
The repo now includes the bootstrap/runtime contract for Mirheo itself, but the heavy generated results remain intentionally outside the repository.
