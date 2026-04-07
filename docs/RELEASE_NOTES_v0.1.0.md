# MesoUQ `v0.1.0` release notes

## Summary

`v0.1.0` is the first public release-oriented line of MesoUQ.

This release turns the repository from a partial public skeleton into a substantially broader public workflow surface with:

- workflow execution for Phase 1, Phase 2, and Phase 3b
- surrogate retraining and lightweight model-selection utilities
- MAP extraction and plotting
- lightweight propagation execution and plotting handoff
- promoted operator / refresh utilities
- vendored Korali build-surface notes
- public smoke tests and release-validation docs

## Public surface in this release

### Core workflows

- compression inference/config workflow surface
- indentation inference/config workflow surface
- reduced workflow configuration surface
- public workflow validation driver utilities

### Surrogates

- compression surrogate retraining
- indentation surrogate retraining
- lightweight width/depth model selection for both modalities
- richer indentation multi-architecture diagnostic trainer

### Postprocessing

- MAP extraction
- posterior plotting helpers
- validation-overlay plotting helpers
- lightweight propagation execution for Phase 1 and Phase 3b

### Validation / release support

- CI workflow plus release-smoke workflow
- public smoke tests for surrogate model selection and postprocessing
- validation matrix and Vega acceptance checklist

## Known limits at `v0.1.0`

- some hardware-specific workflow claims still depend on manual validation rather than public CI
- the vendored Korali subtree has been strengthened substantially, but its strongest backend claims remain tied to documented validation paths rather than blanket public CI proof
- Phase 3a remains outside the required public release path
- propagation support is intentionally centered on the required public path rather than every historical upstream propagation utility

## Canonical docs to read first

- `README.md`
- `docs/README.md`
- `docs/VALIDATION_MATRIX.md`
- `docs/VEGA_ACCEPTANCE_CHECKLIST.md`
- `docs/HPC_GPU_BATCHED_REDUCED_INDENTATION.md`

## Tag intent

Once CI is green and the final release-artifact sweep is accepted, this branch is intended to be ready for tagging as `v0.1.0`.
