# Validation matrix

This document records the validation structure required for the `v0.1.0` release.

## 1. Public CI-gated validation

These checks are expected to run on GitHub-hosted Ubuntu:

- package build metadata
- package import and module smoke
- postprocessing smoke
- surrogate model-selection smoke
- validation-runner smoke
- Vega-acceptance-wrapper smoke
- MPI smoke
- docs presence / link-oriented checks

## 2. Manual GPU / workstation validation

These checks require a Linux workstation with an NVIDIA GPU:

- install the repo on a clean environment
- run at least one surrogate retraining path
- run Phase 1 GPU-batched
- run Phase 3b GPU-batched
- run MAP extraction and plotting on the resulting outputs

## 3. Cluster / Vega acceptance validation

These checks require the real target environment:

- install on Vega
- bootstrap vendored `extern/korali/` into repo-local `_vega/`
- run the public Vega validation matrix first
- validate MPI execution
- run the single Vega-first acceptance command
- validate Phase 1 GPU-batched
- validate Phase 2 on the current Korali backend
- validate Phase 3b GPU-batched
- validate the required propagation + plotting path
- archive the machine-readable acceptance report and logs

## Notes

The public CI layer is intentionally smaller than the full release contract because some required workflows are hardware-specific.

Those hardware-specific checks must still be recorded in-repo so the release does not depend on undocumented tribal knowledge.

The public Vega validation matrix is documented in `VEGA_VALIDATION_MATRIX.md`.

The single Vega-first acceptance command is documented in `VEGA_ACCEPTANCE_COMMAND.md`, and the tiny validation configs intended for that path are documented in `VALIDATION_CONFIGS.md`.
