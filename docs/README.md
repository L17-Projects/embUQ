# MesoUQ documentation

This directory hosts the public documentation set for the `v0.1.0` release line.

## Start here

- `../README.md` for the root overview and install extras
- `../CONTRIBUTING.md` for the contribution and provenance policy
- `../SECURITY.md` for the public security-reporting policy
- `INSTALL.md` for the practical editable-install contract
- `RELEASE_NOTES_v0.1.0.md` for the release summary
- `RELEASE_EVIDENCE_CHECKLIST_v0.1.0.md` for the pre-tag evidence bar
- `DEPENDENCY_EXTRAS.md` for the install-extras contract
- `VALIDATION_MATRIX.md` for the public / manual / Vega validation structure
- `WORKSTATION_ACCEPTANCE_CHECKLIST.md` for the Linux NVIDIA workstation proof layer
- `VEGA_ACCEPTANCE_COMMAND.md` for the single Vega-first acceptance command
- `VEGA_ACCEPTANCE_CHECKLIST.md` for the cluster acceptance layer and pass criteria
- `VEGA_BOOTSTRAP.md` for the fresh-clone Vega bootstrap path
- `VEGA_PRODUCTION_SANITY.md` for the canonical reduced-cost production smoke command
- `VEGA_WORKFLOW_HELPERS.md` for split Vega workflow helpers and sbatch templates
- `VEGA_VALIDATION_MATRIX.md` for the public Vega validation matrix and report surface
- `KAROLINA_FULL_PLATFORM.md` for the Karolina scratch, Slurm, runtime-root, and acceptance-evidence contract
- `MES-125_KAROLINA_ACCEPTANCE_CLOSEOUT.md` for the MES-125 Karolina closeout evidence skeleton
- `GV_EXTENSION_CLOSEOUT.md` for provisional MES-78 release hardening scope and GV rollout gate
- `VALIDATION_CONFIGS.md` for the tiny validation config bundle

## Workflow and operator guides

- `HPC_GPU_BATCHED_REDUCED_INDENTATION.md`
- `SURROGATE_MODEL_SELECTION.md`
- `PROPAGATION_EXECUTION.md`
- `PHASE2_BACKEND_STATUS.md`
- `VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md`
- `DEV_UTILITIES_PROMOTED.md`

## Backend notes

The vendored Korali subtree is documented further in:

- `../extern/korali/README.md`
- `../extern/korali/LOCAL_GPU_BATCHING_NOTES.md`
- `../extern/korali/NATIVE_CUDA_BATCH_STATUS.md`

## Notes

The early bootstrap scaffold is no longer the current state of the repo. The docs in this directory should now be treated as the public entrypoints for release, validation, workflow usage, acceptance, and operator guidance.
