# MesoUQ documentation

This directory hosts the public documentation set for the `v0.1.0` release line.

## Start here

- `../README.md` for the root overview and install extras
- `../CONTRIBUTING.md` for the contribution and provenance policy
- `../SECURITY.md` for the public security-reporting policy
- `INSTALL.md` for the practical editable-install contract
- `DEPENDENCY_EXTRAS.md` for the install-extras contract

## Current architecture docs

- `ARCHITECTURE_OVERVIEW.md` for the migration-stage architecture map and where contracts are maintained
- `MESO_UQ_MIGRATION_INVENTORY.md` for the current repository surface map after Wave 0/1
- `MESO_UQ_COMPATIBILITY_SURFACE.md` for compatibility surface and deprecation boundaries
- `MESO_UQ_MIGRATION_RISK_REGISTER.md` for migration risks and required controls
- `MESO_UQ_GPU_VALIDATION_GATE.md` for the GPU closeout evidence contract
- `CONFIGURATION_POLICY.md` for config contract skeletons and schema expectations
- `ARTIFACT_POLICY.md` for manifest and artifact ownership rules
- `PLATFORM_POLICY.md` for platform-policy direction
- `SCRIPT_PATH_POLICY.md` for canonical script paths and compatibility aliases
- `COMPATIBILITY_SHIMS.md` for legacy wrapper warning and retirement policy
- `architecture/MES-127_131_WAVE0_GOVERNANCE_HUB.md` for governance and cross-reference index

## Current operations and evidence

- `VALIDATION_MATRIX.md` for the public / manual / Vega validation structure
- `WORKSTATION_ACCEPTANCE_CHECKLIST.md` for the Linux NVIDIA workstation proof layer
- `VEGA_ACCEPTANCE_COMMAND.md` for the single Vega-first acceptance command
- `VEGA_ACCEPTANCE_CHECKLIST.md` for the cluster acceptance layer and pass criteria
- `VEGA_BOOTSTRAP.md` for the fresh-clone Vega bootstrap path
- `VEGA_PRODUCTION_SANITY.md` for the canonical reduced-cost production smoke command
- `VEGA_VALIDATION_MATRIX.md` for the public Vega validation matrix and report surface
- `KAROLINA_FULL_PLATFORM.md` for the Karolina scratch, Slurm, runtime-root, and acceptance-evidence contract
- `MES-125_KAROLINA_ACCEPTANCE_CLOSEOUT.md` for the MES-125 Karolina closeout evidence skeleton
- `architecture/MES-127_131_WAVE0_GOVERNANCE_HUB.md` for the architecture-migration governance, inventory, risk, compatibility, and GPU validation gate
- `CONFIGURATION_POLICY.md` for the schema-versioned central config taxonomy
- `ARTIFACT_POLICY.md` for generated-output, curated-artifact, manifest, and cleanup rules
- `PHASE6_GENERATED_CLEANUP_RECORD.md` for the owner-approved generated-root cleanup closeout
- `COMPATIBILITY_SHIM_RETIREMENT_RECORD.md` for compatibility shim retention, blockers, and review triggers
- `PAPER_REPRODUCTION_ARTIFACT_POLICY.md` for the paper source versus external artifact boundary
- `TEST_STRUCTURE_POLICY.md` for the unit, integration, operational, optional-runtime, and GPU/HPC test layout
- `PLATFORM_POLICY.md` for Karolina, Vega, workstation, and generic Slurm platform config policy
- `EMB_WORKFLOW_EXTRACTION_PLAN.md` for the compression/indentation workflow extraction inventory and compatibility plan
- `MES-59_MERGE_READINESS_CHECKLIST.md` for the delayed/thread-aware connector review sweep required before merge readiness
- `MES-59_74_76_79_CLOSEOUT_EVIDENCE.md` for the MES-59/MES-74/MES-76/MES-79 closeout decisions and validation evidence
- `GV_EXTENSION_CLOSEOUT.md` for provisional MES-78 release hardening scope and GV rollout gate
- `VALIDATION_CONFIGS.md` for the tiny validation config bundle
- `WORKFLOWS.md` for workflow entrypoints and operator expectations

## Historical / release-specific docs

- `RELEASE_NOTES_v0.1.0.md` for the current public release summary
- `RELEASE_EVIDENCE_CHECKLIST_v0.1.0.md` for the release evidence bar at tag time
- `MES-59_MERGE_READINESS_CHECKLIST.md` for delayed/thread-aware connector review context
- `MES-59_74_76_79_CLOSEOUT_EVIDENCE.md` for prior closeout evidence and decisions
- `MES-125_KAROLINA_ACCEPTANCE_CLOSEOUT.md` for MES-125 closeout evidence skeleton
- `GV_EXTENSION_CLOSEOUT.md` for provisional MES-78 closeout scope and GV rollout gate
- `EMB_WORKFLOW_EXTRACTION_PLAN.md` for extraction planning under the compression/indentation slice

Also see legacy release/operational notes:

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
