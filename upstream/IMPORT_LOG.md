# Upstream Import Log

## 2026-04-04

- Established bootstrap branch for the release skeleton.
- Added release metadata, CI, tests, docs scaffold, package scaffold, and synchronization tooling.
- Primary source branch for follow-up imports is `vega/gpu-batching`.
- Follow-up pull requests will add imported code and reconciled changes from the corresponding `master` commits.

## 2026-04-21

- Triaged `UQ_DPD` grouped-holdout surrogate validation blocks against current MesoUQ state.
- Confirmed Sobol/sensitivity import is already present via `meso_uq.sensitivity`.
- Documented import plan for grouped-holdout validation in:
  - `docs/SURROGATE_GROUP_HOLDOUT_IMPORT_PLAN.md`
- Updated that plan to include dual-family validation (`dnn`, `bnn`) and paper-facing figure generation targets for:
  - sensitivity comparisons
  - held-out curve L2 error comparisons
- Expanded the plan into an execution-spec format (file-by-file work packages, schema contracts, CLI contracts, test matrix, acceptance criteria) so implementation can be delegated directly.
