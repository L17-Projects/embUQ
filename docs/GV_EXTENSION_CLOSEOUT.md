# GV Extension Closeout

Status and rollout rules for the MesoUQ GV extension closeout slice (MES-78/MES-81).

## Current status

- Scope: GV non-shear experiments only (`stretching`, `buckling`, `torsion`, `eigenmodes`).
- `shear_flow` is excluded from non-experimental acceptance and deferred under `MES-81`.
- This closeout package is **provisional** until merge evidence exists in Linear and repository history.

## Canonical GV operational chain

The intended operational acceptance chain is:

1. Mirheo runtime for canonical non-shear GV experiments.
2. DNN surrogate training / prediction path over the generated GV outputs.
3. Hierarchical Bayesian inference using the same shared phase-chain controls (`phase1`, `phase2`, `phase3b`).

The non-shear flow must demonstrate this `Mirheo -> DNN surrogate -> hierarchical inference` chain end-to-end for rollout.

## Parameter and control contract for release hardening

GV material calibration is constrained to exactly these parameters:

- `ka`
- `kb`
- `mu`
- `b1`
- `b2`
- `a3`
- `a4`
- `mu_l`
- `c`

GV controls are experiment design inputs and must remain excluded from calibrated vectors.

The GV noise model for this release-hardening slice is multiplicative `sigma`.

## Disk hygiene rules

To keep GV rollout storage-bounded on shared accounts:

- keep generated artifacts in ignored runtime/workspace roots such as `_runs`, `_vega`, and temporary staging trees,
- avoid tracking generated mesh/runtime objects, logs, or scratch under source-controlled paths,
- clean stale staging directories once accepted runs are archived,
- preserve only canonical manifests and small handoff records.

## Deferrals and active learning reminder

- `shear_flow` remains behind experimental opt-in and deferred as `MES-81`.
- after non-shear stack operability, hold a dedicated active-learning brainstorm on whether adaptive GV control selection can reduce canary and DNN warm-start cost before any production rollout decisions.

## Merge-evidence rule

Treat this item as blocked for hard closeout until merge evidence is recorded (merge commit, merged PR links, and required review-thread checks).
