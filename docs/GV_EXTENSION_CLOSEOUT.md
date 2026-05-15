# GV Extension Closeout

Status and rollout rules for the MesoUQ GV extension closeout slice (MES-98).

## Current status

- Scope is the core non-shear GV numerical generation stack: `stretching`, `buckling`, `torsion`, `eigenmodes`.
- `shear_flow` is intentionally excluded from non-experimental acceptance and stays behind experimental opt-in.
- Keep acceptance evidence scoped to this slice until post-hardening canary and handoff evidence is available per experiment.

## Canonical GV operational chain

The acceptance chain for this slice is:

1. Mirheo runtime for canonical GV experiments.
2. DNN surrogate training / prediction over generated GV outputs.
3. Hierarchical Bayesian inference through the same shared phase-chain (`phase1`, `phase2`, `phase3b`).

The non-shear stack must demonstrate this `Mirheo -> DNN surrogate -> hierarchical inference` chain end-to-end before hard closeout.

See also [GV_NUMERICAL_DATA_GENERATION.md](GV_NUMERICAL_DATA_GENERATION.md) for the full numerical-data handoff schema.
See [GV_PAPER_REPLAY_CLOSEOUT.md](GV_PAPER_REPLAY_CLOSEOUT.md) for the later
GV-only paper replay inventory, provenance, acceptance matrix, and qualitative
review gates.

## Repository layout

Every checked-in `gv/<experiment>/` directory uses the normalized GV experiment layout:

- `src/`: Mirheo/runtime provenance templates and experiment-local static inputs.
- `evalkit/`: experiment-local evaluation fixtures or manifests.
- `surrogate/`: experiment-local surrogate fixtures, manifests, or artifact placeholders.

Empty `evalkit/` and `surrogate/` directories are committed with `README.md` placeholders. Shared implementation belongs under `src/meso_uq`, not under experiment-local folders.

## Parameter and control contract

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

GV experiment controls are campaign design variables and are excluded from calibration vectors.

The GV noise model for this slice is multiplicative `sigma`.

## Disk policy

- Raw runtime products stay under ignored `_runs` trees.
- Canonical aggregated numerical datasets are authored under `_runs/gv/numerical_data/<campaign_id>/datasets/`.
- Numerical canary/campaign footprints are kept under the `25GB` GV data cap.

## Deferrals and active-learning reminder

- `shear_flow` remains in experimental scope and deferred from the non-shear core closeout.
- See [GV_SHEAR_FLOW_DEFERRAL.md](GV_SHEAR_FLOW_DEFERRAL.md) for the current runtime blocker, minimum data product, and graduation criteria.
- After the core GV stack is accepted, schedule an active-learning brainstorming session on adaptive control selection before any production rollout decisions.

## Merge evidence and hardening gate

Keep this slice blocked until campaign and PR evidence is recorded (campaign IDs, merged merge commits, and required review-thread checks).

## Runtime guardrails

- For `gv:stretching`, `gv:torsion`, `gv:buckling`, and `gv:eigenmodes`, generated command staging should run with `_vega/gv_venv/env.sh` sourced.
- `doctor_vega --with-gv-runtime` remains the hardening check for Mirheo import/lib path, `scale_space` resolution, dynamic-library checks, MDAnalysis importability, and OpenMPI wiring.
- The GV runtime sbatch wrapper and generated command staging (`commands.txt`) require `_vega/gv_venv/env.sh` and fail fast if that environment script is missing.
