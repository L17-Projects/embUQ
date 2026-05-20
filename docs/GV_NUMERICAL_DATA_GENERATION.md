# GV Numerical Data Generation

This is the current MES-98 handoff for GV numerical dataset generation.

## Workflow

The implemented chain is:

`Mirheo -> raw GV DPD output -> aggregated canonical HDF5 numerical dataset`

Raw simulation output is captured under runtime `_runs` trees by each GV experiment, then normalized into canonical HDF5 manifests through the GV postprocessing pipeline.

## Included experiments

Current production scope is restricted to four GV experiments:

- `stretching`
- `buckling`
- `torsion`
- `eigenmodes`

`shear_flow` is deferred and remains experimental until after the core GV stack is accepted.

GV paper replay closeout is tracked separately in
[GV_PAPER_REPLAY_CLOSEOUT.md](GV_PAPER_REPLAY_CLOSEOUT.md). That page is the
GV-only inventory for Figure 3, Figure 7, Figure 8, and the SI-backed torsion
diagnostic. It explicitly excludes EMB figures and deferred `shear_flow`.

## Mapping from raw output to canonical channels

Each experiment posts through `meso_uq.structures.gv.postprocessing.POSTPROCESSORS`.

- Stretching maps force-displacement data:
  - required channels: `tot_force`, `force`, `displacement`
- Buckling maps candidate buckling channels:
  - required at least one of `buckling_response`, `force_response`, `pressure_response`, `shape_amplitude`, `deformation_amplitude`
  - trajectory and support arrays under `mesh_*`, `ply_*`, `anchor_*`, `object_*`, `object_stats_*`, `stats_*` are accepted as candidate channels
- Torsion maps candidate channels:
  - required channels: `torsion_coord` and `torsion_response`
  - optional aliases currently supported: `theta`, `twist_angle`, `torsion_torque`, `cap_angular_displacement`, `stored_elastic_response`, `cap_motion`, `force`
- Eigenmodes maps spectral output:
  - required channel: `eigenvalues` (aliases accepted)
  - optional channel: `eigenvectors`

Postprocessing emits a canonical manifest and HDF5 payload per dataset id.

## Vega runtime stack

- Python runtime bootstrap for GV: `_vega/gv_venv`
- Runtime module stack includes `OpenMPI/4.1.4-GCC-12.2.0`
- Geometry preprocessing tooling for `scale_space` expects `MPFR/4.2.0-GCCcore-12.2.0` and `GMP/6.2.1-GCCcore-12.2.0`
- GV runtime postprocessing stack includes `h5py` and `MDAnalysis`
- Mirheo source lock is read from `extern/mirheo.lock.json`
- Health checks use `python scripts/platforms/hpc/doctor_hpc.py --site vega --with-gv-runtime`
- Runtime-generated GV command wrappers require `_vega/gv_venv/env.sh` and intentionally do not fall back to `_vega/mirheo/env.sh`.

## Geometry identity

Geometry is identified strictly by `radGV` (radius) and `height`:

- `geometry = gv_rad{radius}_height{height}`

The numerical manifest enforces this through `radius` and `height` parameters, and dataset ids always include that computed geometry token.

## Calibration contract

Exactly nine calibrated GV material parameters are allowed:

- `ka`, `kb`, `mu`, `b1`, `b2`, `a3`, `a4`, `mu_l`, `c`

No controls are calibrated in this stack; control names are kept as experiment design variables.

## Disk and canary policy

- Keep raw outputs and generated datasets under ignored `_runs/...` layouts for maintained campaigns and operational canaries.
- Explicit non-`_runs` output roots are reserved for local tests or scratch diagnostics and must remain untracked.
- Canonical dataset layout: `_runs/gv/numerical_data/<campaign_id>/datasets/`
- GV runtime canary guardrail: dataset footprint is limited to `25GB`.
- Mirror log hardening: any `NaN`/`Inf` token in Mirheo output logs fails the canary result.
- Policy assumes every experiment needs fresh post-hardening canary evidence, typically via short scientifically meaningful runs.

## Downstream handoff

- Downstream handoff is to DNN surrogate retraining/prediction and hierarchical inference.
- Active-learning acquisition/scoring/selection remains outside the GV runtime stack.
- Selected active-learning candidates can be converted to GV launch requests with
  `meso_uq.structures.gv.build_gv_active_learning_launch_handoff`.
- The GV handoff adapter owns validation against the existing launch schema,
  expected HDF5 dataset paths, and render-only scheduler artifact generation
  through `render_gv_active_learning_launch_handoff`.
- The adapter does not submit scheduler jobs. Rendered scripts and manifests
  remain operator-reviewed campaign artifacts under the requested `_runs/...`
  campaign root.
- Candidate payloads may provide only experiment, material, geometry, and
  control fields. Platform, walltime, GPU count, output root, and provenance are
  provided at the handoff boundary.
- Shared default controls are merged with candidate controls before launch
  validation, which keeps common fixed controls such as `bpress` from being
  repeated in every selected candidate.
- A minimal two-candidate selected-candidates input example lives at
  `configs/active_learning/gv_selected_candidates.example.yaml`. The matching
  derived, non-submitting handoff manifest with expected HDF5 dataset paths is
  `configs/active_learning/gv_selected_candidates_handoff_manifest.example.json`.
