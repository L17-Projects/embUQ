# GV shear-flow deferral

`gv:shear_flow` remains outside the active GV numerical-data launch interface.
It is tracked as an experimental lane, not as part of the four-lane production
GV DPD campaign surface.

## Current status

The active GV launch/sampling interface supports:

- `stretching`
- `buckling`
- `torsion`
- `eigenmodes`

`shear_flow` is still represented in the repo for provenance and future
recovery work:

- source assets: `gv/shear_flow/src/`
- runtime descriptor: `src/meso_uq/structures/gv/runtime/shear_flow.py`
- surrogate catalog metadata: `src/meso_uq/surrogate/gv_catalog.py`
- evidence excerpt:
  `gv/shear_flow/src/fixtures/bouncer_collision_candidates_coarse_excerpt.txt`

The runtime descriptor is explicitly `experimental=True`, and public runtime
paths require experimental opt-in. The launch request validator rejects
`shear_flow` through the shared GV sampling validation path.

## Blocking issues

The known runtime blocker is the mirheoOBMD bouncer collision overflow recorded
by the staged shear-flow source:

- issue id: `bouncer_collision_candidates_coarse`
- runtime package: `mirheoOBMD`
- affected bouncer: `membrane_bounce`
- evidence:
  `gv/shear_flow/src/fixtures/bouncer_collision_candidates_coarse_excerpt.txt`

This is a runtime validity blocker, not a documentation-only gap. File
production alone is not enough for graduation; the lane must produce finite,
scientifically meaningful observables under the same canary policy as the
non-shear GV lanes.

## Minimum data product

Before `shear_flow` can join the active GV launch interface, it needs one
canonical aggregate HDF5 dataset shape for a bounded shear-flow response:

- dataset id under the existing GV convention, for example
  `gv:shear_flow:<geometry>:<control_id>`
- geometry identity derived from `radGV` and `height`
- nine material parameters: `ka`, `kb`, `mu`, `b1`, `b2`, `a3`, `a4`,
  `mu_l`, `c`
- controls: `ptan`, `afsi`, `bpress`
- at least one finite response channel, currently cataloged as
  `shear_flow_response`
- raw-runtime provenance, log scan results, and postprocessing provenance

## Graduation criteria

Do not add `shear_flow` to the non-experimental GV launch API until all of the
following are true:

- the mirheoOBMD bouncer overflow has a reproduced fix or a documented runtime
  workaround;
- a short GPU canary completes on the target platform without `NaN`/`Inf` log
  tokens and without bouncer overflow;
- the canary writes the minimum HDF5 data product above;
- a postprocessor validates the aggregate dataset schema and finite observable
  channels;
- scheduler-ready render tests cover the lane without submission;
- launch documentation names the required runtime environment and output-root
  policy;
- Active Learning handoff examples continue to exclude `shear_flow` until the
  above evidence is merged.

Until those criteria pass, keep `shear_flow` behind experimental opt-in and
exclude it from GV campaign manifests generated for production or Active
Learning selected candidates.
