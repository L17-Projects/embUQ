# Phase 2 backend status

This note describes the current branch-level `Phase 2` contract in `MesoUQ`.

## Public entrypoint

The public `Phase 2` entrypoint remains:

- `inference/scripts/run_phase_2.py`

That entrypoint now exposes an explicit runtime switch:

- `--phase2-backend cpu-mpi`
- `--phase2-backend native-cuda`

If no explicit override is provided, the current workflow defaults are:

- `production` profile: `native-cuda`
- `validation` profile: `cpu-mpi`

## Current runtime behavior

For `phase2_backend=native-cuda`, `run_phase_2.py` now:

- requires a single MPI rank,
- configures a Sequential Korali conduit,
- enables batch evaluation,
- selects `Batch Evaluation Backend = NativeCuda`,
- checks repo-local Meson metadata when available to verify that Korali was built with
  `native_cuda_batch` enabled.

For `phase2_backend=cpu-mpi`, the entrypoint keeps the MPI path available as an explicit fallback.

## Build surface

The repo-local Korali build still depends on the vendored build surface exposing:

- the `native_cuda_batch` Meson option,
- CUDA driver / NVRTC linkage,
- the corresponding Korali native-CUDA batch path.

## Honest current conclusion

The code-level backend contract is now implemented in the public workflow spine:

- `Phase 2` supports `cpu-mpi` and `native-cuda`,
- production orchestration defaults to `native-cuda`,
- validation orchestration defaults to `cpu-mpi`.

What is still separate from the contract is runtime proof:

- target-hardware canaries still need to confirm that the native-CUDA path runs cleanly,
- downstream `Phase 3b` and postprocess steps must consume those outputs correctly,
- release/audit evidence must capture that result honestly.

Until that runtime evidence exists, the branch can truthfully claim an implemented native-CUDA
operator path, but not a finished audit pass for the full HUQ-EMB rebuild.

## Validation surface

The hardware-facing validation checklist remains:

- `docs/VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md`

That checklist is the place to record the actual canary evidence and any failure modes.
