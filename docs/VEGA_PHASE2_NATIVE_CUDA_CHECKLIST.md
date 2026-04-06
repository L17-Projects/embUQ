# Vega Phase 2 native-CUDA validation checklist

This checklist is the minimum acceptance layer needed before `MesoUQ` can honestly claim a supported **Phase 2 native-CUDA** path.

## Goal

Validate whether the vendored Korali native-CUDA backend can support the public `Hierarchical/Psi` Phase 2 workflow on the real target environment.

## Preconditions

Before attempting this checklist, verify:

- the desired `MesoUQ` branch is checked out,
- the vendored `extern/korali/` subtree is present and internally coherent,
- a compatible CUDA module stack is loaded,
- a matching MPI toolchain is loaded,
- `mpi4py` imports correctly in the runtime environment,
- successful Phase 1 outputs already exist for the intended dataset set.

## Build-side checks

1. Configure vendored Korali with the intended Meson options.
2. Record whether `-Dnative_cuda_batch=true` is enabled.
3. Record the CUDA toolkit path used by Meson.
4. Confirm that the configuration step resolves:
   - `cuda.h`
   - `nvrtc.h`
   - driver stub/library
   - NVRTC library
5. Save the exact Meson summary in the validation notes.

## Runtime-side checks

1. Run a small public Phase 2 dataset set on Vega.
2. Record the exact config used.
3. Record whether the run completes without Korali load/state errors.
4. Record whether the run writes a valid `results_phase_2/latest` output.
5. Record whether downstream Phase 3b can load that Phase 2 result.

## Performance / behavior checks

1. Confirm whether the native-CUDA path is actually exercised, not merely compiled.
2. Record GPU visibility and allocation during Phase 2.
3. Record whether runtime behavior differs from the documented CPU-MPI baseline.
4. Record any correctness mismatches relative to the standard public Phase 2 path.

## Minimum evidence to archive

For a successful acceptance result, archive at least:

- the exact branch and commit SHA,
- the Meson configure command,
- the Meson summary,
- the runtime launch command,
- stdout/stderr logs,
- the resulting `results_phase_2/latest` artifact path,
- a short statement about whether Phase 3b successfully consumed the Phase 2 output.

## Honest pass/fail rule

Do **not** mark Phase 2 native-CUDA as supported unless all of the following are true:

- build configuration succeeds,
- runtime execution succeeds,
- valid Phase 2 outputs are written,
- Phase 3b can consume those outputs,
- the evidence listed above is archived.
