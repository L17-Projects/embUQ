# Native CUDA batch backend status

This note clarifies the current status of the vendored Korali `native_cuda_batch` option in `MesoUQ`.

## What exists in the vendored build surface

The vendored Korali Meson surface currently exposes:

- a `native_cuda_batch` option in `meson_options.txt`,
- conditional CUDA/NVRTC detection logic in `meson.build`,
- a `_KORALI_USE_CUDA_BATCH` configuration define.

This means the vendored subtree acknowledges a native-CUDA-oriented backend mode at build time.

## What is not yet publicly proven

The public `MesoUQ` repository does not yet provide a validated statement that:

- vendored Korali builds successfully with `-Dnative_cuda_batch=true` on a supported target,
- the public Phase 2 `Hierarchical/Psi` path actually exercises that backend,
- the resulting Phase 2 outputs are accepted downstream by the public Phase 3b path.

## Practical interpretation

For now, treat `native_cuda_batch` as:

- a real build-surface option,
- relevant to the intended backend direction,
- but not yet a fully validated public feature contract.

## Validation expectation

Use `docs/VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md` before claiming support for a native-CUDA Phase 2 path.
