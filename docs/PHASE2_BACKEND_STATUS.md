# Phase 2 backend status

This note clarifies the current public status of **Phase 2** in `MesoUQ`.

## What the public workflow currently does

The current public Phase 2 entrypoint is `inference/scripts/run_phase_2.py`.

That script:

- loads Phase 1 sub-experiments,
- constructs a `Hierarchical/Psi` Korali experiment,
- runs TMCMC through the standard Korali Python path,
- sets `Ranks Per Worker = 1`.

At the public workflow level, there is currently **no explicit runtime switch** that selects a native-CUDA backend path from `run_phase_2.py`.

## What the vendored Korali build surface exposes

The vendored `extern/korali/meson_options.txt` does define a `native_cuda_batch` Meson option.

The vendored `extern/korali/meson.build` also contains conditional logic for:

- CUDA driver headers and libraries,
- NVRTC,
- the `_KORALI_USE_CUDA_BATCH` configuration define.

So the build surface acknowledges a native-CUDA-oriented backend option.

## What the public operator documentation currently says

The current public HPC workflow guide says:

- `Phase 1`: GPU batched
- `Phase 2`: CPU MPI
- `Phase 3b`: GPU batched

and explicitly states that Korali itself remains a CPU/MPI library for that workflow.

## Honest public conclusion

As of this PR22 slice, the public repository should be understood as follows:

- the **supported documented workflow** still treats Phase 2 as **CPU MPI**,
- the vendored Korali subtree exposes a **native-CUDA build option**,
- but the public repo does **not yet provide a validated, supported native-CUDA Phase 2 execution contract**.

That means native-CUDA Phase 2 should currently be treated as an **experimental backend direction**, not as a finished public feature.

## What PR22 should achieve incrementally

A truthful PR22 pass should therefore do two things:

1. document the gap clearly,
2. define an acceptance checklist for the hardware-specific validation needed before Phase 2 can be claimed as a supported native-CUDA path.
