# Local GPU Batching Notes

This Korali fork is the required backend for the GPU-batched workflows in the MesoUQ operator path.

## Scope

Korali itself is still built and run as a CPU/MPI library.

The GPU acceleration used by the reduced indentation workflow happens in the PyTorch surrogates in the workflow repository, not inside Korali’s own C++ kernels.

What this fork adds is batch-aware problem evaluation so that the Python side can hand a whole particle batch to the surrogate in one call.

## Required branch

Use:

- repo: `BrieucB/korali`
- branch: `gpu-batch-eval`

## Core modified source areas

The GPU-batching workflow depends on these local Korali changes:

- `source/modules/problem/bayesian/reference/reference.config`
- `source/modules/problem/hierarchical/theta/theta.cpp.base`
- `source/modules/solver/sampler/TMCMC/TMCMC.cpp.base`

There are also matching checked-in `.cpp` and `.hpp` files. Build from the checked-in branch exactly as-is; no extra code generation step is needed for normal use.

## Build summary

Recommended Meson options:

```bash
meson setup build \
  --wipe \
  --buildtype=release \
  --prefix=$KORALI_PREFIX \
  -Dmpi=true \
  -Dmpi4py=true \
  -Dopenmp=false

meson install -C build
```
