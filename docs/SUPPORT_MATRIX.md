# Support matrix

This matrix describes the intended support target for the v0.1.0 hardening line.

| Environment | Install | Execute workflows | Notes |
|---|---:|---:|---|
| GitHub-hosted Ubuntu CI | Yes | Limited | package import, unit/smoke, MPI smoke, lightweight analysis |
| Linux CPU workstation | Yes | Partial | analysis, retraining, plotting, light testing |
| Linux single NVIDIA GPU workstation | Yes | Yes | required target for GPU-batched workflows |
| Linux multi-GPU node | Yes | Yes | required target for larger workflow execution |
| Linux MPI cluster | Yes | Yes | required target for MPI-aware workflows |
| Vega | Yes | Yes | required acceptance-validation platform |
| macOS developer machine | Optional | Light only | convenience/dev use, not a required execution target |

## What “execute workflows” means here

It does not mean every workflow is proven on every platform.
It means the platform is part of the intended validation target for at least part of the public release contract.

## Public CI vs real-hardware validation

Some release requirements cannot be fully proven on public CI.
For this repo, validation must be split into:
- public CI-gated smoke and unit coverage
- manual GPU workstation validation
- HPC / Vega acceptance validation

## Required-at-release execution targets

The release contract requires real execution support for:
- surrogate retraining, compression
- surrogate retraining, indentation
- sensitivity analysis
- Phase 1 GPU-batched
- Phase 2 native-CUDA
- Phase 3b GPU-batched
- MAP extraction
- plotting / postprocessing
- Phase 3b propagation + plotting

Phase 3a can remain partial for the first serious release.
