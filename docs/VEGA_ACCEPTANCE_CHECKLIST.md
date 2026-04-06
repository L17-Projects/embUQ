# Vega acceptance checklist

This checklist is the in-repo acceptance layer for workflows that cannot be fully validated on public CI.

## Environment capture

Record:

- branch and commit SHA
- loaded module stack
- Python version
- MPI version
- CUDA toolkit version
- GPU type

## Required checks

### Install / import

- editable install succeeds
- `import meso_uq` succeeds
- `import mpi4py` succeeds in the same environment

### MPI execution

- a two-rank MPI smoke command runs successfully

### Phase 1 GPU-batched

- run the public Phase 1 path
- archive command, logs, and output location

### Phase 2 native-CUDA

- follow `docs/VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md`
- archive command, logs, and output location

### Phase 3b GPU-batched

- run the public Phase 3b path
- archive command, logs, and output location

### Propagation + plotting

- run the public propagation scripts
- generate overlay / posterior plots from the resulting outputs
- archive command, logs, and output location

## Pass condition

Do not mark Vega acceptance complete unless all required checks above have recorded commands, logs, and artifact paths.
