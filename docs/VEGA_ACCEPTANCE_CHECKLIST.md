# Vega acceptance checklist

This checklist is the in-repo acceptance layer for workflows that cannot be fully validated on public CI.

## Primary command

The preferred entrypoint is the single Vega-first acceptance command:

```bash
python scripts/run_vega_acceptance.py --output-root ../vega_acceptance
```

That command wraps the richer operator runner and writes:

- `vega_acceptance_report.json`
- step logs
- workflow summary artifacts
- packaged workflow outputs

See `VEGA_ACCEPTANCE_COMMAND.md` for the command contract.

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

- run the public Phase 1 path, ideally through the Vega acceptance command
- archive command, logs, and output location

### Phase 2 native-CUDA

- follow `docs/VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md`
- archive command, logs, and output location

### Phase 3b GPU-batched

- run the public Phase 3b path, ideally through the Vega acceptance command
- archive command, logs, and output location

### Propagation + plotting

- run the public propagation scripts or acceptance wrapper path
- generate overlay / posterior plots from the resulting outputs
- archive command, logs, and output location

## Pass condition

Do not mark Vega acceptance complete unless all required checks above have recorded commands, logs, and artifact paths.

For the preferred acceptance path, archive the machine-readable `vega_acceptance_report.json` alongside the workflow artifacts.
