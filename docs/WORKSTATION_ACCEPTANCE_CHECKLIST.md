# Workstation acceptance checklist

This checklist is the manual Linux NVIDIA workstation validation layer for the `v0.1.0` release line.

It exists because some release claims require real GPU hardware, but do not need the full Vega acceptance path.

## Scope

This checklist is intended for:

- a clean Linux environment with one accessible NVIDIA GPU
- the public `MesoUQ` repository checkout
- reduced-cost validation or smoke-scale runs that still execute the real public entrypoints

This checklist does **not** add a native-CUDA Phase 2 claim for `v0.1.0`.
That backend claim remains explicitly out of scope for this release line.

## Required checks

Record all of the following:

1. install the repo in a clean environment
2. run at least one public surrogate retraining path
3. run the public Phase 1 GPU-batched path
4. run the public Phase 3b GPU-batched path
5. run MAP extraction and at least one plotting step from the produced outputs

## What to archive

Archive enough evidence to reconstruct the workstation pass:

- the exact checked-out commit
- machine/environment metadata
- the exact commands run
- stdout/stderr logs or equivalent captured logs
- the key output artifact paths for each required step
- one machine-readable workstation acceptance report

## Machine-readable record

The canonical report filename is:

- `workstation_acceptance_report.json`

An example report is shipped at:

- `examples/reports/workstation_acceptance_report.example.json`

Validate a completed record with:

```bash
python scripts/workstation/validate_acceptance_record.py \
  --report /path/to/workstation_acceptance_report.json \
  --must-exist
```

## Minimum report structure

The report must contain:

- `target: workstation`
- final `status`
- `environment` metadata including hostname, platform, Python, GPU summary, and checked-out commit
- one entry for each required step
- top-level artifact paths for the archived report root

The required step names are:

- `install`
- `surrogate_retraining`
- `phase1_gpu_batched`
- `phase3b_gpu_batched`
- `map_extraction_and_plotting`

## Phase backend contract

- **Phase 1**: GPU-batched surrogate (Sequential Korali conduit) when `--device gpu`.
- **Phase 2**: **CPU-MPI only.** No native-CUDA Phase 2 is validated for this release line.
- **Phase 3b**: GPU-batched surrogate (Sequential Korali conduit) when `--device gpu`.
- **Propagation**: GPU surrogate where applicable.

## Low-load guidance

- Use **at most 9 CPUs** for Phase 2 MPI workers.
- Keep **>2 GB RAM free** at all times during a run.
- Do not run competing compilation or large data-transfer jobs concurrently.
- Use validation-profile configs (reduced cost), not production configs, for acceptance runs.

## Practical guidance

- Prefer the shipped validation configs or explicit reduced-cost overrides over ad hoc unpublished settings.
- Record any overrides that materially reduce runtime.
- If a step fails but the failure is informative, keep the logs and mark the step status honestly instead of deleting the attempt.
- Do not claim workstation acceptance complete unless every required step has a recorded command and artifact trail.
