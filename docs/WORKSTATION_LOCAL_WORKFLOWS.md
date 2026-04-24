# Workstation Local Workflows (o369)

This page defines the **local non-SLURM** workflow path for workstation runs (for example `o369`).

It is an alternative run path and does not replace any Vega sbatch workflow.

## What "validation profile" means

A workflow selection in this repository has three axes:

- `experiment`: `compression` or `indentation`
- `model-family`: `full-model` or `reduced-model`
- `profile`: `production` or `validation`

The `validation` profile means "reduced-cost configs intended for acceptance/smoke usage", not the production-cost configs.

## Phase backend contract

Each phase has a distinct compute backend contract:

| Phase | Backend | Notes |
|-------|---------|-------|
| Phase 1 | **GPU-batched surrogate** (when `--inference-device gpu`) or CPU MPI surrogate | Single-process Sequential Korali conduit on GPU; Distributed MPI conduit on CPU |
| Phase 2 | **Validation-profile default: CPU MPI** | The broader workflow spine now supports `cpu-mpi` and `native-cuda`, but this local validation runner exercises the validation-profile default `phase2_backend=cpu-mpi`. |
| Phase 3b | **GPU-batched surrogate** (when `--inference-device gpu`) or CPU MPI surrogate | Same conduit choice as Phase 1 |
| Propagation Phase 3b | **GPU surrogate** (when `--propagation-device gpu`) or CPU | Lightweight; GPU reduces wall time |

This workstation page is a validation-profile smoke surface. It does not, by itself, certify the
production native-CUDA `Phase 2` path.

## GPU batching scope

GPU batching (`--device gpu`) applies to:

- **Phase 1**: `run_phase_1.py --device gpu` — batched TMCMC surrogate evaluation
- **Phase 3b**: `run_phase_3b.py --device gpu` — batched TMCMC surrogate evaluation
- **Propagation**: where applicable, `--device gpu` enables GPU-batched forward surrogate calls

For this local validation runner, `Phase 2` remains on the validation-profile default CPU-MPI path.

## Local validation target outputs

The local runner targets four validation lanes:

| Lane | Experiment | Model family |
|------|-----------|--------------|
| 1 | `compression` | `full-model` |
| 2 | `compression` | `reduced-model` |
| 3 | `indentation` | `full-model` |
| 4 | `indentation` | `reduced-model` |

Each lane runs the full pipeline:

```
phase1 → phase2 → phase3b → propagation_phase3b → map_phase3b
```

Expected artifacts per lane (under `<output-root>/<lane>/`):

- `results_phase_1/<exp>_<diam>um/latest` — per-diameter posterior samples
- `results_phase_2/latest` — hierarchical Psi posterior
- `results_phase_3b/<exp>_<diam>um/latest` — per-diameter Theta posteriors
- `results_propagation_phase3b/<exp>_<diam>um/` — forward uncertainty propagation
- `results_map_phase3b/<exp>_<diam>um/` — MAP parameter estimates
- `local_validation_report.json` — machine-readable summary report

The local validation report records:

- `phase2_backend_contract: dual_backend`
- `phase2_backend_default_for_profile: cpu-mpi`
- `phase2_backend_effective: cpu-mpi`

## Low-load guidance

To keep the workstation usable during a local validation run:

- Use **at most 9 CPUs** for Phase 2 MPI ranks (`--phase2-cpu-ranks 9` or fewer).
- Leave **at least 2 GB RAM free** at all times; the GPU surrogate and MPI workers compete for system memory.
- Do **not** run heavy background workloads (compilation, large data transfers) concurrently.
- Use the `validation` profile (not `production`) for local runs — the reduced-cost configs are designed for this context.

## Local runner

Use:

```bash
python scripts/platforms/workstation/run_local_validation_matrix.py \
  --output-root _o369_runs \
  --python-bin /temp/brieuc/workspace/myenv_torchfix/bin/python \
  --phase2-cpu-ranks 2 \
  --inference-device gpu \
  --propagation-device gpu
```

Default behavior:

- runs all 4 validation lanes:
  - `compression:full-model:validation`
  - `compression:reduced-model:validation`
  - `indentation:full-model:validation`
  - `indentation:reduced-model:validation`
- applies local smoke overrides to shorten runtime
- runs:
  - `phase1 -> phase2 -> phase3b -> propagation_phase3b -> map_phase3b`
- generates overlays from produced artifacts:
  - propagation uncertainty vs reference
  - MAP surrogate prediction vs reference
- writes machine-readable report:
  - `_o369_runs/local_validation_report.json`
