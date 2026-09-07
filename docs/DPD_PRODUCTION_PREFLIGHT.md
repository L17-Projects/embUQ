# DPD Production Preflight

This document defines the runtime preflight gate for production DPD candidate arrays.

The gate exists to catch execution-context failures before Mirheo starts. It is not a
scientific acceptance test and must not be used to convert failed DPD outputs into
training or scoring evidence.

## Required Checks

Every production candidate array that writes fresh DPD labels must check:

- output root is under an allowed scratch root;
- output root is writable with a write/fsync probe;
- free space is above the configured minimum;
- nested Slurm CPU environment is sanitized before nested `srun`;
- HDF5 can write and read a smoke-test file.

The default HDF5 smoke driver is `serial`, because the Karolina production Python stack
does not provide MPI-enabled `h5py`. Operators can explicitly request the stricter
`mpio` driver with `DPD_PREFLIGHT_HDF5_DRIVER=mpio` when that stack is available.

## Entry Points

Shared implementation:

```bash
python scripts/workflows/dpd/run_dpd_production_preflight.py \
  --output-root <scratch-output-root> \
  --scratch-root <allowed-scratch-root> \
  --hdf5-smoke-test
```

Shared Slurm canary dispatcher:

```bash
export MESOUQ_SITE=karolina
export MESOUQ_SITE_RUNTIME_ROOT=<karolina-runtime-root>
bash scripts/platforms/hpc/sbatch/dpd_production_preflight_canary.sbatch --site karolina
```

Vega must use the same dispatcher with an explicit Vega scratch root when the site
environment does not export one:

```bash
MESOUQ_SITE=vega \
MESOUQ_SITE_RUNTIME_ROOT=<vega-runtime-root> \
MESOUQ_SCRATCH_ROOT=<vega-scratch-root> \
bash scripts/platforms/hpc/sbatch/dpd_production_preflight_canary.sbatch --site vega
```

Submit the shared dispatcher with `bash`, not `sbatch`. It selects the site and then
submits the correct site-specific Slurm template so scheduler directives are preserved.
Each site-specific template activates the canonical runtime through
`scripts/platforms/hpc/site_env.sh`; old split env paths are not supported
fallbacks.

## Current Evidence Gate

Karolina has passed the preflight canary:

- Slurm job: `4394152`
- manifest: `/scratch/project/eu-26-17/eubrieucb/mesouq/runs/dpd_preflight_canary/20260528_karolina_preflight_retry1/dpd_production_preflight.json`
- status: `passed`

The first Karolina canary, job `4394149`, failed usefully: it caught unsanitized Slurm
CPU env and an overly strict default `mpio` HDF5 check. The wrappers now sanitize the
environment before preflight and default to serial HDF5 smoke.

Vega has passed the same canary under the canonical site runtime. Vega GPU submissions
must keep the `gn10` exclusion until support clears that node or an equivalent
operator-reviewed policy replaces it.

## Karolina Array Packing Guard

Karolina EMB DPD production wrappers also use an allocation guard that is separate from
the preflight checks:

- independently reproduced bad GPU nodes are excluded in the Karolina wrappers;
- EMB DPD arrays use `#SBATCH --exclusive` so Slurm does not pack multiple one-GPU
  DPD candidates onto the same GPU node;
- a small array-task launch stagger remains as a secondary guard.

This is intentionally Karolina-specific scheduler policy. Controlled reruns showed that
candidate manifests can complete on nodes that previously saw signal-15 failures when
the same-node packed launch pattern is avoided. Do not remove the exclusive allocation
guard without replacing it with equivalent Slurm evidence.
