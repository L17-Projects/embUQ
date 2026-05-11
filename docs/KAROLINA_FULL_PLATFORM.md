# Karolina full-platform enablement

This page defines the initial Karolina acceptance contract. Karolina is intended to become a first-class MesoUQ execution platform for EMB workflows, surrogate utilities, hierarchical inference, Mirheo, mirheoOBMD, GV paper replay, and machine-readable validation evidence.

## Storage policy

Keep the git clone and small source files in HOME. Put runtime installs, generated workflow data, validation reports, and staged provenance in project scratch:

```bash
export MESOUQ_SITE=karolina
export MESOUQ_PROJECT_ID=eu-26-17
export MESOUQ_SCRATCH_ROOT=/scratch/project/eu-26-17/eubrieucb/mesouq
export MESOUQ_SITE_RUNTIME_ROOT="${MESOUQ_SCRATCH_ROOT}/runtime"
export MESOUQ_RUNS_ROOT="${MESOUQ_SCRATCH_ROOT}/runs"
```

The site-neutral runtime helper resolves Karolina bootstrap state under `MESOUQ_SITE_RUNTIME_ROOT` when set. Without that override it uses clone-local `_karolina/` paths:

- `_karolina/venv`
- `_karolina/korali`
- `_karolina/mirheo`
- `_karolina/gv_venv`

## Slurm policy

Use the project/account that has Karolina GPU allocation:

```bash
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --gpus=1
```

One requested GPU on Karolina maps to one eighth of an accelerated node: 1 A100 GPU, 16 CPU cores, and the corresponding memory allocation. Use `qgpu_exp` for short tests, `qgpu` for standard GPU work, `qgpu_free` for free-resource GPU work, and `qgpu_preempt` only for re-runnable preemptible jobs. Use `qgpu_big` only for jobs above 16 GPU nodes.

The `eu-26-17` account is currently GPU-only from this login context. CPU, fat-memory, and visualization partitions are visible but are not submit-accessible with this account.

## Runtime env-script contract

Generated GV runtime commands must source the explicit `MESOUQ_GV_ENV_SCRIPT` when set. Otherwise they resolve the site runtime root:

```bash
"${MESOUQ_SITE_RUNTIME_ROOT}/gv_venv/env.sh"
```

If no override is set, the fallback is clone-local and site-qualified:

```bash
_karolina/gv_venv/env.sh
```

Vega compatibility is preserved through `get_vega_paths()` and clone-local `_vega/` defaults.

## Initial evidence target

Karolina acceptance evidence must be machine-readable and include:

- source branch, commit, and clean/dirty state;
- Slurm account, QOS, partition, GPU allocation, and node metadata;
- runtime doctor report for Python, CUDA, OpenMPI, GSL/Eigen/HDF5, Korali, Mirheo, mirheoOBMD, h5py, MDAnalysis, trimesh, and GV geometry tools;
- EMB validation matrix and production-sanity report paths;
- GV non-shear canary report paths for stretching, buckling, torsion, and eigenmodes;
- explicit blockers, especially missing Vega provenance or unresolved eigenmodes paper-replay mismatch.

