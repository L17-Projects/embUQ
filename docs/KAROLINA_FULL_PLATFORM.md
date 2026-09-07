# Karolina full-platform enablement

This page defines the initial Karolina acceptance contract. Karolina is a first-class execution platform for EMB workflows, surrogate utilities, hierarchical inference, Mirheo, mirheoOBMD, and machine-readable validation evidence.

## Storage policy

Keep the git clone and small source files in HOME. Put runtime installs, generated workflow data, validation reports, and staged provenance in project scratch:

```bash
export MESOUQ_SITE=karolina
export MESOUQ_PROJECT_ID=eu-26-17
export MESOUQ_SCRATCH_ROOT=/scratch/project/eu-26-17/eubrieucb/mesouq
export MESOUQ_RUNTIME_TAG=unified-platform-env/$(date -u +%Y%m%dT%H%M%SZ)
export MESOUQ_SITE_RUNTIME_ROOT="${MESOUQ_SCRATCH_ROOT}/${MESOUQ_RUNTIME_TAG}/runtime"
export MESOUQ_RUNS_ROOT="${MESOUQ_SCRATCH_ROOT}/${MESOUQ_RUNTIME_TAG}/runs"
export MESOUQ_PROVENANCE_ROOT="${MESOUQ_SCRATCH_ROOT}/${MESOUQ_RUNTIME_TAG}/provenance"
```

The site-neutral runtime helper resolves Karolina bootstrap state under `MESOUQ_SITE_RUNTIME_ROOT`. That variable is required for Karolina runtime discovery; missing it is a hard error instead of falling back to clone-local paths:

- `${MESOUQ_SITE_RUNTIME_ROOT}/env`
- `${MESOUQ_SITE_RUNTIME_ROOT}/korali`
- `${MESOUQ_SITE_RUNTIME_ROOT}/mirheo`

The supported activation contract is site-neutral:

```bash
export MESOUQ_SITE=karolina
export MESOUQ_SITE_RUNTIME_ROOT=<karolina-runtime-root>
source scripts/platforms/hpc/site_env.sh
mesouq_activate_site_env karolina "$PWD"
```

After activation, `python`, `PYTHON_BIN`, `MESOUQ_ENV_ROOT`,
`MESOUQ_ENV_SCRIPT`, and `MESOUQ_GV_ENV_SCRIPT` all resolve through the
canonical `${MESOUQ_SITE_RUNTIME_ROOT}/env` environment. Non-canonical
`MESOUQ_ENV_ROOT`, `MESOUQ_ENV_SCRIPT`, or `MESOUQ_GV_ENV_SCRIPT` overrides are
rejected by `site_env.sh`.

`MESOUQ_PROVENANCE_ROOT` is explicit and site-aware. On Karolina it should point to scratch-accessible provenance staging that matches the chosen isolated runtime/run tag, and generated runtime env scripts export the resolved value.

## Slurm policy

Use the project/account that has Karolina GPU allocation:

```bash
#SBATCH --account=eu-26-17
#SBATCH --partition=qgpu
#SBATCH --gpus=1
```

One requested GPU on Karolina maps to one eighth of an accelerated node: 1 A100 GPU, 16 CPU cores, and the corresponding memory allocation. Use `qgpu_exp` for short tests, `qgpu` for standard GPU work, `qgpu_free` for free-resource GPU work, and `qgpu_preempt` only for re-runnable preemptible jobs. Use `qgpu_big` only for jobs above 16 GPU nodes.

The `eu-26-17` account is currently GPU-only from this login context. CPU, fat-memory, and visualization partitions are visible but are not submit-accessible with this account.

Karolina GPU sbatch templates should use the Karolina policy directive
`#SBATCH --gpus=<n>`, not Vega-style `#SBATCH --gres=gpu:<n>`. The surrogate
group-holdout and repository-bootstrap templates request one GPU because they run
single-task orchestration or setup work; they do not consume a full eight-GPU
node.

Do not expect `/ceph/hpc/home/eubrieucb` to be mounted on Karolina. That path is not part of the Karolina acceptance contract and should not be treated as a missing platform feature.

## Runtime env-script contract

Generated EMB runtime commands source the canonical env script:

```bash
"${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh"
```

`MESOUQ_SITE_RUNTIME_ROOT` is required and the canonical script is resolved
under that root. Vega compatibility is preserved through `get_vega_paths()`,
but resolved paths now come from the canonical site runtime root.

Standard Mirheo and mirheoOBMD follow the Vega runtime policy: they are separate lane/process imports, not same-interpreter imports. Validation should import each module in a separate subprocess.

## Initial evidence target

Karolina acceptance evidence must be machine-readable and include:

- source branch, commit, and clean/dirty state;
- Slurm account, QOS, partition, GPU allocation, and node metadata;
- runtime doctor report for Python, CUDA, OpenMPI, GSL/Eigen/HDF5, Korali, Mirheo, mirheoOBMD, h5py, MDAnalysis, and trimesh;
- EMB validation matrix and production-sanity report paths;
- explicit blockers, especially missing Vega provenance or cross-site runtime mismatches.
