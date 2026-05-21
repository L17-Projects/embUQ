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
export MESOUQ_PROVENANCE_ROOT="${MESOUQ_SCRATCH_ROOT}/provenance"
```

The site-neutral runtime helper resolves Karolina bootstrap state under `MESOUQ_SITE_RUNTIME_ROOT`. That variable is required for Karolina runtime discovery; missing it is a hard error instead of falling back to clone-local paths:

- `${MESOUQ_SITE_RUNTIME_ROOT}/env`
- `${MESOUQ_SITE_RUNTIME_ROOT}/korali`
- `${MESOUQ_SITE_RUNTIME_ROOT}/mirheo`
- `${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools`

`MESOUQ_PROVENANCE_ROOT` is explicit and site-aware. On Karolina it should point to scratch-accessible provenance staging (default `${MESOUQ_SCRATCH_ROOT}/provenance` from `env_karolina.sh`), and generated runtime env scripts export the resolved value.

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

Generated GV runtime commands must source the explicit `MESOUQ_GV_ENV_SCRIPT` when set. Otherwise they resolve the site runtime root:

```bash
"${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh"
```

If no override is set, `MESOUQ_SITE_RUNTIME_ROOT` is required and the canonical script is resolved under that root. Vega compatibility is preserved through `get_vega_paths()`, but resolved paths now come from the canonical site runtime root.

GV geometry tooling is staged under:

```bash
"${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools/bin/scale_space"
```

When present, `${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools/env.sh` exports `GV_SCALE_SPACE_BINARY`, `GV_CGAL_TOOLS_ROOT`, updates `PATH`, and adds the Karolina MPFR/GMP runtime library paths needed by the CGAL `scale_space` binary.

Standard Mirheo and mirheoOBMD follow the Vega runtime policy: they are separate lane/process imports, not same-interpreter imports. Non-shear GV lanes use `mirheo` by default; `shear_flow` uses `mirheoOBMD`. Validation should import each module in a separate subprocess.

## Initial evidence target

Karolina acceptance evidence must be machine-readable and include:

- source branch, commit, and clean/dirty state;
- Slurm account, QOS, partition, GPU allocation, and node metadata;
- runtime doctor report for Python, CUDA, OpenMPI, GSL/Eigen/HDF5, Korali, Mirheo, mirheoOBMD, h5py, MDAnalysis, trimesh, and GV geometry tools;
- EMB validation matrix and production-sanity report paths;
- GV non-shear canary report paths for stretching, buckling, torsion, and eigenmodes;
- explicit blockers, especially missing Vega provenance or unresolved eigenmodes paper-replay mismatch.
