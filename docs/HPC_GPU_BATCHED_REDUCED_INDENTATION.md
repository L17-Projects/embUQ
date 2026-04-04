# HPC Install and GPU-Batched Reduced Indentation Workflow

This guide explains how to:

1. perform a clean user-local install of the modified Korali fork on an HPC system,
2. prepare the dev workflow for GPU-batched execution,
3. retrain the `3.4um` indentation surrogate when new training data arrives,
4. run the full reduced indentation workflow on a single node with one `A40` or `L40S` GPU,
5. skip `Phase 3a`, but still produce `Phase 3b`, MAP extraction, posterior plots, propagation, and overlay plots.

This is the recommended operator path for the current GPU-batched reduced indentation workflow.

## Scope

- Primary install path: `conda`
- Fallback install path: `venv`
- Runtime scope: single-node SLURM jobs
- GPU target: one `A40` or one `L40S`
- Workflow target: reduced indentation only
- Workflow mode: `Phase 1` + `Phase 2` + `Phase 3b` + propagation + MAP postprocessing
- `Phase 3a`: skipped intentionally

## Repositories and branches

You need both repositories:

- `BrieucB/UQ_DPD`, branch `gpu-batching-dev`
- `BrieucB/korali`, branch `gpu-batch-eval`

Recommended layout on the HPC filesystem:

```text
$HOME/work/UQ_DPD/
├── Hierarchical_UQ_compression_dev/
└── korali/
```

## What is GPU-accelerated

For this workflow, Korali itself remains a CPU/MPI library. The GPU acceleration happens in the PyTorch surrogates called from Python.

That means:

- `Phase 1`: GPU batched
- `Phase 2`: CPU MPI
- `Phase 3a`: skipped
- `Phase 3b`: GPU batched
- propagation from `Phase 1` and `Phase 3b`: GPU batched
- plotting and MAP postprocessing: CPU

This is why you should build Korali with MPI and MPI4Py, but you do not need to enable Korali’s own CUDA/cuDNN options for this workflow.

## Clean install on the login node

Assumptions:

- no root privileges
- compute nodes may be offline
- software is installed once on the login node
- jobs only activate and use the prepared environment

### 1. Load an approximate module stack

Use your site’s actual module names, but this is the intended stack:

- GCC `12.x`
- OpenMPI `4.1.x`
- CMake `3.24+`
- CUDA `12.2` or `12.3`
- optional site packages for `GSL`, `Eigen`, `HDF5`

### 2. Create and activate the conda environment

The existing runtime wrappers assume `conda activate env3.8`, so keep that name unless you also patch the wrappers.

Recommended Python range for this workflow:

- Python `3.10` or `3.11`

### 3. Install Python dependencies

Install the core scientific stack and a GPU-enabled PyTorch build in `conda`, then build `mpi4py` against the loaded MPI.

The relevant dependencies in the upstream workflow are:

- `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml`, `h5py`, `pydantic`
- `torch`
- `mpi4py`

### 4. Build and install the modified Korali fork

For this workflow, the required source tree is the modified branch in the Korali fork, not upstream vanilla Korali.

### 5. Runtime environment variables

At runtime you typically need:

```bash
export HUQ_ROOT=$HOME/work/UQ_DPD/Hierarchical_UQ_compression_dev
export KORALI_ROOT=$HOME/work/UQ_DPD/korali
export KORALI_PREFIX=$HOME/software/korali-gpu-batch
export KORALI_PYTHONPATH=$(find "$KORALI_PREFIX" -type d -path '*/site-packages' | head -n 1)
export PYTHONPATH=$KORALI_PYTHONPATH:$HUQ_ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$KORALI_PREFIX/lib64:$KORALI_PREFIX/lib:${LD_LIBRARY_PATH:-}
```

## Recommended operator path for reduced indentation

### Case A: new `3.4um` training data arrived

Use `inference/scripts/run_indentation_reduced_refresh.py` to stage the new data, retrain the surrogate, skip Phase 3a, and produce the reduced indentation workflow outputs.

### Case B: the surrogate is already retrained and you only want the workflow

Use the same helper with `--skip-training`.

## Important output locations

The refresh helper creates a fresh timestamped root under an output directory with:

- refresh backups
- a refresh manifest
- a derived workflow config
- the reduced indentation workflow output

## Troubleshooting

### `import korali` fails

Usually `PYTHONPATH` is missing the installed Korali site-packages directory.

### `mpi4py` or MPI launcher errors

Rebuild `mpi4py` after loading the same MPI module stack used for Korali.

### GPU utilization looks low during `Phase 2`

That is expected. `Phase 2` is CPU MPI only.

### GPU utilization looks bursty during `Phase 1` or `Phase 3b`

That is also expected. The surrogate runs in batched kernels, but orchestration and plotting still happen on the CPU.
