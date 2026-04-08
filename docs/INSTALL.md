# Installation

This document is the practical installation contract for the current public alpha line.

## Supported Python versions

The intended public Python range is:
- Python 3.10
- Python 3.11

## Installation modes

## 1. Minimal package install

For light package inspection, docs, and non-heavy utilities:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Current `pyproject.toml` covers only the minimal base dependencies. For real scientific use, install the relevant workflow extras manually until the package metadata is widened in the next hardening pass.

## 2. Surrogate / plotting / analysis install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install torch pandas matplotlib scipy SALib
```

Use this mode for:
- surrogate retraining and evaluation
- sensitivity analysis
- MAP extraction
- plotting and postprocessing
- lightweight sampling design

## 3. MPI-oriented workflow install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install torch pandas matplotlib scipy SALib mpi4py
```

You also need a working MPI runtime on the machine or cluster.

Use this mode for:
- MPI smoke tests
- hierarchical workflow scripts
- cluster-side execution

## 4. GPU / HPC install notes

For GPU-oriented or cluster-side execution, the Python environment is only one part of the install contract.
You also need:
- the correct NVIDIA driver and CUDA stack for the target machine
- a working MPI stack where relevant
- a build/runtime path compatible with the vendored Korali surface

The public alpha line does not yet encode CUDA runtime dependencies inside `pyproject.toml`.
Those paths remain environment-specific and must be validated on the actual target machine.

## Suggested environment files

This repo now ships small environment definitions under `environments/`:
- `cpu-dev.yml`
- `analysis.yml`
- `mpi.yml`

These are convenience starting points, not a guarantee that every HPC system matches them exactly.

## Sanity checks after install

Try these in order:

```bash
python -c "import meso_uq; print('meso_uq import ok')"
python scripts/config/list_experiment_datasets.py --config inference/configs/production/inference_config_compression.yaml
python sampling/run_LHS.py --output lhs_samples.csv
```

For analysis installs, also try:

```bash
python -c "import torch, pandas, scipy, SALib, matplotlib; print('analysis stack ok')"
```

For MPI installs:

```bash
mpirun --oversubscribe -np 2 python -c "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())"
```

## Important current limitation

The package metadata and the public runtime surface are still being reconciled. This means the installation contract is now documented here, but not yet fully encoded in the package extras metadata. That is being hardened as part of the v0.1.0 release work.
