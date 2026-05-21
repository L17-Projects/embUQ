# Installation

This document is the practical installation contract for the current public release line.

## Supported Python versions

The supported public Python range is:

- Python 3.10
- Python 3.11

## Editable install

Start from a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

## Extras contract

The package metadata already declares the public extras in `pyproject.toml`.

Minimal package install:

```bash
pip install -e .
```

Plotting/postprocess utilities:

```bash
pip install -e ".[plot]"
```

Surrogate/model-selection utilities:

```bash
pip install -e ".[surrogate]"
```

MPI-aware workflow support:

```bash
pip install -e ".[mpi]"
```

Local pytest contract:

```bash
pip install -e ".[test]"
```

CI-parity Python dependency contract:

```bash
pip install -e ".[ci]"
```

For Vega bootstrap work, the practical combination is usually:

```bash
pip install -e ".[test,mpi]"
pip install pybind11 meson ninja
```

## Host-side requirements that extras do not provide

The Python extras do not ship the full Korali backend or system MPI/CUDA runtimes.

For workflow execution on GPU/HPC targets you still need:

- a working MPI runtime
- the appropriate NVIDIA/CUDA stack where applicable
- a Korali build compatible with the vendored `extern/korali/` tree

For Vega, the supported path is repo-managed and clone-local:

- build vendored `extern/korali/`
- install it under `${MESOUQ_SITE_RUNTIME_ROOT}/korali/install`
- source `${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh`

Use [VEGA_BOOTSTRAP.md](VEGA_BOOTSTRAP.md) for the exact bootstrap commands.

## Convenience environment files

This repo also ships small environment definitions under `environments/`:

- `cpu-dev.yml`
- `analysis.yml`
- `mpi.yml`

These are convenience starting points, not a guarantee that every HPC system matches them exactly.

## Sanity checks after install

Try these in order:

```bash
python -c "import meso_uq; print('meso_uq import ok')"
python scripts/shared/config/list_experiment_datasets.py --config inference/configs/production/inference_config_compression.yaml
python sampling/run_LHS.py --output lhs_samples.csv
```

For surrogate/plot installs:

```bash
python -c "import torch, pandas, scipy, matplotlib; print('surrogate and plot stack ok')"
```

For MPI installs:

```bash
mpirun --oversubscribe -np 2 python -c "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())"
```
