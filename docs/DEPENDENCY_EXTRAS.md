# Dependency extras

This page explains the install extras declared in `pyproject.toml` and the intended contract for local testing and CI.

## Extras

### `.[plot]`

Install the plotting-oriented stack used by the public postprocessing utilities.

Includes:

- `matplotlib`
- `pandas`

### `.[surrogate]`

Install the surrogate/model-selection stack used by the public surrogate utilities.

Includes:

- `pandas`
- `scipy`
- `torch`

### `.[mpi]`

Install the MPI Python layer used by the MPI smoke path.

Includes:

- `mpi4py`

System MPI libraries are still expected to be installed separately on the host.

### `.[test]`

Install enough Python dependencies to run all committed pytest tests locally.

This is the authoritative local test contract.

### `.[ci]`

Install the Python dependency set used by the GitHub-hosted CI smoke and release-smoke workflows.

At the moment, this intentionally matches the local pytest dependency surface closely.

### `.[dev]`

Install a broader development stack used for local development and debugging.

## Important note on Korali/backend expectations

The package extras do **not** currently make the full Korali backend available purely through pip.

For the full workflow and Vega acceptance paths, the environment still needs:

- a locally built Korali Python path
- system MPI libraries
- CUDA/NVIDIA runtime support where applicable

For Vega, the supported bootstrap path is now repo-managed:

- build vendored `extern/korali/`
- install it into repo-local `_vega/korali/install`
- source `_vega/korali/env.sh`

See `VEGA_BOOTSTRAP.md` for the exact commands and helper scripts.
