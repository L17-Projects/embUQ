# Getting started

## What MesoUQ is

`MesoUQ` is the public release line for Bayesian uncertainty quantification and calibration of mesoscopic DPD models. The repository combines:
- the shared Python package `meso_uq`
- workflow scripts for hierarchical inference
- reduced-workflow wrappers
- surrogate retraining and evaluation entrypoints
- sensitivity and postprocessing helpers
- a focused vendored Korali patch surface in `extern/korali/`

## What you need

At minimum, expect to need:
- Python 3.10 or 3.11
- `numpy`, `pyyaml`, `pydantic`
- `torch` for surrogate retraining/evaluation
- `SALib`, `pandas`, `matplotlib`, and `scipy` for sensitivity, sampling, and plotting
- `mpi4py` and MPI runtime for the MPI-oriented workflow scripts

Some parts of the workflow are CPU-only, some are MPI-aware, and some are designed to sit on top of GPU-enabled or HPC environments.

## Basic installation

Create and activate an environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Then install the scientific extras you actually need for your tasks. For example:

```bash
python -m pip install torch pandas matplotlib scipy SALib mpi4py
```

## Repository orientation

The most important top-level directories are:

- `src/meso_uq/` shared package code
- `inference/` full hierarchical workflow entrypoints and configs
- `reduced/` reduced-model configs and wrappers
- `compression/` compression-specific surrogate/evalkit surface
- `indentation/` indentation-specific surrogate/evalkit surface
- `sampling/` lightweight design-of-experiments helpers
- `propagation/` public plotting/postprocessing entrypoints
- `extern/korali/` focused vendored backend patch surface
- `tests/` public tests and smoke checks

## First commands to try

List the datasets enabled by a config:

```bash
python scripts/config/list_experiment_datasets.py --config inference/configs/production/inference_config_compression.yaml
```

Train a compression surrogate from a wide table:

```bash
python compression/surrogate/scripts/emb_train.py path/to/training_table.dat --out trained/microbubble_force_BEST.pkl
```

Train an indentation surrogate:

```bash
python indentation/surrogate/scripts/emb_train.py path/to/training_table.dat --out trained/microbubble_disp_BEST.pkl
```

Generate a lightweight Latin-hypercube design:

```bash
python sampling/run_LHS.py --output lhs_samples.csv
```

Extract a MAP sample from a Korali run directory:

```bash
python scripts/vega/extract_map.py \
  --experiment compression \
  --model-family full-model \
  --profile validation \
  --stage phase1 \
  --dataset compression_2.1um
```

Create a posterior marginal plot:

```bash
python propagation/scripts/plot_posterior_marginals.py posterior_samples.csv --output posterior_marginals.png
```

## Read next

- `docs/WORKFLOWS.md`
- `docs/VEGA_WORKFLOW_HELPERS.md`
- `docs/RELEASE_SCOPE.md`
- `docs/HPC_GPU_BATCHED_REDUCED_INDENTATION.md`
- `CONTRIBUTING.md`
