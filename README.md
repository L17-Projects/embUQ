# MesoUQ
[![CI](https://github.com/BrieucB/MesoUQ/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BrieucB/MesoUQ/actions/workflows/ci.yml)
[![Release Smoke](https://github.com/BrieucB/MesoUQ/actions/workflows/release-smoke.yml/badge.svg?branch=main)](https://github.com/BrieucB/MesoUQ/actions/workflows/release-smoke.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](./pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-included-blueviolet.svg)](./docs/)
[![Linux](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#)
[![MPI](https://img.shields.io/badge/MPI-supported-orange.svg)](#)
[![CUDA](https://img.shields.io/badge/CUDA-optional-green.svg)](#)
[![SLURM](https://img.shields.io/badge/SLURM-supported-blue.svg)](#)
[![DPD](https://img.shields.io/badge/model-DPD-informational.svg)](#)
[![Hierarchical Bayes](https://img.shields.io/badge/inference-Hierarchical%20Bayes-purple.svg)](#)

MesoUQ is a public software line for Bayesian uncertainty quantification and calibration of mesoscopic DPD models.

The repository exposes a release-oriented workflow surface for compression and indentation calibration, hierarchical inference, surrogate retraining and model selection, MAP extraction, plotting, propagation, Vega-first acceptance, and supporting operator utilities.

## Public workflow surface

The current public release surface includes:

- surrogate retraining for compression and indentation
- lightweight surrogate model-selection tooling
- Phase 1 workflow execution
- Phase 2 workflow execution
- Phase 3b workflow execution
- MAP extraction and plotting utilities
- lightweight propagation execution for Phase 1 and Phase 3b
- a richer GPU/operator validation runner
- a thin Vega-first acceptance command with a machine-readable report
- dedicated tiny validation configs for full and reduced workflows
- vendored Korali build surface and backend notes
- public smoke tests and release-validation documentation

## Getting started

Install the package in editable mode with the extras that match your usage.

Basic package install:

```bash
pip install -e .
```

Local pytest/test install:

```bash
pip install -e ".[test]"
```

CI / smoke-test install:

```bash
pip install -e ".[ci]"
```

Plotting utilities:

```bash
pip install -e ".[plot]"
```

Surrogate / model-selection utilities:

```bash
pip install -e ".[surrogate]"
```

MPI support:

```bash
pip install -e ".[mpi]"
```

See `docs/DEPENDENCY_EXTRAS.md` for the full extras contract and the remaining Korali/backend caveats.

## Canonical configuration entrypoints

Full workflows:

- `inference/configs/production/inference_config_compression.yaml`
- `inference/configs/production/inference_config_indentation.yaml`

Reduced workflows:

- `reduced/configs/production/reduced_config_compression.yaml`
- `reduced/configs/production/reduced_config_indentation.yaml`

Validation workflows:

- `inference/configs/validation/validation_config_compression.yaml`
- `inference/configs/validation/validation_config_indentation.yaml`
- `reduced/configs/validation/validation_config_compression.yaml`
- `reduced/configs/validation/validation_config_indentation.yaml`

A small release-oriented example bundle is also provided under `examples/configs/`.

## Documentation index

Start here:

- `docs/README.md`
- `docs/RELEASE_NOTES_v0.1.0.md`
- `docs/DEPENDENCY_EXTRAS.md`
- `docs/VALIDATION_MATRIX.md`
- `docs/VEGA_ACCEPTANCE_COMMAND.md`
- `docs/VEGA_ACCEPTANCE_CHECKLIST.md`
- `docs/VALIDATION_CONFIGS.md`
- `docs/HPC_GPU_BATCHED_REDUCED_INDENTATION.md`
- `docs/SURROGATE_MODEL_SELECTION.md`
- `docs/PROPAGATION_EXECUTION.md`

## Repository layout

```text
MesoUQ/
├── README.md
├── pyproject.toml
├── src/meso_uq/
├── tests/
├── docs/
├── examples/
├── scripts/
├── extern/korali/
├── compression/
├── indentation/
├── inference/
├── propagation/
├── reduced/
└── upstream/
```

## Provenance

The public release line is synchronized from `BrieucB/UQ_DPD`, with explicit upstream-tracking material preserved in the repository history and release hardening work captured through the PR sequence leading to `v0.1.0`.
