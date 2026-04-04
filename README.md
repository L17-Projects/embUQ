# MesoUQ
[![CI](https://github.com/BrieucB/MesoUQ/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BrieucB/MesoUQ/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](./pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-included-blueviolet.svg)](./docs/)
[![Linux](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#)
[![MPI](https://img.shields.io/badge/MPI-supported-orange.svg)](#)
[![CUDA](https://img.shields.io/badge/CUDA-optional-green.svg)](#)
[![SLURM](https://img.shields.io/badge/SLURM-supported-blue.svg)](#)
[![DPD](https://img.shields.io/badge/model-DPD-informational.svg)](#)
[![Hierarchical Bayes](https://img.shields.io/badge/inference-Hierarchical%20Bayes-purple.svg)](#)
MesoUQ is a release-grade software line for Bayesian uncertainty quantification and calibration of mesoscopic DPD models.

The repository is intended to support hierarchical Bayesian calibration workflows across multiple diameters and experiment classes, including ultrasound-facing applications.

## Scope

MesoUQ is designed to host:

- the Python package `meso_uq`
- release-grade workflow modules for compression, indentation, inference, propagation, reduced models, and sampling
- the imported Korali backend under `extern/korali/`
- runnable tests, documentation, and synchronization tooling

## Repository layout

```text
MesoUQ/
├── README.md
├── pyproject.toml
├── src/meso_uq/
├── tests/
├── docs/
├── extern/korali/
├── scripts/
├── sync/
└── upstream/
```

## Provenance

The release line is synchronized from `BrieucB/UQ_DPD`, with explicit upstream SHA tracking under `upstream/` and the synchronization contract recorded in `SYNC_MANIFEST.yaml`.
