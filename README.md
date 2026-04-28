# MesoUQ
[![CI](https://github.com/BrieucB/MesoUQ/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BrieucB/MesoUQ/actions/workflows/ci.yml)
[![Release Smoke](https://github.com/BrieucB/MesoUQ/actions/workflows/release-smoke.yml/badge.svg?branch=main)](https://github.com/BrieucB/MesoUQ/actions/workflows/release-smoke.yml)
[![codecov](https://codecov.io/github/BrieucB/MesoUQ/graph/badge.svg?token=WNXV45WSWM)](https://codecov.io/github/BrieucB/MesoUQ)
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

The repository exposes a release-oriented workflow surface for compression and indentation calibration, hierarchical inference, surrogate retraining and model selection, MAP extraction, plotting, propagation, a public Vega validation matrix, Vega-first acceptance, and supporting operator utilities.

## Validation split

The release validation contract is intentionally split across two environments:

- GitHub CI is the fast PR gate. It proves package/tests/docs health, one real CPU micro workflow lane, and one real public surrogate retraining smoke.
- Vega remains the full workflow proof surface for the broader validation matrix, acceptance, production sanity, and hardware-specific backend claims.

The GitHub CI test job also publishes line coverage for the committed Python surface across `src/meso_uq`, `compression`, `indentation`, `inference`, `propagation`, `reduced`, and `scripts`, excluding vendored code, tests, and data-only directories.

For this private repo, Codecov uploads support either GitHub OIDC or a repository secret named `CODECOV_TOKEN`. The README badge itself also needs the private badge token from Codecov's `Badges & Graphs` settings appended to the badge URL query string before it will render real coverage instead of `unknown`.

## Public workflow surface

The current public release surface includes:

- surrogate retraining for compression and indentation
- lightweight surrogate model-selection tooling
- Phase 1 workflow execution
- Phase 2 workflow execution
- Phase 3b workflow execution
- MAP extraction and plotting utilities
- lightweight propagation execution for Phase 1 and Phase 3b
- a public Vega validation matrix that runs the real validation workflows
- a richer GPU/operator validation runner
- a thin Vega-first acceptance command with a machine-readable report
- dedicated tiny validation configs for full-model and reduced-model workflows
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

Local coverage run:

```bash
python -m coverage run -m pytest
python -m coverage report --skip-covered
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

Sensitivity utilities:

```bash
pip install -e ".[sensitivity]"
```

MPI support:

```bash
pip install -e ".[mpi]"
```

See `docs/DEPENDENCY_EXTRAS.md` for the full extras contract and the remaining Korali/backend caveats.

For a fresh Vega clone, the supported bootstrap path is documented in `docs/VEGA_BOOTSTRAP.md` and builds vendored `extern/korali/` into repo-local `_vega/`.

## Quick start

To test the implementation on Vega from a fresh clone, the shortest supported path is:

```bash
module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  openmpi/4.1.2.1 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0

python -m venv _vega/venv
source _vega/venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test,mpi]"
pip install pybind11 meson ninja

python scripts/platforms/vega/doctor_vega.py
bash scripts/platforms/vega/bootstrap_korali.sh --jobs 8
source _vega/korali/env.sh
python scripts/platforms/vega/doctor_vega.py --strict

REPO_ROOT=$(pwd) sbatch scripts/platforms/vega/sbatch/validation_matrix.sbatch
```

That submission runs the public validation workflow matrix on the Vega `dev` partition and writes the machine-readable report to `_vega/validation_matrix/workflow_matrix_report.json`.

If you are already inside an allocated Vega job, replace the last line with:

```bash
python scripts/platforms/vega/run_validation_matrix.py \
  --experiments compression indentation \
  --model-families full-model reduced-model \
  --output-root _vega/validation_matrix \
  --phase2-cpu-ranks 4
```

## GitHub CI real canaries

The default GitHub CI lane now includes two real canaries:

- one CPU workflow lane for `compression:reduced-model:validation`, driven by `reduced/configs/ci/ci_canary_config_compression.yaml`
- one public surrogate retraining smoke, driven by `compression/surrogate/ci/retraining_smoke.yaml`

These are intentionally much smaller than the Vega validation matrix, but they still run the public entrypoints and assert real artifacts.

## First Vega validation

After the repo-local Vega bootstrap is complete, the first real validation step is the public validation matrix. It runs the shipped validation workflows across compression and indentation, full-model and reduced-model, and executes the real Korali path for:

- Phase 1
- MAP extraction from Phase 1 outputs
- Phase 2
- Phase 3b
- propagation Phase 3b
- MAP extraction from Phase 3b outputs

Direct command inside an allocated Vega job:

```bash
python scripts/platforms/vega/run_validation_matrix.py \
  --output-root _vega/validation_matrix \
  --phase2-cpu-ranks 4
```

Tracked `sbatch` template:

```bash
REPO_ROOT=$(pwd) sbatch scripts/platforms/vega/sbatch/validation_matrix.sbatch
```

The machine-readable report is written to `_vega/validation_matrix/workflow_matrix_report.json`. Detailed usage and artifact layout are documented in `docs/VEGA_VALIDATION_MATRIX.md`.

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
- `CONTRIBUTING.md`
- `SECURITY.md`
- `docs/RELEASE_NOTES_v0.1.0.md`
- `docs/RELEASE_EVIDENCE_CHECKLIST_v0.1.0.md`
- `docs/DEPENDENCY_EXTRAS.md`
- `docs/VALIDATION_MATRIX.md`
- `docs/WORKSTATION_ACCEPTANCE_CHECKLIST.md`
- `docs/VEGA_ACCEPTANCE_COMMAND.md`
- `docs/VEGA_ACCEPTANCE_CHECKLIST.md`
- `docs/VEGA_BOOTSTRAP.md`
- `docs/VEGA_VALIDATION_MATRIX.md`
- `docs/VALIDATION_CONFIGS.md`
- `docs/HPC_GPU_BATCHED_REDUCED_INDENTATION.md`
- `docs/SURROGATE_MODEL_SELECTION.md`
- `docs/PROPAGATION_EXECUTION.md`
- `examples/reports/workstation_acceptance_report.example.json`
- `examples/reports/release_evidence_manifest.example.json`

## Governance

- `CONTRIBUTING.md` defines contribution and provenance expectations for the public release line.
- `SECURITY.md` defines the supported version line and the private-first vulnerability reporting path.
- `.github/CODEOWNERS` and `.github/dependabot.yml` keep review ownership and dependency maintenance explicit.

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
