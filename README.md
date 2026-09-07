# embUQ
[![CI](https://github.com/L17-Projects/embUQ/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/L17-Projects/embUQ/actions/workflows/ci.yml)
[![Release Smoke](https://github.com/L17-Projects/embUQ/actions/workflows/release-smoke.yml/badge.svg?branch=main)](https://github.com/L17-Projects/embUQ/actions/workflows/release-smoke.yml)
[![codecov](https://codecov.io/github/L17-Projects/embUQ/graph/badge.svg)](https://codecov.io/github/L17-Projects/embUQ)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](./pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-included-blueviolet.svg)](./docs/)
[![Linux](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#)
[![MPI](https://img.shields.io/badge/MPI-supported-orange.svg)](#)
[![CUDA](https://img.shields.io/badge/CUDA-optional-green.svg)](#)
[![SLURM](https://img.shields.io/badge/SLURM-supported-blue.svg)](#)
[![DPD](https://img.shields.io/badge/model-DPD-informational.svg)](#)
[![Hierarchical Bayes](https://img.shields.io/badge/inference-Hierarchical%20Bayes-purple.svg)](#)

embUQ is a production repository for Bayesian uncertainty quantification and calibration of elastic-microbubble (EMB) mesoscopic DPD models and for reproducing the associated paper.

The repository separates reusable package code from numerical-experiment assets. The installable package under `src/meso_uq` owns contracts, registries, config schemas, orchestration helpers, surrogate/inference APIs, validation helpers, reporting helpers, and platform-neutral runtime utilities. The `emb/` tree owns the Mirheo, evalkit, and surrogate assets specific to elastic microbubbles.

The release workflow surface covers EMB compression and indentation calibration, hierarchical inference, surrogate retraining and model selection, MAP extraction, plotting, propagation, a public validation matrix, HPC acceptance, and supporting operator utilities.

The paper uses the committed deterministic DNN (`*_BEST.pkl`) surrogate artifacts. Optional Bayesian neural-network (BNN/Pyro) training and evaluation support is retained as a tested production capability.

## Architecture split

The main source and asset layers are:

- `src/meso_uq/`: reusable package layer. Put shared contracts, registries, package APIs, validation logic, config helpers, orchestration helpers, platform/runtime helpers, and cross-experiment surrogate/inference code here.
- `emb/`: elastic microbubble numerical experiment assets. The canonical layout is `emb/compression/{src,evalkit,surrogate}` and `emb/indentation/{src,evalkit,surrogate}`.
- `configs/`: schema-versioned composition examples for agents, modalities, datasets, surrogates, inference settings, noise, platforms, reports, and artifacts. Current operator-facing EMB production and validation configs still live under `inference/configs/` and `reduced/configs/`.
- `scripts/`, `inference/`, `reduced/`, `propagation/`, `sampling/`: maintained workflow and compatibility entrypoints that call package helpers and experiment assets.

Use `src/meso_uq` for reusable behavior that should be importable and tested independently of one physical experiment. Use `emb/` for experiment-local Mirheo source templates, evalkit reference data/helpers, surrogate training/evaluation entrypoints, small curated fixtures, and asset descriptors tied to EMB.

## Validation split

The release validation contract is intentionally split across two environments:

- GitHub CI is the fast PR gate. It proves package/tests/docs health, one real CPU micro workflow lane, and one real public surrogate retraining smoke.
- Vega and Karolina are the HPC proof surfaces for broader validation matrices, acceptance, production sanity, and hardware-specific backend claims.

The GitHub CI test job also publishes line coverage for the committed Python surface across `src/meso_uq`, `emb/compression`, `emb/indentation`, `inference`, `propagation`, `reduced`, and `scripts`, excluding vendored code, tests, and data-only directories.

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
- public HPC validation matrices that run the real validation workflows
- a richer GPU/operator validation runner
- thin HPC acceptance commands with machine-readable reports
- dedicated tiny validation configs for full-model and reduced-model workflows
- vendored Korali build surface and backend notes
- public smoke tests and release-validation documentation

## Config composition

Config composition is split between a policy/example layer and active runtime configs:

- `configs/agents/*.example.yaml` describes agent families and the modality/config roots they can combine.
- `configs/modalities/*.example.yaml` names a structure and experiment, such as `structure: emb` with `experiment: compression`.
- `configs/datasets/*.example.yaml`, `configs/surrogates/*.example.yaml`, `configs/inference/*.example.yaml`, `configs/noise/*.example.yaml`, and `configs/platforms/*.example.yaml` describe the dataset, surrogate artifact, inference algorithm, observation model, and execution platform selected for a lane.
- `inference/configs/{production,validation}/` and `reduced/configs/{production,validation}/` are the current active EMB runtime config roots used by the full-model and reduced-model workflow scripts.

A lane is therefore identified by structure, modality/experiment, model family, profile, dataset/geometries, surrogate backend/artifacts, inference settings, platform, and output root. New structure-general composition examples belong under `configs/`; current EMB production knob changes belong in the active `inference/configs/` or `reduced/configs/` files that the workflow runners actually load.

## Outputs and local guides

Generated outputs do not belong in source directories. Runtime products, logs, scratch data, generated reports, transient Mirheo state, coverage files, build products, and scheduler outputs belong under configured run roots such as `_runs/...`, external scratch/data roots, or documented paper-data artifact roots. The repository `.gitignore` also excludes transitional generated roots such as `_out`, `_ci`, `_init_compression_*`, `_init_indentation_*`, `out_hierarchical`, `runtime`, and `logs`.

`dir.md` files are local-only placement guides. They are intentionally ignored everywhere by the root `**/dir.md` rule and must not be committed.

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

Merge-ready coverage gate (strict, manual/local):

GitHub CI is intentionally a fast feedback gate. The `full-ci`-labelled path still runs a coverage delta check in CI feedback mode; before marking a PR merge-ready, run the strict local gate:

```bash
python scripts/qa/ci/run_merge_ready_coverage.py \
  --base-ref origin/main \
  --test-command "pytest"
```

If you already have coverage JSON artifacts, run the strict checker directly:

```bash
python scripts/qa/ci/check_coverage_increase.py \
  --base-json path/to/base-coverage.json \
  --head-json path/to/head-coverage.json \
  --strict
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

For fresh Vega and Karolina clones, the supported bootstrap path is documented in `docs/VEGA_BOOTSTRAP.md` and `docs/KAROLINA_FULL_PLATFORM.md` and builds vendored `extern/korali/` into `${MESOUQ_SITE_RUNTIME_ROOT}/korali/install`.

## Quick start

To test the implementation on Vega from a fresh clone, the shortest supported path is:

```bash
module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  OpenMPI/4.1.4-GCC-12.2.0 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0

export MESOUQ_SITE=vega
export MESOUQ_SITE_RUNTIME_ROOT="${PWD}/_vega"
bash scripts/platforms/hpc/bootstrap_env.sh --site vega
source scripts/platforms/hpc/site_env.sh
mesouq_activate_site_env vega "$PWD"
python -m pip install --upgrade pip
pip install -e ".[test,mpi]"
pip install pybind11 meson ninja

python scripts/platforms/hpc/doctor_hpc.py --site vega
bash scripts/platforms/hpc/bootstrap_korali.sh --site vega --jobs 8
mesouq_activate_site_env vega "$PWD"
python scripts/platforms/hpc/doctor_hpc.py --site vega --strict

REPO_ROOT=$(pwd) sbatch scripts/platforms/vega/sbatch/validation_matrix.sbatch
```

That submission runs the public validation workflow matrix on the Vega `dev` partition and writes the machine-readable report under `_runs/vega/validation_matrix/<run-tag>/workflow_matrix_report.json`.

If you are already inside an allocated Vega job, replace the last line with:

```bash
python scripts/platforms/hpc/run_validation_matrix.py \
  --site vega \
  --experiments compression indentation \
  --model-families full-model reduced-model \
  --output-root _runs/vega/validation_matrix \
  --phase2-cpu-ranks 4
```

## GitHub CI real canaries

The default GitHub CI lane now includes two real canaries:

- one CPU workflow lane for `compression:reduced-model:validation`, driven by `reduced/configs/ci/ci_canary_config_compression.yaml`
- one public surrogate retraining smoke, driven by `emb/compression/surrogate/ci/retraining_smoke.yaml`

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
python scripts/platforms/hpc/run_validation_matrix.py \
  --site vega \
  --output-root _runs/vega/validation_matrix \
  --phase2-cpu-ranks 4
```

Tracked `sbatch` template:

```bash
REPO_ROOT=$(pwd) sbatch scripts/platforms/vega/sbatch/validation_matrix.sbatch
```

The machine-readable report is written to `_runs/vega/validation_matrix/workflow_matrix_report.json` when you pass the explicit root above, or under `_runs/vega/validation_matrix/<run-tag>/` when you use the sbatch default. Detailed usage and artifact layout are documented in `docs/VEGA_VALIDATION_MATRIX.md`.

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

## Inference lane walkthrough

The best-supported small lane is `compression:reduced-model:validation`, also used by the GitHub CI real canary through `reduced/configs/ci/ci_canary_config_compression.yaml`.

Conceptually, this lane runs as follows:

1. The operator selects the reduced compression validation/canary config through a workflow runner such as `reduced/scripts/run_phase_1.py` or the platform matrix wrapper.
2. The reduced wrapper delegates to `inference/scripts/run_phase_1.py` with the selected config and output root.
3. The phase driver loads the YAML config, resolves enabled experiments with `meso_uq.experiments.load_experiments`, and normalizes the EMB diameter selections to geometry IDs such as `diameter_2.1um`.
4. Package registries and contracts provide the shared vocabulary: `meso_uq.agents.registry` declares EMB support for `compression` and `indentation`, `meso_uq.modalities.registry` declares the compression modality contract, and `meso_uq.surrogate.catalogs`/`emb_catalog` record the canonical EMB surrogate artifact locations under `emb/compression/surrogate/diameters/.../trained/`.
5. The EMB evalkit prepares reference data under `emb/compression/evalkit/data`, preloads the selected DNN or BNN surrogate, and wires Korali to `compute_compression_surrogate` or the GPU batch variant from `emb/compression/evalkit/posterior_compression.py`.
6. Korali executes TMCMC for Phase 1 and writes phase outputs below the selected output root, for example `_runs/<site>/runs/<run-tag>/compression/reduced-model/validation/results_phase_1/...` when using the platform default root.
7. Downstream validation/reporting wrappers read those artifacts for MAP extraction, propagation, plotting, lane manifests, and workflow-matrix reports. Reports and logs stay in the run root rather than in `src/meso_uq` or `emb/`.

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
embUQ/
├── README.md
├── pyproject.toml
├── src/meso_uq/
├── configs/
├── tests/
├── docs/
├── examples/
├── scripts/
├── extern/korali/
├── emb/compression/
├── emb/indentation/
├── inference/
├── propagation/
├── reduced/
└── upstream/
```

## Provenance

The public release line is synchronized from `BrieucB/UQ_DPD`, with explicit upstream-tracking material preserved in the repository history and release hardening work captured through the PR sequence leading to `v0.1.0`.
