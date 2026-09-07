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

`embUQ` is the reference codebase for the encapsulated microbubble
uncertainty-quantification paper and its frozen `review2_v1` submission. It
contains the production HBI, surrogate, direct-DPD, PCA, figure, and manuscript
replay code for SonoVue and Definity on Karolina and Vega.

The immutable submitted manuscript and figures are tracked under
[`papers/UQ_EMB/editor_submission/review2_v1`](papers/UQ_EMB/editor_submission/review2_v1).
Large scientific artifacts stay outside Git and are bound to the code by the
checksum manifests in [`papers/UQ_EMB/manifests`](papers/UQ_EMB/manifests).

## Install and test

Python 3.10+ is required. On Karolina, create environments with Python 3.11:

```bash
/usr/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest -q
```

Production runs additionally require the documented HPC environment. Korali is
bootstrapped from `extern/korali`; Mirheo is an external dependency selected by
`MESOUQ_MIRHEO_SRC`. CUDA, OpenMPI, and Slurm are required for DPD production.
See the [Karolina](docs/KAROLINA_FULL_PLATFORM.md) and
[Vega](docs/VEGA_BOOTSTRAP.md) setup guides.

## Reproduction data layout

Set `PAPER_BUNDLE_ROOT` to the recovered reproduction bundle and use its current
artifact directory as `MESOUQ_UQ_EMB_ARTIFACT_ROOT`:

```text
${PAPER_BUNDLE_ROOT}/
├── scientific/current/artifacts/
│   ├── accepted_production_outputs_202607/
│   ├── frozen_runtime_dependencies_202607/
│   ├── frozen_plotting_dependencies_202607/
│   ├── frozen_legacy_paper_runtime_complete_202606/
│   └── frozen_tinytex_runtime_202606/
├── recovery-v2/vega_pca_table_s7/latest_map_campaign/
└── recovery-v4/vega_pca_table_s7/accepted_complete_pca/

${MESOUQ_SCRATCH_ROOT}/papers/UQ_EMB/replay/   # new outputs only
```

The five artifact directories are verified against the tracked manifests. The
two PCA recovery layers jointly contain the compact case metadata and the six
complete trajectories needed to regenerate supplementary Table S7. Never write
new results into the recovered bundle.

## What the code regenerates

| Paper product | Maintained entrypoint |
| --- | --- |
| SonoVue and Definity HBI states | `materialize_hbi_config.py`, then `run_hbi_replay.py` |
| Mechanical/acoustic direct-DPD checks | `materialize_direct_dpd_replay.py` and `verify_direct_dpd_replay_plan.py`; execution requires Mirheo |
| Accepted acoustic polynomial surrogates | `replay_acoustic_polynomial_surrogates.py` |
| Table S7 PCA values | [`pca_table_s7/run_replay.py`](scripts/workflows/emb/uq_emb/pca_table_s7/run_replay.py) |
| Figures 6, 7, and 9 | `render_figure6_replay.py`, `render_figure7_replay.py`, and `render_figure9_replay.py` |

Exact commands, safety guards, and output receipts are documented in the
[`review2_v1` reproduction guide](papers/UQ_EMB/README.md) and the
[Table S7 PCA guide](papers/UQ_EMB/pca_table_s7/README.md).

The paper consumes deterministic `*_BEST.pkl` DNN surrogates. Their accepted
weights and training tables are checksum-verifiable, but byte-identical
retraining is not claimed because the original architecture-sweep seeds and
receipts were not preserved. BNN/Pyro support is optional and is not used by
the paper.

Raw observations and immutable accepted baselines cannot be recreated from
source code alone; they must first be restored in the layout above. Given those
inputs and the external HPC dependencies, the derived paper datasets can be
regenerated with this repository.

## Citation

Please cite the paper and the software metadata in [`CITATION.cff`](CITATION.cff).
Released under the [MIT License](LICENSE).
