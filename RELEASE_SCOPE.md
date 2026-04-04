# MesoUQ Release Scope

MesoUQ is the standalone software release line for Bayesian uncertainty quantification and calibration of mesoscopic DPD models.

## Approved bootstrap scope

The repository is populated only by copying content into `BrieucB/MesoUQ`.
No source repository is modified during the bootstrap.

Primary source repository:

- `BrieucB/UQ_DPD`

Primary source branch:

- `vega/gpu-batching`

Additional source reconciliation:

- manually incorporate the 2 commits currently present on `master` but absent from `vega/gpu-batching`

## Layout policy

Release-grade clean layout from day 1.

- root metadata at repository root
- Python package under `src/meso_uq/`
- workflow modules at repository root under descriptive directories
- imported Korali backend under `extern/korali/`

## Content policy

Included in scope:

- release-grade code
- configs
- documentation
- tests
- directly required runnable data and credibility assets
- Korali backend code and related install and documentation content

Excluded from scope:

- `_paper/`
- `markdowns/`
- local caches and local IDE state
- logs and transient run state
- any material that explicitly shows AI agent contribution
