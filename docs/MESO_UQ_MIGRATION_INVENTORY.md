# MesoUQ migration inventory

This inventory records the current repository surface for the scalable-architecture
migration. It is intentionally factual: current roots first, then owner decisions.

## Top-level areas

| Area | Current role |
|---|---|
| `src/meso_uq` | Shared package API and runtime helpers |
| `compression` | Compression workflow, surrogate, evaluation, and sensitivity code |
| `indentation` | Indentation workflow, surrogate, evaluation, and sensitivity code |
| `inference` | Full-model workflow scripts and configs |
| `reduced` | Reduced-model workflow scripts and configs |
| `sampling` | Lightweight design generation and related helpers |
| `propagation` | Plotting and propagation postprocessing |
| `scripts` | Shared, platform, QA, release, and workflow launchers |
| `ops/huq_emb` | Legacy shim surface for HUQ-EMB launch compatibility |
| `gv` | GV provenance/source staging roots |
| `papers/huq_emb` | Paper-facing replay and asset generation inputs |
| `examples/configs` | Example configs that are refreshed or retired, not treated as stable source of truth |
| `extern/korali` | Vendored backend patch surface |
| `docs` | Public documentation and release evidence |
| `tests` | Repo governance, workflow, and docs sanity checks |

## Current source roots

| Root | Notes |
|---|---|
| `src/meso_uq` | Shared import surface; keep `import meso_uq` stable |
| `compression/src` | Compression-specific runtime and geometry helpers |
| `compression/surrogate` | Compression surrogate training and evaluation |
| `indentation/src` | Indentation-specific runtime and geometry helpers |
| `indentation/surrogate` | Indentation surrogate training and evaluation |
| `inference/scripts` | Full-model phase entrypoints |
| `reduced/scripts` | Reduced-model phase entrypoints |
| `sampling` | Sampling and LHS generation |
| `propagation/scripts` | Propagation and plotting entrypoints |
| `scripts/shared` | Cross-platform helpers |
| `scripts/platforms` | Karolina, Vega, and HPC launcher surfaces |
| `scripts/workflows` | Higher-level workflow orchestration |
| `ops/huq_emb` | Compatibility shim for the HUQ-EMB paper workflow |
| `gv` | GV provenance/source staging tree |
| `extern/korali` | Vendored Korali subtree |
| `papers/huq_emb` | Paper replay assets and generated figure pipelines |

## Config roots

| Root | Notes |
|---|---|
| `inference/configs/production` | Canonical full-model production configs |
| `inference/configs/validation` | Full-model validation configs |
| `reduced/configs/production` | Canonical reduced-model production configs |
| `reduced/configs/validation` | Reduced-model validation configs |
| `examples/configs` | Example and archive candidate configs; refresh or retire as needed |

## Workflow roots

| Root | Notes |
|---|---|
| `scripts/platforms/hpc` | Site-neutral HPC dispatchers |
| `scripts/platforms/karolina` | Karolina-specific launchers, sbatch templates, and validation helpers |
| `scripts/platforms/vega` | Vega-specific launchers, sbatch templates, and validation helpers |
| `scripts/workflows/emb/huq_emb` | HUQ-EMB campaign orchestration |
| `scripts/workflows/gv` | GV workflow orchestration and canary surfaces |
| `scripts/qa` | Documentation and release sanity checks |
| `inference/scripts`, `reduced/scripts` | Public phase runners |
| `scripts/run_vega_acceptance.py` | Thin Vega acceptance wrapper |

## Generated roots

| Root | Current interpretation |
|---|---|
| `_runs` | Canonical run output root for site-aware validation and workflow outputs |
| `_out` | Legacy generated output tree not needed for reproduction |
| `_ci` | CI and coverage artifacts |
| `_init_compression_*` | Generated compression initial conditions |
| `out_hierarchical` | Legacy hierarchical outputs |
| `logs` | Root Slurm logs and similar transient operator output |

## Artifact classes

| Class | Owner decision |
|---|---|
| Release-critical surrogate artifacts | Keep in git only when curated and manifest-backed |
| Raw/reference data | Keep only if small, license-clear, documented, and reproducibility-critical |
| Generated training data | Move out of git |
| Serialized models and pickles | Track via manifests and compatibility tests, not by accidental directory retention |
| Reports and manifests | Keep as the primary reproducibility contract |
| Paper-facing generated artifacts | Keep `papers/` for now, then move heavier outputs out once manifests exist |
| Legacy path aliases | Keep for one release cycle with shims and warnings |

## Owner decisions

- Public import API stays `meso_uq`; CLI/project branding may be `mesouq`.
- `extern/korali` remains vendored.
- GV should stage from manifests, not own permanent `gv/<experiment>/src` state.
- `examples/configs` should be refreshed or archived instead of treated as durable source.
- Legacy imports and launch paths get one release cycle of compatibility shims and warnings.
- Supported platforms remain Karolina, Vega, and workstation.
- Generated training data and non-reproducible outputs should move out of git.
