# Workflows

This document summarizes the public workflow surfaces currently exposed by `MesoUQ`.

## 1. Full-model hierarchical inference

The full workflow entrypoints live under `inference/scripts/`:
- `run_phase_1.py`
- `run_phase_2.py`
- `run_phase_3a.py`
- `run_phase_3b.py`

Canonical configs live under:
- `inference/configs/production/inference_config_compression.yaml`
- `inference/configs/production/inference_config_indentation.yaml`

Typical order:
1. prepare or verify the relevant surrogate/data surface
2. run phase 1 for per-dataset posterior inference
3. run phase 2 for the hierarchical `Psi` stage
4. run phase 3a / 3b for joint or per-dataset follow-up stages
5. extract MAP samples and generate validation/posterior plots

## 2. Reduced-model hierarchical inference

The reduced-model workflow is exposed through:
- `reduced/scripts/run_phase_1.py`
- `reduced/scripts/run_phase_2.py`
- `reduced/scripts/run_phase_3a.py`
- `reduced/scripts/run_phase_3b.py`

Canonical reduced configs live under:
- `reduced/configs/production/reduced_config_compression.yaml`
- `reduced/configs/production/reduced_config_indentation.yaml`

These wrappers delegate to the main workflow spine while selecting the reduced-model configs by default.

The former single-dataset Phase 3b helper is no longer part of the public script surface. It remains available only as an internal helper for narrowly scoped operator/debug work.

## 3. Execution profiles are separate from model family

The repository also ships reduced-cost validation configs under both the full-model and reduced-model trees:

- full-model validation:
  - `inference/configs/validation/validation_config_compression.yaml`
  - `inference/configs/validation/validation_config_indentation.yaml`
- reduced-model validation:
  - `reduced/configs/validation/validation_config_compression.yaml`
  - `reduced/configs/validation/validation_config_indentation.yaml`

These validation configs are execution-profile choices for smoke and acceptance work. They are not the same concept as the reduced-model scientific surface.

## 4. Surrogate retraining and evaluation

Compression:
- evaluator: `compression/surrogate/evaluate.py`
- training entrypoint: `compression/surrogate/scripts/emb_train.py`

Indentation:
- evaluator: `indentation/surrogate/evaluate.py`
- training entrypoint: `indentation/surrogate/scripts/emb_train.py`

Shared training logic is implemented once in `src/meso_uq/surrogate/`.

## 5. Sensitivity and lightweight design generation

Compression Sobol sensitivity:
- `compression/surrogate/sensitivity/scripts/run_sobol_vs_disp.py`

Indentation Sobol sensitivity:
- `indentation/surrogate/sensitivity/scripts/run_sobol_vs_force.py`

Lightweight design generation:
- `sampling/run_LHS.py`

This surface is intentionally focused on analysis/design support and does not yet include the heavier Mirheo execution layer.

## 6. MAP extraction and plotting

MAP extraction:
- `inference/scripts/extract_phase1_map.py`
- `inference/scripts/extract_phase3b_map.py`

Plotting/postprocessing:
- `propagation/scripts/plot_validation_overlay.py`
- `propagation/scripts/plot_d0_correlations.py`
- `propagation/scripts/plot_posterior_marginals.py`

Shared postprocessing helpers live under `src/meso_uq/postprocess/`.

## 7. Vega helper surface

For Vega-first operation, the repo now also ships split helpers under `scripts/vega/`:

- `run_inference_stage.py`
- `run_propagation.py`
- `extract_map.py`
- `sbatch/*.sbatch`

These helpers expose experiment, model family, run profile, and stage explicitly so the operator surface does not overload the word `reduced`.

## 8. Vendored Korali patch surface

The vendored backend under `extern/korali/` is intentionally focused. It contains the public patch surface needed by the current release line, not a full indiscriminate vendor dump.

In particular, the public line now includes focused execution-level slices for:
- `Hierarchical/Theta`
- `TMCMC`
- `Hierarchical/Psi`

## Suggested usage pattern for new users

A good order for an outside user is:
1. understand the config and dataset surface with `list_experiment_datasets.py`
2. retrain or inspect the surrogate surface
3. run reduced or full inference depending on the goal
4. extract MAP samples
5. generate validation and posterior plots
