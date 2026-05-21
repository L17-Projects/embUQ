# Workflows

This document summarizes the public workflow surfaces currently exposed by `MesoUQ`.

## 1. Full-model hierarchical inference

The supported full-model workflow entrypoints live under `inference/scripts/`:
- `run_phase_1.py`
- `run_phase_2.py`
- `run_phase_3b.py`

Canonical configs live under:
- `inference/configs/production/inference_config_compression.yaml`
- `inference/configs/production/inference_config_indentation.yaml`

Typical order:
1. prepare or verify the relevant surrogate/data surface
2. run phase 1 for per-dataset posterior inference
3. run phase 2 for the hierarchical `Psi` stage
4. run phase 3b for the per-dataset follow-up stage
5. extract MAP samples and generate validation/posterior plots

## 2. Reduced-model hierarchical inference

The supported reduced-model workflow is exposed through:
- `reduced/scripts/run_phase_1.py`
- `reduced/scripts/run_phase_2.py`
- `reduced/scripts/run_phase_3b.py`

Canonical reduced configs live under:
- `reduced/configs/production/reduced_config_compression.yaml`
- `reduced/configs/production/reduced_config_indentation.yaml`

These wrappers delegate to the main workflow spine while selecting the reduced-model configs by default.

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
- evaluator: `emb/compression/surrogate/evaluate.py`
- lightweight training entrypoint: `emb/compression/surrogate/scripts/emb_train.py`
- paper-facing 12-architecture sweep + BEST promotion: `emb/compression/surrogate/scripts/train_multi_arch.py`

Indentation:
- evaluator: `emb/indentation/surrogate/evaluate.py`
- lightweight training entrypoint: `emb/indentation/surrogate/scripts/emb_train.py`
- paper-facing 12-architecture sweep + BEST promotion: `emb/indentation/surrogate/scripts/train_multi_arch.py`

Cross-platform HPC DNN rebuild matrix:
- `python scripts/platforms/hpc/run_dnn_surrogate_training.py --site vega|karolina`
- Vega compatibility batch template: `scripts/platforms/vega/sbatch/train_dnn_surrogates.sbatch`

Cross-platform surrogate group-holdout matrix:
- `python scripts/platforms/hpc/run_surrogate_group_holdout.py --site vega|karolina`
- Site sbatch templates: `scripts/platforms/{vega,karolina}/sbatch/surrogate_group_holdout.sbatch`
- The site templates expose the shared seed, split, predictive-MC, device, indentation-loader,
  and model-selection knobs. Karolina writes to `${MESOUQ_RUNS_ROOT}/surrogate_group_holdout/<run-tag>`
  by default and uses `#SBATCH --gpus=1` on the `qgpu` partition.

The DNN matrix writes one per-spec provenance manifest under:
- `<output-root>/<spec>/dnn_training_manifest.json`

Each per-spec manifest records:
- the training config used for that diameter
- the SLURM array job id
- submit / array / finalize logs
- the promoted `*_BEST.pkl` artifact path and checksum
- the final training report path and checksum when present

Shared training logic is implemented once in `src/meso_uq/surrogate/`.

## 5. Sensitivity and lightweight design generation

Compression Sobol sensitivity:
- `emb/compression/surrogate/sensitivity/scripts/run_sobol_vs_disp.py`

Indentation Sobol sensitivity:
- `emb/indentation/surrogate/sensitivity/scripts/run_sobol_vs_force.py`

Lightweight design generation:
- `sampling/run_LHS.py`

This surface also feeds the MAP Mirheo execution layer used by the paper-facing workflow.

## 6. MAP extraction and plotting

MAP extraction:
- `scripts/shared/postprocess/extract_phase1_map.py`
- `scripts/shared/postprocess/extract_phase3b_map.py`
- `scripts/platforms/hpc/run_map_mirheo.py --site vega|karolina`

Plotting/postprocessing:
- `propagation/scripts/plot_validation_overlay.py`
- `propagation/scripts/plot_d0_correlations.py`
- `propagation/scripts/plot_posterior_marginals.py`
- `scripts/shared/postprocess/generate_map_overlay_figure.py`

Shared postprocessing helpers live under `src/meso_uq/postprocess/`.

Phase 1 posterior-figure policy:
- keep posterior figures based on the raw posterior sample database (no duplicate filtering)
- compute and report duplicate-particle diagnostics (`unique_particle_count`, `top_duplicate_mass`, `top_10_duplicate_mass`)
- compute and report chain-leader multiplicity diagnostics from Korali `Chain Leaders`
- include per-lane comparison between chain-leader and posterior-sample duplication mass in machine-readable diagnostics outputs

## 7. Cross-platform HPC helper surface

The shared operator implementations live under `scripts/platforms/hpc/` and select the cluster with `--site vega|karolina`:

- `run_validation_suite.py`
- `run_inference_stage.py`
- `run_propagation.py`
- `extract_map.py`
- `run_workflow_matrix.py`
- `run_validation_matrix.py`

The `scripts/platforms/vega/` and `scripts/platforms/karolina/` Python entrypoints are strict compatibility wrappers around the shared HPC implementations. Site-specific sbatch templates remain under each site directory when scheduler directives or module setup differ.

These helpers expose experiment, model family, run profile, and stage explicitly so the operator surface does not overload the word `reduced`.

The production complete scripts orchestrate:
1. Phase 1 on GPU (GPU-batched surrogate, Sequential Korali conduit)
2. Phase 2 with `phase2_backend=native-cuda` by default for `production` lanes
3. Phase 2 fallback on CPU MPI remains available through `--phase2-backend cpu-mpi`
3. Phase 3b on exclusive GPU (GPU-batched surrogate, Sequential Korali conduit)
4. Propagation phase 3b on GPU in the production wrappers shipped for the HUQ-EMB rebuild path

**Phase 2 backend note:** the public Phase 2 entrypoint now exposes an explicit backend switch:
`--phase2-backend {cpu-mpi,native-cuda}`. The current branch contract sets
`production -> native-cuda` and `validation -> cpu-mpi` by default. Runtime validation of the
native-CUDA production path is tracked separately and must still pass on the target hardware
before it can be treated as operationally proven.

**GPU partition note:** the Vega orchestration now routes GPU jobs by strict walltime policy:
jobs with runtime strictly `<00:30:00` go to `dev`; jobs with runtime `>=00:30:00` go to `gpu`.
That policy also applies to MAP Mirheo jobs.

Default production lanes:
- `scripts/platforms/vega/sbatch/production/complete_inference_compression.sbatch`
- `scripts/platforms/vega/sbatch/production/complete_inference_indentation.sbatch`
- `scripts/platforms/vega/sbatch/production/complete_reduced_compression.sbatch`
- `scripts/platforms/vega/sbatch/production/complete_reduced_indentation.sbatch`

## 8. Vendored Korali patch surface

The vendored backend under `extern/korali/` is intentionally focused. It contains the public patch surface needed by the current release line, not a full indiscriminate vendor dump.

In particular, the public line now includes focused execution-level slices for:
- `Hierarchical/Theta`
- `TMCMC`
- `Hierarchical/Psi`

## 8b. External Mirheo bootstrap surface

Mirheo is not vendored under `extern/`. The current contract is:

- source path lock in `extern/mirheo.lock.json`
- repo-local build/install state under `${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/`
- Python package install into `${MESOUQ_SITE_RUNTIME_ROOT}/env`
- source snapshot manifest at `${MESOUQ_SITE_RUNTIME_ROOT}/mirheo/source_snapshot.json`

Canonical bootstrap entrypoint:
- `scripts/platforms/hpc/bootstrap_mirheo.sh`

Current default source lock:
- `/ceph/hpc/home/eubrieucb/software/Mirheo`

Supported MAP Mirheo micro-canary contract on Vega:
- `n_displacements=1`
- `numsteps=200`
- `numsteps_eq=200`
- strict `<00:30:00 => dev` routing still applies

Important separation:
- the dedicated sanity runner exports the `200/200` micro-canary floor explicitly
- the generic `workflow_map_mirheo.sbatch` wrapper keeps `numsteps` unset by default so production uses the scientific Mirheo payload defaults

Scratch/runtime rule:
- MAP Mirheo must use lane-local scratch under the lane output tree
- do not reuse repo-root `_init_*_map` scratch directories across concurrent full/reduced lanes
- the orchestration path now passes a unique scratch root per dataset under `map_mirheo/_scratch/`
- operators do not pre-create MAP smoke-test init directories; `run_map_mirheo.py` passes `--scratch-root <lane output>/map_mirheo/_scratch/<dataset_name>` to the evaluator
- compression smoke init directories are regenerated from `emb/compression/src` through `generate_sim` and `write_parameters`
- indentation smoke init directories are copied from `emb/indentation/src`, then their `parameter/parameters-default*.yaml` files are rewritten for the selected diameter and retry settings
- missing Phase 3b MAP manifests fail before Slurm submission; missing init templates or Mirheo bootstrap inputs fail the MAP Mirheo lane explicitly and are not skip conditions

Production walltime rule:
- indentation MAP Mirheo currently fits the `00:45:00` production wrapper budget
- compression MAP Mirheo requires a longer production budget in the real 50k runner (`02:00:00`)
- both remain on the `gpu` partition because they are not strict `<00:30:00` jobs
- the generic `workflow_map_mirheo.sbatch` wrapper now defaults to a safer `02:00:00`; shorter callers should override `--time` and `GPU_TIME_LIMIT` explicitly

## 9. Local workstation validation (o369 / non-SLURM)

For local workstation runs (non-SLURM, e.g. `o369`), see `docs/WORKSTATION_LOCAL_WORKFLOWS.md`.

Local validation targets four lanes: `(compression, indentation) × (full-model, reduced-model)`.

Low-load guidance for local runs:
- Use **max 9 CPUs** for Phase 2 only when running the CPU-MPI backend.
- Keep **>2 GB RAM free** at all times.
- Use the `validation` profile (reduced-cost configs), not `production`.
- The local validation runner currently exercises the validation-profile default
  `phase2_backend=cpu-mpi`; native-CUDA Phase 2 validation is tracked separately.

## Suggested usage pattern for new users

A good order for an outside user is:
1. understand the config and dataset surface with `scripts/shared/config/list_experiment_datasets.py`
2. retrain or inspect the surrogate surface
3. run full-model or reduced-model inference depending on the goal
4. extract MAP samples
5. generate validation and posterior plots

## 10. GV extension closeout scope (provisional)

The GV extension closeout slice documents a constrained non-shear rollout:

1. Mirheo execution from canonical GV runtime sources for `stretching`, `buckling`, `torsion`, and `eigenmodes`,
2. DNN surrogate prediction for those generated outputs,
3. hierarchical inference on the resulting lane artifacts.

For the exact parameter and governance constraints, see `GV_EXTENSION_CLOSEOUT.md`.

## 11. HUQ-EMB Vega production rebuild

For the real 50k HUQ-EMB rebuild on Vega, use:
- `scripts/workflows/emb/huq_emb/run_vega_50k_campaign.py`

Do not use `scripts/workflows/emb/huq_emb/run_paper_data_campaign.py` for the Vega full rebuild launch path.
That legacy runner is reserved for postprocess / asset-graph work and will reject the Vega full-rebuild selection set unless `--skip-workflow` is used.

This runner performs, in order:
1. DNN surrogate 12-architecture rebuild for all six diameters
2. four production lanes through `sbatch/production/complete_*.sbatch`
3. MAP extraction (`phase1`, `phase3b`)
4. MAP Mirheo
5. paper asset generation via the postprocess-only paper runner

For exact paper-facing figure rendering on Vega, bootstrap and source repo-local TinyTeX first:
- `bash scripts/platforms/vega/bootstrap_tex.sh`
- `source ${MESOUQ_SITE_RUNTIME_ROOT}/tinytex/env.sh`

For one-command replay of the exact HUQ-EMB paper figures from a stored `paper_data` campaign, use:
- `scripts/workflows/emb/huq_emb/run_exact_uqdpd_asset_port.py`

This wrapper auto-stages the DNN holdout/Sobol inputs, renders the exact paper figures, copies them into `paper_data/figures/*` and `paper_data/tables/`, and writes a replay report under `runs/<campaign_id>/paper_exact_stage/`.
If no TeX deps are configured, it falls back to non-TeX matplotlib rendering automatically. See `docs/HUQ_EMB_EXACT_FIGURE_REPLAY.md`.

## 12. Compatibility wrapper retirement plan

The compatibility wrappers under `scripts/platforms/vega/` and `scripts/platforms/karolina/` keep legacy command shapes alive while delegating to the shared `scripts/platforms/hpc/` implementations with a fixed site. A site wrapper rejects a conflicting `--site`; use the shared HPC entrypoint when selecting a site explicitly.

Current plan:

- keep the delegation wrappers until launcher and documentation migration is complete,
- continue emitting behavior checks that these wrappers preserve the same CLI surface,
- remove deprecated wrappers in a follow-up release only after:
  - a deprecation period has passed,
  - `scripts/platforms/hpc/... --site vega|karolina` launch paths are documented as the primary entrypoint,
  - and this compatibility test matrix is green for at least one full release cycle.

The following public surfaces are guarded by tests:

- `scripts/platforms/hpc/run_validation_matrix.py` forwards to `scripts/platforms/hpc/run_workflow_matrix.py` with the `validation` profile.
- `scripts/platforms/vega/*.py` and `scripts/platforms/karolina/*.py` compatibility wrappers forward to the matching shared HPC implementation and reject conflicting site selectors.
