# Vega validation matrix

This page documents the public Vega validation-matrix surface.

GitHub CI now carries a smaller real-canary layer for one CPU workflow lane and one surrogate retraining smoke. This Vega matrix remains the broader proof surface and should not be collapsed into the GitHub-hosted merge gate.

## Goal

The validation matrix proves that a clean clone can execute the shipped public validation workflows without manual path repair.

It keeps the scientific/workflow axes explicit:

- experiment: `compression` or `indentation`
- model family:
  - `full-model`
  - `reduced-model`

The public validation command fixes the execution profile to `validation`. This separation is intentional. `reduced-model` is not the same concept as a `validation` profile.

## Default validation matrix

The default matrix is the public validation path:

- experiments: `compression indentation`
- model families: `full-model reduced-model`
- profiles: `validation`

That exercises the full-model and reduced-model codepaths while keeping the workflow cost small enough for fresh-clone validation work.

## Public command

Run the matrix directly inside an allocated Vega job:

```bash
python scripts/platforms/hpc/run_validation_matrix.py \
  --site vega \
  --experiments compression indentation \
  --model-families full-model reduced-model \
  --output-root _runs/vega/validation_matrix \
  --phase2-cpu-ranks 4
```

The public command always fixes the execution profile to `validation`.

It delegates to the lower-level `scripts/platforms/hpc/run_workflow_matrix.py` operator runner, which remains available for broader matrix/debugging use. The Vega site wrapper remains available at `scripts/platforms/vega/run_validation_matrix.py` for compatibility and injects `--site vega`.

Each selection runs:

- Phase 1
- MAP extraction from Phase 1 outputs
- Phase 2
- Phase 3b
- Phase 3b propagation
- MAP extraction from Phase 3b outputs

## Machine-readable outputs

The command writes:

- one top-level report:
  - `_runs/vega/validation_matrix/workflow_matrix_report.json`
- one summary per selection:
  - `_runs/vega/validation_matrix/summaries/<experiment>__<model-family>__<profile>.json`
- captured stdout/stderr logs for every step:
  - `_runs/vega/validation_matrix/logs/...`
- nested workflow outputs:
  - `_runs/vega/validation_matrix/runs/<experiment>/<model-family>/<profile>/`

## sbatch template

The canned template is:

- `scripts/platforms/vega/sbatch/validation_matrix.sbatch`

It targets the Vega `dev` partition and therefore keeps the wall clock at 30 minutes or less. For longer non-smoke operator runs, copy the template and adjust the partition/time budget explicitly.

Submit the template from the repo root, or set `REPO_ROOT=/abs/path/to/clone` explicitly when calling `sbatch`.

It assumes:

- `${MESOUQ_SITE_RUNTIME_ROOT}/env` exists
- `${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh` exists
- the recommended Vega modules are available

The template exposes:

- `EXPERIMENTS`
- `MODEL_FAMILIES`
- `PROFILE=validation`
- `OUTPUT_ROOT`
- `PHASE2_CPU_RANKS`

## Config overrides

For targeted debugging, a selection-specific config can be injected with:

```bash
python scripts/platforms/hpc/run_validation_matrix.py \
  --site vega \
  --selection compression:full-model:validation \
  --output-root _runs/vega/validation_matrix/custom_debug \
  --config-override compression:full-model:validation=/abs/path/config.yaml
```

### Output-root policy

For active runtime outputs, route artifacts under canonical `_runs/...` trees (or the site scratch override configured by the caller).
For durable paper-facing campaign outputs, use `paper_data/...`.

If an explicit `--output-root` does not resolve into one of those trees, this command fails fast.
