# Vega workflow matrix

This page documents the fresh-clone workflow-matrix surface for Vega.

## Goal

The matrix runner proves that a clean clone can execute the shipped public workflow stages without manual path repair.

It keeps these axes explicit:

- experiment: `compression` or `indentation`
- model family:
  - `full-model`
  - `reduced-model`
- execution profile:
  - `validation`
  - `production`

This separation is intentional. `reduced-model` is not the same concept as a `validation` profile.

## Default smoke matrix

The default matrix is the reduced-cost smoke path:

- experiments: `compression indentation`
- model families: `full-model reduced-model`
- profiles: `validation`

That exercises the full-model and reduced-model codepaths while keeping the workflow cost small enough for fresh-clone validation work.

## Runner

Run the matrix directly inside an allocated Vega job:

```bash
python scripts/vega/run_workflow_matrix.py \
  --experiments compression indentation \
  --model-families full-model reduced-model \
  --profiles validation \
  --output-root _vega/workflow_matrix/validation_smoke \
  --phase2-cpu-ranks 4
```

Each selection runs:

- Phase 1
- MAP extraction from Phase 1 outputs
- Phase 2
- Phase 3b
- Phase 3b propagation
- MAP extraction from Phase 3b outputs

## Machine-readable outputs

The runner writes:

- one top-level report:
  - `_vega/workflow_matrix/<label>/workflow_matrix_report.json`
- one summary per selection:
  - `_vega/workflow_matrix/<label>/summaries/<experiment>__<model-family>__<profile>.json`
- captured stdout/stderr logs for every step:
  - `_vega/workflow_matrix/<label>/logs/...`
- nested workflow outputs:
  - `_vega/workflow_matrix/<label>/runs/<experiment>/<model-family>/<profile>/`

## sbatch template

The canned template is:

- `scripts/vega/sbatch/workflow_matrix_smoke.sbatch`

It targets the Vega `dev` partition and therefore keeps the wall clock at 30 minutes or less. For longer non-smoke operator runs, copy the template and adjust the partition/time budget explicitly.

Submit the template from the repo root, or set `REPO_ROOT=/abs/path/to/clone` explicitly when calling `sbatch`.

It assumes:

- `_vega/venv` exists
- `_vega/korali/env.sh` exists
- the recommended Vega modules are available

The template exposes:

- `EXPERIMENTS`
- `MODEL_FAMILIES`
- `PROFILES`
- `OUTPUT_ROOT`
- `PHASE2_CPU_RANKS`

## Config overrides

For targeted debugging, a selection-specific config can be injected with:

```bash
python scripts/vega/run_workflow_matrix.py \
  --selection compression:full-model:validation \
  --config-override compression:full-model:validation=/abs/path/config.yaml
```

Legacy aliases such as `compression_full` are accepted for the override selector, but the matrix report always records the explicit `experiment:model-family:profile` selection.
