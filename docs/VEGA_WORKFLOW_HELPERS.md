# Vega workflow helpers

This page documents the repo-managed Vega helpers added for fresh-clone workflow execution.

For the public Vega validation matrix and its machine-readable report, see `VEGA_VALIDATION_MATRIX.md`.

## Selection axes

The public Vega helper surface separates four concerns explicitly:

- experiment: `compression` or `indentation`
- model family:
  - `full-model`
  - `reduced-model` where `b1=b2=a3=a4=0`
- run profile:
  - `production`
  - `validation` for reduced-cost smoke and acceptance work
- stage:
  - inference: `phase1`, `phase2`, `phase3b`
  - propagation: `phase1`, `phase3b`
  - MAP extraction: `phase1`, `phase3b`

This is intentional. `reduced-model` is a scientific/model-family choice. `validation` is an execution-profile choice. They must not be collapsed into one `reduced` label.

## Inference wrapper

Run one public inference stage with explicit workflow selection:

```bash
python scripts/vega/run_inference_stage.py \
  --experiment compression \
  --model-family full-model \
  --profile validation \
  --stage phase1
```

For phase 2, MPI ranks can be requested explicitly:

```bash
python scripts/vega/run_inference_stage.py \
  --experiment compression \
  --model-family full-model \
  --profile validation \
  --stage phase2 \
  --cpu-ranks 4
```

Default outputs land under:

```text
_vega/runs/<experiment>/<model-family>/<profile>/
```

## Propagation wrapper

Run the public lightweight propagation layer with the same explicit selection:

```bash
python scripts/vega/run_propagation.py \
  --experiment compression \
  --model-family reduced-model \
  --profile production \
  --stage phase3b
```

## Generic MAP wrapper

Extract MAP samples from either Phase 1 or Phase 3b without selecting run directories manually:

```bash
python scripts/vega/extract_map.py \
  --experiment compression \
  --model-family full-model \
  --profile validation \
  --stage phase3b
```

Optional selectors:

- `--dataset compression_2.1um`
- `--diameter 2.1`
- `--output /abs/path/map.csv` for a single selected dataset

The wrapper writes:

- per-dataset MAP CSV files
- one machine-readable manifest:
  - `map_phase1/phase1_map_manifest.json`
  - or `map_phase3b/phase3b_map_manifest.json`

That manifest is the intended reuse handoff for later plotting, reporting, or operator scripts.

## sbatch templates

The following canned templates live under `scripts/vega/sbatch/`:

- `workflow_phase1_to_3b.sbatch`
- `workflow_propagation.sbatch`
- `workflow_map.sbatch`
- `validation_matrix.sbatch`
- `production/complete_inference_compression.sbatch`
- `production/complete_inference_indentation.sbatch`
- `production/complete_reduced_compression.sbatch`
- `production/complete_reduced_indentation.sbatch`
- `production/phase1_gpu.sbatch`
- `production/phase2_cpu.sbatch`
- `production/phase3b_gpu.sbatch`
- `production/propagation_phase3b.sbatch`
- `production/validation_phase1_to_3b.sbatch`
- `production/validation_propagation.sbatch`

They assume:

- the repo-local bootstrap from `docs/VEGA_BOOTSTRAP.md` is already complete
- `_vega/venv` exists
- `_vega/korali/env.sh` exists

Each template exposes `EXPERIMENT`, `MODEL_FAMILY`, and `PROFILE` at the shell-variable level so fresh-clone workflow jobs do not rely on editing Python code or guessing config paths.

Top-level templates target the Vega `dev` partition for smoke/acceptance usage.
The `production/complete_*` templates orchestrate multi-node production lanes by submitting child
phase jobs (`phase1 -> phase2 -> phase3b -> propagation phase3b`) with explicit partition and memory
settings.

The production complete scripts require passing `REPO_ROOT` at submission time:

```bash
REPO_ROOT=$(pwd) sbatch scripts/vega/sbatch/production/complete_reduced_compression.sbatch
```

Useful production overrides:
- `GPU_PARTITION=dev` to run GPU phases on `dev`
- `PHASE3B_MEM_ARG="--mem=8000"` when `--exclusive` is too strict on `dev`
- `RUN_TAG=<tag>` to control output/log folder naming

When submitting with `sbatch`, run them from the repo root or set `REPO_ROOT` explicitly so the batch job can resolve the clone-local `_vega/` runtime correctly.
