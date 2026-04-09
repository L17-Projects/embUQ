# Vega Reduced-Model Indentation Workflow

This guide describes the current supported operator path for the reduced-model indentation workflow on Vega.

## Scope

- clone-local runtime rooted under `_vega/`
- vendored `extern/korali/` bootstrap only
- single-node SLURM jobs on the Vega `dev` partition
- one GPU for surrogate-backed `Phase 1` and `Phase 3b`
- workflow path:
  - `Phase 1`
  - `Phase 2`
  - `Phase 3b`
  - propagation `Phase 3b`
  - MAP extraction and plotting

## Backend expectations

For this workflow, the current documented backend contract is:

- `Phase 1`: GPU-batched surrogate path
- `Phase 2`: CPU MPI using the current documented Korali state
- `Phase 3b`: GPU-batched surrogate path
- propagation, MAP extraction, and plotting: CPU-side postprocess work

This is why the supported bootstrap path builds vendored Korali with MPI support, but does not claim a validated native-CUDA `Phase 2` lane.

## Bootstrap the repo-local runtime

From a fresh clone on Vega:

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
python -m pip install -U pip
pip install -e ".[test,mpi]"
pip install pybind11 meson ninja

bash scripts/vega/bootstrap_korali.sh --jobs 8
source _vega/korali/env.sh
python scripts/vega/doctor_vega.py --strict
```

For more detail on the bootstrap path, see [VEGA_BOOTSTRAP.md](VEGA_BOOTSTRAP.md).

## Recommended execution path

Use the explicit Vega helper surface with the reduced-model indentation production config.

Phase 1:

```bash
python scripts/vega/run_inference_stage.py \
  --experiment indentation \
  --model-family reduced-model \
  --profile production \
  --stage phase1
```

Phase 2:

```bash
python scripts/vega/run_inference_stage.py \
  --experiment indentation \
  --model-family reduced-model \
  --profile production \
  --stage phase2 \
  --cpu-ranks 4
```

Phase 3b:

```bash
python scripts/vega/run_inference_stage.py \
  --experiment indentation \
  --model-family reduced-model \
  --profile production \
  --stage phase3b
```

Propagation `Phase 3b`:

```bash
python scripts/vega/run_propagation.py \
  --experiment indentation \
  --model-family reduced-model \
  --profile production \
  --stage phase3b
```

MAP extraction:

```bash
python scripts/vega/extract_map.py \
  --experiment indentation \
  --model-family reduced-model \
  --profile production \
  --stage phase3b
```

## sbatch helpers

The checked-in Vega templates under `scripts/vega/sbatch/` are the preferred batch entrypoints.

For reduced-model indentation, set:

- `EXPERIMENT=indentation`
- `MODEL_FAMILY=reduced-model`
- `PROFILE=production`

Relevant templates:

- `workflow_phase1_to_3b.sbatch`
- `workflow_propagation.sbatch`
- `workflow_map.sbatch`

## Output locations

By default, the reduced-model indentation workflow lands under:

```text
_vega/runs/indentation/reduced-model/production/
```

That tree then contains:

- `results_phase_1/`
- `results_phase_2/`
- `results_phase_3b/`
- `propagation_phase3b/`
- `map_phase3b/`

## Troubleshooting

### `import korali` fails

Make sure `_vega/korali/env.sh` is sourced in the current shell or batch job.

### `mpi4py` or MPI launcher errors

Rebuild or reinstall `mpi4py` after loading the same MPI module stack used for the Korali bootstrap.

### GPU utilization looks low during `Phase 2`

That is expected. `Phase 2` is currently a CPU MPI stage in the documented public workflow.

### GPU utilization looks bursty during `Phase 1` or `Phase 3b`

That is expected. The surrogate work is GPU-batched, but orchestration and postprocess steps still happen on the CPU.
